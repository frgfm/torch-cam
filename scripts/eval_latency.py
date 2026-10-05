# Copyright (C) 2021-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
CAM latency benchmark
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess  # noqa: S404
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.models import get_model, get_model_weights

from torchcam import __version__, methods

if __package__:
    from .cam_example import METHOD_NAMES, build_extractor, nonnegative_int, positive_int
else:
    from cam_example import METHOD_NAMES, build_extractor, nonnegative_int, positive_int


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_sample(model, input_tensor, extractor, requested_class, device, scope, cam_kwargs):
    model.zero_grad(set_to_none=True)
    input_tensor.grad = None

    if scope == "extractor":
        scores = model(input_tensor)
        class_idx = scores.squeeze(0).argmax().item() if requested_class is None else requested_class

    _synchronize(device)
    started_at = time.perf_counter()

    if scope == "end-to-end":
        scores = model(input_tensor)
        class_idx = scores.squeeze(0).argmax().item() if requested_class is None else requested_class

    cams = extractor(class_idx, scores, **cam_kwargs)
    _synchronize(device)
    elapsed = time.perf_counter() - started_at
    return elapsed, cams


def _validate_cams(cams, expected_shapes):
    if not cams or any(
        not isinstance(cam, torch.Tensor)
        or cam.ndim != 3
        or cam.shape[0] != 1
        or cam.numel() == 0
        or not torch.isfinite(cam).all().item()
        for cam in cams
    ):
        raise RuntimeError("CAM extractor returned an invalid activation map")
    shapes = tuple(tuple(cam.shape) for cam in cams)
    if expected_shapes is not None and shapes != expected_shapes:
        raise RuntimeError(f"CAM shapes changed across samples: expected {expected_shapes}, got {shapes}")
    return shapes


def _build_parser():
    parser = argparse.ArgumentParser(
        description="CAM method latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("method", choices=METHOD_NAMES, help="CAM method to use")
    parser.add_argument("--arch", default="resnet18", help="Name of the torchvision architecture")
    parser.add_argument("--size", type=positive_int, default=224, help="The image input size")
    parser.add_argument("--class-idx", type=int, default=232, help="Index of the class to inspect")
    parser.add_argument("--device", default=None, help="Device (auto-selects CUDA, otherwise CPU; pass mps explicitly)")
    parser.add_argument("--it", type=positive_int, default=100, help="Number of iterations to run")
    parser.add_argument(
        "--warmup", type=nonnegative_int, default=10, help="Full CAM warm-up calls after the first call"
    )
    parser.add_argument("--repeat", type=positive_int, default=5, help="Fresh worker processes")
    parser.add_argument("--threads", type=positive_int, default=1, help="CPU threads; inter-op threads stay at 1")
    parser.add_argument("--seed", type=nonnegative_int, default=0, help="Random seed")
    parser.add_argument("--output", type=Path, help="Save settings, environment, trials, and summary as JSON")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--scope", choices=("extractor", "end-to-end"), default="extractor", help="Region included in timing"
    )
    parser.add_argument("--weights", choices=("default", "none"), default="default", help="Torchvision weights")
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=32,
        help="Masked-input batch size for ScoreCAM-family methods",
    )
    parser.add_argument("--target-layer", action="append", help="Target layer name; repeat for multiple layers")
    return parser


def _evaluate(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    weights = get_model_weights(args.arch).DEFAULT if args.weights == "default" else None
    if device.type == "mps":
        with device:
            model = get_model(args.arch, weights=weights).eval()
    else:
        model = get_model(args.arch, weights=weights).eval().to(device=device)
    model.requires_grad_(False)

    input_tensor = torch.rand((1, 3, args.size, args.size), device=device, requires_grad=True)

    extractor_cls = getattr(methods, args.method)
    extractor_kwargs = {"batch_size": args.batch_size} if issubclass(extractor_cls, methods.ScoreCAM) else {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings = []
    expected_shapes = None
    cam_kwargs = {"target_shape": tuple(input_tensor.shape[2:])} if extractor_cls is methods.RefineCAM else {}
    with build_extractor(
        model, args.method, args.target_layer, tuple(input_tensor.shape[1:]), **extractor_kwargs
    ) as cam_extractor:
        for idx in range(1 + args.warmup + args.it):
            elapsed, cams = _time_sample(
                model,
                input_tensor,
                cam_extractor,
                args.class_idx,
                device,
                args.scope,
                cam_kwargs,
            )
            expected_shapes = _validate_cams(cams, expected_shapes)
            if idx == 0:
                first_ms = 1000 * elapsed
            elif idx > args.warmup:
                timings.append(1000 * elapsed)

    q1, median, q3 = np.percentile(timings, (25, 50, 75))
    result = {
        "pid": os.getpid(),
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "device": str(device),
        "weights": str(weights),
        "checkpoint": weights.url if weights else None,
        "target_layers": cam_extractor.target_names,
        "cam_shapes": expected_shapes,
        "samples_ms": timings,
        "first_ms": first_ms,
        "median_ms": float(median),
        "p95_ms": float(np.percentile(timings, 95, method="higher")),
        "iqr_ms": float(q3 - q1),
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "peak_rss_mib": None,
    }
    if sys.platform != "win32":
        import resource  # noqa: PLC0415

        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (
            1024**2 if sys.platform == "darwin" else 1024
        )
    if device.type == "cuda":
        result.update(
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 1024**2,
            cuda_peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 1024**2,
        )
    return result


def main(args):
    if args.worker:
        print(json.dumps(_evaluate(args), allow_nan=False))
        return
    command = [sys.executable, str(Path(__file__).resolve()), args.method, "--worker"]
    for name, value in vars(args).items():
        if name not in {"method", "worker", "output"} and value is not None:
            for item in value if isinstance(value, list) else [value]:
                command.extend(["--" + name.replace("_", "-"), str(item)])
    runs = [
        json.loads(subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True).stdout)  # noqa: S603
        for _ in range(args.repeat)
    ]
    summary = {}
    for key in runs[0]:
        if key != "samples_ms" and key.endswith(("_ms", "_mib")):
            values = [run[key] for run in runs if run[key] is not None]
            summary[key] = (max(values) if "peak" in key else float(np.median(values))) if values else None
    summary["median_range_ms"] = [min(run["median_ms"] for run in runs), max(run["median_ms"] for run in runs)]
    if args.output:
        root = Path(__file__).resolve().parents[1]
        git = shutil.which("git")
        revision = (
            subprocess.run([git, "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)  # noqa: S603
            if git
            else None
        )
        report = {
            "schema_version": 1,
            "config": {key: value for key, value in vars(args).items() if key not in {"worker", "output"}},
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor() or platform.machine(),
                "versions": {
                    "torch": torch.__version__,
                    "torchvision": torchvision.__version__,
                    "torchcam": __version__,
                },
                "numpy": np.__version__,
                "cuda_version": torch.version.cuda,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_precision": torch.backends.cudnn.fp32_precision
                if hasattr(torch.backends.cudnn, "fp32_precision")
                else torch.backends.cudnn.allow_tf32,
                "matmul_precision": torch.backends.cuda.matmul.fp32_precision
                if hasattr(torch.backends.cuda.matmul, "fp32_precision")
                else torch.backends.cuda.matmul.allow_tf32,
                "revision": revision.stdout.strip() or None if revision else None,
                "dirty": bool(
                    subprocess.run(  # noqa: S603
                        [git, "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                        capture_output=True,
                        text=True,
                        check=True,
                    ).stdout
                )
                if git and revision and revision.returncode == 0
                else None,
                "harness_sha256": hashlib.sha256(
                    Path(__file__).read_bytes() + Path(__file__).with_name("cam_example.py").read_bytes()
                ).hexdigest(),
            },
            "runs": runs,
            "summary": summary,
        }
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    rss = "unavailable" if summary["peak_rss_mib"] is None else f"{summary['peak_rss_mib']:.1f} MiB"
    print(
        f"{args.method}/{args.arch}: {args.scope}, {runs[0]['device']}, {args.repeat} fresh processes; "
        f"weights={runs[0]['weights']} target_layers={','.join(runs[0]['target_layers'])} threads={args.threads} seed={args.seed}"
    )
    print(
        f"First call {summary['first_ms']:.2f} ms; median {summary['median_ms']:.2f} ms; "
        f"p95 {summary['p95_ms']:.2f} ms; IQR {summary['iqr_ms']:.2f} ms; peak RSS {rss}"
    )


if __name__ == "__main__":
    main(_build_parser().parse_args())
