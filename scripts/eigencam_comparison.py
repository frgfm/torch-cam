# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Reproduce EigenCAM's tradeoffs against class-specific methods on a pretrained ResNet18."""

import argparse
import hashlib
import json
import platform
import subprocess  # noqa: S404
import sys
import time
from copy import deepcopy
from functools import partial
from pathlib import Path
from urllib.request import urlopen

import matplotlib as mpl
import numpy as np
import torch
import torchvision
from PIL import Image, ImageOps
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.functional import to_pil_image

from torchcam import __version__, methods
from torchcam.metrics import ClassificationMetric

if __package__:
    from .cam_example import positive_int
    from .eval_latency import _synchronize
else:
    from cam_example import positive_int
    from eval_latency import _synchronize  # noqa: PLC2701 - reuse the benchmark synchronization helper

SOURCE_REVISION = "e8a42ba85aaf5fd9322ef9ca51bc21063b22fcae"  # scikit-image v0.25.2
SOURCES = {
    "chelsea": "596aa1e7cb875eb79f437e310381d26b338a81c2da23439704a73c4651e8c4bb",
    "coffee": "cc02f8ca188b167c775a7101b5d767d1e71792cf762c33d6fa15a4599b5a8de7",
    "astronaut": "88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5",
}
METHODS = ("EigenCAM", "CAM", "GradCAM", "LayerCAM")
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


def _load_images(directory):
    directory.mkdir(parents=True, exist_ok=True)
    images = {}
    for name, expected_hash in SOURCES.items():
        path = directory / f"{name}.png"
        if not path.exists():
            url = (
                f"https://raw.githubusercontent.com/scikit-image/scikit-image/{SOURCE_REVISION}/skimage/data/{name}.png"
            )
            with urlopen(url, timeout=60) as response:  # nosec B310
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError(f"Unexpected image checksum: {name}")
            path.write_bytes(content)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Unexpected image checksum: {name}")
        with Image.open(path) as image:
            images[name] = image.convert("RGB")
    # A controlled two-object illustration, not a photograph of a natural scene.
    composite = Image.new("RGB", (448, 448))
    for idx, name in enumerate(("chelsea", "coffee")):
        composite.paste(ImageOps.fit(images[name], (448, 224), method=Image.Resampling.BILINEAR), (0, idx * 224))
    images["cat-and-coffee"] = composite
    return images


def _context(method):
    return torch.enable_grad() if method in {"GradCAM", "LayerCAM"} else torch.inference_mode()


def _input(image, device):
    return WEIGHTS.transforms()(image).unsqueeze(0).to(device)


def _extractor(model, method):
    kwargs = {"fc_layer": "fc"} if method == "CAM" else {}
    return getattr(methods, method)(model, target_layer="layer4", **kwargs)


def _extract(model, tensor, method, class_idx):
    with _extractor(model, method) as extractor, _context(method):
        tensor = tensor.detach().clone().requires_grad_(method in {"GradCAM", "LayerCAM"})
        scores = model(tensor)
        cams = extractor() if method == "EigenCAM" else extractor(class_idx, scores)
    return cams[0].detach().cpu()


def _time_sample(model, tensor, extractor, method, class_idx, device, scope):
    # Only extraction is timed in the extractor scope; every call still gets a fresh forward/graph.
    with _context(method):
        if scope == "extractor":
            scores = model(tensor)
        _synchronize(device)
        started = time.perf_counter()
        if scope == "end-to-end":
            scores = model(tensor)
        cams = extractor() if method == "EigenCAM" else extractor(class_idx, scores)
        _synchronize(device)
        elapsed = 1000 * (time.perf_counter() - started)
    if any(not torch.isfinite(cam).all() for cam in cams):
        raise RuntimeError("Nonfinite CAM in timing run")
    return elapsed


def _benchmark(args):
    device = torch.device(args.device)
    model = resnet18(weights=WEIGHTS).eval().requires_grad_(False).to(device)
    image = _load_images(args.inputs)["chelsea"]
    tensor = _input(image, device)
    with torch.inference_mode():
        class_idx = int(model(tensor).argmax())
    results = {}
    # Rotate method order between fresh worker processes to reduce ordering effects.
    order = METHODS[args.worker % len(METHODS) :] + METHODS[: args.worker % len(METHODS)]
    for method in order:
        method_input = tensor.detach().clone().requires_grad_(method in {"GradCAM", "LayerCAM"})
        results[method] = {}
        with _extractor(model, method) as extractor:
            for scope in ("extractor", "end-to-end"):
                timings = [
                    _time_sample(model, method_input, extractor, method, class_idx, device, scope)
                    for _ in range(1 + args.warmup + args.iterations)
                ]
                samples = timings[1 + args.warmup :]
                results[method][scope] = {
                    "first_ms": timings[0],
                    "samples_ms": samples,
                    "median_ms": float(np.median(samples)),
                    "p95_ms": float(np.percentile(samples, 95, method="higher")),
                }
    return results


def _display(tensor):
    preprocess = WEIGHTS.transforms()
    mean = tensor.new_tensor(preprocess.mean).view(3, 1, 1)
    std = tensor.new_tensor(preprocess.std).view(3, 1, 1)
    return to_pil_image((tensor[0].cpu() * std.cpu() + mean.cpu()).clamp(0, 1))


def _overlay(image, cam):
    mask = torch.nn.functional.interpolate(cam.unsqueeze(1), image.size[::-1], mode="bilinear", align_corners=False)
    color = mpl.colormaps["magma"](mask[0, 0].numpy())[..., :3]
    return 0.5 * np.asarray(image) / 255 + 0.5 * color


def _save_grid(path, rows, title):
    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    with plt.rc_context({"font.size": 11, "font.family": "DejaVu Sans"}):
        fig, axes = plt.subplots(
            len(rows), len(rows[0]), figsize=(3.1 * len(rows[0]), 3.6 * len(rows)), layout="constrained"
        )
        fig.get_layout_engine().set(h_pad=0.12, w_pad=0.08)
        axes = np.asarray(axes).reshape(len(rows), len(rows[0]))
        for axes_row, entries in zip(axes, rows, strict=True):
            for ax, (image, label) in zip(axes_row, entries, strict=True):
                ax.imshow(image)
                ax.set_title(label, fontsize=11)
                ax.axis("off")
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.savefig(path, dpi=140, facecolor="white", metadata={"Software": "TorchCAM EigenCAM comparison"})
        plt.close(fig)
    # Keep the full-resolution PNG locally; compact WebP copies fit the repository's file-size gate.
    with Image.open(path) as screenshot:
        screenshot.convert("RGB").save(path.with_suffix(".webp"), quality=90, method=6)


def _visual_evidence(args):
    device = torch.device(args.device)
    model = resnet18(weights=WEIGHTS).eval().requires_grad_(False).to(device)
    images = _load_images(args.inputs)
    quality = {method: [] for method in METHODS}
    rows, predictions, eigen_maps = [], {}, {}
    for name in SOURCES:
        tensor = _input(images[name], device)
        display = _display(tensor)
        with torch.inference_mode():
            probabilities = model(tensor).softmax(1)
            class_idx = int(probabilities.argmax())
        label = WEIGHTS.meta["categories"][class_idx]
        predictions[name] = {"class_idx": class_idx, "class": label, "probability": float(probabilities[0, class_idx])}
        row = [(display, f"{name}\nTop prediction: {label}")]
        for method in METHODS:
            cam = _extract(model, tensor, method, class_idx)
            if method == "EigenCAM":
                eigen_maps[name] = cam
            row.append((_overlay(display, cam), f"{method}\n{'class-independent' if method == 'EigenCAM' else label}"))
            with _extractor(model, method) as extractor, _context(method):
                metric = ClassificationMetric(extractor, partial(torch.softmax, dim=-1))
                metric.update(tensor.detach().clone().requires_grad_(method in {"GradCAM", "LayerCAM"}), class_idx)
                quality[method].append({"image": name, **metric.summary()})
        rows.append(row)
    _save_grid(args.output / "comparison.png", rows, "Same pretrained ResNet18 / layer4 / 224 x 224 crop")

    class_difference = _class_independence(model, images["cat-and-coffee"], args)
    headless = _headless_evidence(model, images["chelsea"], eigen_maps["chelsea"], args)
    return {
        "predictions": predictions,
        "classification_metric_per_image": quality,
        "classification_metric_mean": {
            method: {key: float(np.mean([item[key] for item in values])) for key in ("avg_drop", "conf_increase")}
            for method, values in quality.items()
        },
        "class_independence_max_abs_difference": class_difference,
        "headless_encoder": headless,
    }


def _class_independence(model, image, args):
    tensor = _input(image, torch.device(args.device))
    display = _display(tensor)
    selected = [WEIGHTS.meta["categories"].index(label) for label in ("tabby", "espresso")]
    with methods.EigenCAM(model, "layer4") as extractor, torch.inference_mode():
        scores = model(tensor)
        cams = [extractor(idx, scores)[0].cpu() for idx in selected]
        class_difference = float((cams[0] - cams[1]).abs().max())
    rows = []
    for idx, cam in zip(selected, cams, strict=True):
        label = WEIGHTS.meta["categories"][idx]
        row = [
            (display, f"Constructed two-object input\nRequested class: {label}"),
            (_overlay(display, cam), "EigenCAM\nSame map for either class"),
        ]
        row.extend(
            (_overlay(display, _extract(model, tensor, method, idx)), f"{method}\nTarget: {label}")
            for method in ("CAM", "LayerCAM")
        )
        rows.append(row)
    _save_grid(
        args.output / "class-independence.png", rows, "Feature structure is different from class-specific evidence"
    )

    if class_difference != 0:
        raise RuntimeError("EigenCAM changed when selecting another class")
    return class_difference


def _headless_evidence(model, image, original_cam, args):
    # A feature encoder has an embedding, not class logits. Its layer4 features are unchanged.
    model.fc = torch.nn.Identity()
    tensor = _input(image, torch.device(args.device))
    with methods.EigenCAM(model, "layer4") as extractor, torch.inference_mode():
        embedding = model(tensor)
        headless_cam = extractor()[0].cpu()
    headless_difference = float((headless_cam - original_cam).abs().max())
    errors = {}
    try:
        with methods.CAM(deepcopy(model), "layer4"):
            pass
    except ValueError as exc:
        errors["CAM"] = str(exc)
    try:
        with methods.LayerCAM(model, "layer4") as extractor, torch.inference_mode():
            output = model(tensor)
            extractor(0, output)
    except (ValueError, RuntimeError) as exc:
        errors["LayerCAM_under_inference_mode"] = str(exc)
    if headless_difference > 1e-5 or len(errors) != 2:
        raise RuntimeError("Class independence or headless-encoder check failed")
    display = _display(tensor)
    _save_grid(
        args.output / "headless-encoder.png",
        [
            [
                (display, "Preprocessed input"),
                (_overlay(display, original_cam), "EigenCAM with classifier\nNo class selected"),
                (_overlay(display, headless_cam), "EigenCAM with Identity head\nFrozen encoder + inference_mode"),
            ]
        ],
        "Inspect feature structure without a classifier or autograd",
    )
    return {
        "embedding_shape": list(embedding.shape),
        "map_max_abs_difference": headless_difference,
        "baseline_errors": errors,
    }


def main(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    if args.worker is not None:
        print(json.dumps(_benchmark(args), allow_nan=False))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = _visual_evidence(args)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    runs, worker_warnings = [], []
    for repeat in range(args.repeat):
        result = subprocess.run([*command, "--worker", str(repeat)], check=True, capture_output=True, text=True)  # noqa: S603
        runs.append(json.loads(result.stdout))
        worker_warnings.append(result.stderr.strip().replace(str(Path(__file__).resolve().parents[1]), "<checkout>"))
    summary = {
        method: {
            scope: {
                "median_ms": float(np.median([run[method][scope]["median_ms"] for run in runs])),
                "median_range_ms": [
                    min(run[method][scope]["median_ms"] for run in runs),
                    max(run[method][scope]["median_ms"] for run in runs),
                ],
                "p95_ms": float(np.median([run[method][scope]["p95_ms"] for run in runs])),
            }
            for scope in ("extractor", "end-to-end")
        }
        for method in METHODS
    }
    report = {
        "schema_version": 1,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "worker"
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torchcam": __version__,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        },
        "model": {
            "architecture": "resnet18",
            "weights": str(WEIGHTS),
            "checkpoint_url": WEIGHTS.url,
            "target_layer": "layer4",
            "feature_shape": [1, 512, 7, 7],
            "input_shape": [1, 3, 224, 224],
            "parameters_frozen": True,
        },
        "input_sources": {"revision": SOURCE_REVISION, "sha256": SOURCES},
        "source_sha256": {
            "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "activation.py": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "torchcam/methods/activation.py").read_bytes()
            ).hexdigest(),
        },
        "evidence": evidence,
        "latency": {"input": "chelsea", "runs": runs, "worker_stderr": worker_warnings, "summary": summary},
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"latency": summary, "quality": evidence["classification_metric_mean"]}, indent=2))


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--inputs", type=Path, default=Path("downloads/eigencam-inputs"))
    parser.add_argument("--output", type=Path, default=Path("downloads/eigencam-comparison"))
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--warmup", type=positive_int, default=10)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
