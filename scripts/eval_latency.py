# Copyright (C) 2021-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
CAM latency benchmark
"""

import argparse
import time

import numpy as np
import torch
from torchvision.models import get_model, get_model_weights

from torchcam import methods

METHOD_NAMES = tuple(sorted(name for name, value in vars(methods).items() if isinstance(value, type)))


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return value


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
        or cam.ndim < 3
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
    parser.add_argument("--size", type=_positive_int, default=224, help="The image input size")
    parser.add_argument("--class-idx", type=int, default=232, help="Index of the class to inspect")
    parser.add_argument("--device", default=None, help="Device (auto-selects CUDA, otherwise CPU; pass mps explicitly)")
    parser.add_argument("--it", type=_positive_int, default=100, help="Number of iterations to run")
    parser.add_argument(
        "--scope", choices=("extractor", "end-to-end"), default="extractor", help="Region included in timing"
    )
    parser.add_argument("--weights", choices=("default", "none"), default="default", help="Torchvision weights")
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=32,
        help="Masked-input batch size for ScoreCAM-family methods",
    )
    parser.add_argument("--target-layer", action="append", help="Target layer name; repeat for multiple layers")
    return parser


def main(args):
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)

    weights = get_model_weights(args.arch).DEFAULT if args.weights == "default" else None
    if device.type == "mps":
        with device:
            model = get_model(args.arch, weights=weights).eval()
    else:
        model = get_model(args.arch, weights=weights).eval().to(device=device)
    model.requires_grad_(False)

    input_tensor = torch.rand((1, 3, args.size, args.size), device=device, requires_grad=True)

    for _ in range(10):
        with torch.no_grad():
            _ = model(input_tensor)
        _synchronize(device)

    extractor_cls = getattr(methods, args.method)
    extractor_kwargs = (
        {"target_layer": args.target_layer[0] if len(args.target_layer) == 1 else args.target_layer}
        if args.target_layer
        else {}
    )
    if issubclass(extractor_cls, methods.ScoreCAM):
        extractor_kwargs["batch_size"] = args.batch_size

    timings = []
    expected_shapes = None
    cam_kwargs = {"target_shape": tuple(input_tensor.shape[2:])} if extractor_cls is methods.RefineCAM else {}
    with extractor_cls(model, **extractor_kwargs) as cam_extractor:
        for _ in range(args.it):
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
            timings.append(1000 * elapsed)

    timings_ = np.asarray(timings)
    q1, median, q3 = np.percentile(timings_, (25, 50, 75))
    target_layers = ",".join(args.target_layer) if args.target_layer else "auto"
    cam_batch_size = args.batch_size if issubclass(extractor_cls, methods.ScoreCAM) else "n/a"
    print(
        f"method={args.method} model={args.arch} device={device} input=1x3x{args.size}x{args.size} "
        f"iterations={args.it} scope={args.scope} weights={args.weights} seed=0 input_batch_size=1 "
        f"cam_batch_size={cam_batch_size} target_layers={target_layers}"
    )
    print(f"samples_ms=[{', '.join(f'{sample:.3f}' for sample in timings_)}]")
    print(f"median {median:.2f}ms, IQR {q3 - q1:.2f}ms (q1 {q1:.2f}ms, q3 {q3:.2f}ms)")
    print(f"mean {timings_.mean():.2f}ms, std {timings_.std():.2f}ms")


if __name__ == "__main__":
    main(_build_parser().parse_args())
