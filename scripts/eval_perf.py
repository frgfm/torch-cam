# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
CAM performance evaluation
"""

import argparse
import math
import os
import platform
from functools import partial
from pathlib import Path

import torch
import torchvision
from torch.utils.data import SequentialSampler
from torchvision.datasets import ImageFolder
from torchvision.models import get_model, get_model_weights
from torchvision.transforms import v2 as T
from torchvision.transforms.functional import InterpolationMode

from torchcam import __version__
from torchcam.metrics import ClassificationMetric, DeletionInsertionMetric

if __package__:
    from .cam_example import METHOD_NAMES, build_extractor, nonnegative_int, positive_int
else:
    from cam_example import METHOD_NAMES, build_extractor, nonnegative_int, positive_int

BENCHMARK_WEIGHTS = {"resnet18": "IMAGENET1K_V1", "mobilenet_v3_large": "IMAGENET1K_V2"}


def _resolve_weights(arch, name):
    available = get_model_weights(arch)
    name = name or BENCHMARK_WEIGHTS.get(arch, "DEFAULT")
    try:
        return available[name]
    except KeyError:
        raise ValueError(
            f"unknown weights {name!r} for {arch}; choose from {', '.join(available.__members__)}"
        ) from None


def _build_transform(size):
    return T.Compose([
        T.Resize(math.floor(size / 0.875), interpolation=InterpolationMode.BILINEAR, antialias=True),
        T.CenterCrop(size),
        T.PILToTensor(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def _report_protocol(args, weights, cam_extractor, dataset):
    print(
        f"method={args.method} model={args.arch} device={args.device} seed={args.seed} "
        f"weights={weights} target_layers={','.join(cam_extractor.target_names)}"
    )
    print(f"checkpoint={weights.url}")
    print(
        f"dataset={Path(dataset.root).resolve()} samples={len(dataset)} batch_size={args.batch_size} "
        f"workers={args.workers} threads={torch.get_num_threads()}"
    )
    print(
        f"python={platform.python_version()} torch={torch.__version__} "
        f"torchvision={torchvision.__version__} torchcam={__version__}"
    )
    print(
        f"resize={math.floor(args.size / 0.875)} interpolation=bilinear antialias=True crop={args.size} "
        "mean=(0.485,0.456,0.406) std=(0.229,0.224,0.225) "
        "target=original-predicted-class scoring=softmax masking=normalized-input"
    )
    if args.deletion_insertion:
        print(
            f"deletion_insertion=True baseline=normalized-zero-for-both-curves steps={args.di_steps} "
            f"di_batch_size={args.di_batch_size} cam_draws=separate-per-metric"
        )


def main(args):
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    weights = _resolve_weights(args.arch, args.weights)
    model = get_model(args.arch, weights=weights).eval().to(device=device)
    model.requires_grad_(False)

    ds = ImageFolder(
        Path(args.data_path).joinpath("val"),
        _build_transform(args.size),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=False,
        sampler=SequentialSampler(ds),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    # Hook the corresponding layer in the model
    with build_extractor(model, args.method, args.target, (3, args.size, args.size)) as cam_extractor:
        _report_protocol(args, weights, cam_extractor, ds)
        metric = ClassificationMetric(cam_extractor, partial(torch.softmax, dim=-1))
        deletion_insertion_metric = (
            DeletionInsertionMetric(
                cam_extractor,
                partial(torch.softmax, dim=-1),
                steps=args.di_steps,
                batch_size=args.di_batch_size,
            )
            if args.deletion_insertion
            else None
        )

        # Evaluation runs
        for x, _ in loader:
            model.zero_grad()
            x = x.to(device=device)
            x.requires_grad_(True)
            metric.update(x)
            if deletion_insertion_metric is not None:
                model.zero_grad()
                deletion_insertion_metric.update(x.detach().requires_grad_(True))

    print(f"{args.method} w/ {args.arch} ({len(ds)} validation inputs of size ({args.size}, {args.size}))")
    metrics_dict = metric.summary()
    print(
        f"Average Drop {metrics_dict['avg_drop']:.2%}, Increase in Confidence {metrics_dict['conf_increase']:.2%}, "
        f"Valid {metric.total} samples, Skipped {metric.nan_count} samples"
    )
    if deletion_insertion_metric is not None:
        faithfulness = deletion_insertion_metric.summary()
        print(
            f"Deletion AUC {faithfulness['deletion_auc']:.4f}, Insertion AUC {faithfulness['insertion_auc']:.4f}, "
            f"Valid {deletion_insertion_metric.total} samples, Skipped {deletion_insertion_metric.nan_count} samples"
        )


def _build_parser():
    parser = argparse.ArgumentParser(
        description="CAM method performance evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_path", type=str, help="path to dataset folder")
    parser.add_argument("method", choices=METHOD_NAMES, help="CAM method to use")
    parser.add_argument(
        "--arch",
        type=str,
        default="mobilenet_v3_large",
        help="Name of the torchvision architecture",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Torchvision weights name (ResNet18: IMAGENET1K_V1; MobileNet V3 Large: IMAGENET1K_V2; others: DEFAULT)",
    )
    parser.add_argument("--seed", type=nonnegative_int, default=0, help="PyTorch random seed")
    parser.add_argument("--target", type=str, default=None, help="Target layer name")
    parser.add_argument("--size", type=positive_int, default=224, help="The image input size")
    parser.add_argument("-b", "--batch-size", default=32, type=positive_int, help="batch size")
    parser.add_argument(
        "--deletion-insertion",
        action="store_true",
        help="also compute deletion and insertion faithfulness AUCs",
    )
    parser.add_argument(
        "--di-steps", default=20, type=positive_int, help="maximum deletion/insertion perturbation intervals"
    )
    parser.add_argument(
        "--di-batch-size", default=32, type=positive_int, help="deletion/insertion perturbation chunk size"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Default device to perform computation on",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=min(os.cpu_count() or 1, 16),
        type=nonnegative_int,
        help="number of data loading workers",
    )
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
