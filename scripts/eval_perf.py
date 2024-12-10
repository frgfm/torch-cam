# Copyright (C) 2022-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
CAM performance evaluation
"""

import argparse
import math
import os
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms as T

from torchcam import methods
from torchcam.metrics import ClassificationMetric


def main(args):
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)

    # Pretrained imagenet model
    model = models.__dict__[args.arch](pretrained=True).to(device=device)
    # Freeze the model
    for p in model.parameters():
        p.requires_grad_(False)

    eval_tf = []
    crop_pct = 0.875
    scale_size = min(math.floor(args.size / crop_pct), 320)
    if scale_size < 320:
        eval_tf.append(T.Resize(scale_size))
    eval_tf.extend([
        T.CenterCrop(args.size),
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    ds = ImageFolder(
        Path(args.data_path).joinpath("val"),
        T.Compose(eval_tf),
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        drop_last=False,
        sampler=SequentialSampler(ds),
        num_workers=args.workers,
        pin_memory=True,
    )

    # Hook the corresponding layer in the model
    with methods.__dict__[args.method](model, args.target.split(",")) as cam_extractor:
        metric = ClassificationMetric(cam_extractor, partial(torch.softmax, dim=-1))

        # Evaluation runs
        for x, _ in loader:
            model.zero_grad()
            x = x.to(device=device)
            x.requires_grad_(True)
            metric.update(x)

    print(f"{args.method} w/ {args.arch} (validation set of Imagenette on ({args.size}, {args.size}) inputs)")
    metrics_dict = metric.summary()
    print(f"Average Drop {metrics_dict['avg_drop']:.2%}, Increase in Confidence {metrics_dict['conf_increase']:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CAM method performance evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data_path", type=str, help="path to dataset folder")
    parser.add_argument("method", type=str, help="CAM method to use")
    parser.add_argument(
        "--arch",
        type=str,
        default="mobilenet_v3_large",
        help="Name of the torchvision architecture",
    )
    parser.add_argument("--target", type=str, default=None, help="Target layer name")
    parser.add_argument("--size", type=int, default=224, help="The image input size")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Default device to perform computation on",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=min(os.cpu_count(), 16),
        type=int,
        help="number of data loading workers",
    )
    args = parser.parse_args()

    main(args)
