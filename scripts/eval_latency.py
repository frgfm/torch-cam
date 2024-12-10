# Copyright (C) 2021-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
CAM latency benchmark
"""

import argparse
import time

import numpy as np
import torch
from torchvision import models

from torchcam import methods


def main(args):
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)

    # Pretrained imagenet model
    model = models.__dict__[args.arch](pretrained=True).to(device=device)
    # Freeze the model
    for p in model.parameters():
        p.requires_grad_(False)

    # Input
    img_tensor = torch.rand((1, 3, args.size, args.size)).to(device=device)
    img_tensor.requires_grad_(True)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(img_tensor)

    timings = []

    # Evaluation runs
    with methods.__dict__[args.method](model) as cam_extractor:
        for _ in range(args.it):
            scores = model(img_tensor)

            # Select the class index
            class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx

            # Use the hooked data to compute activation map
            start_ts = time.perf_counter()
            _ = cam_extractor(class_idx, scores)
            timings.append(time.perf_counter() - start_ts)

    timings_ = np.array(timings)
    print(f"{args.method} w/ {args.arch} ({args.it} runs on ({args.size}, {args.size}) inputs)")
    print(f"mean {1000 * timings_.mean():.2f}ms, std {1000 * timings_.std():.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CAM method latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("method", type=str, help="CAM method to use")
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet18",
        help="Name of the torchvision architecture",
    )
    parser.add_argument("--size", type=int, default=224, help="The image input size")
    parser.add_argument("--class-idx", type=int, default=232, help="Index of the class to inspect")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Default device to perform computation on",
    )
    parser.add_argument("--it", type=int, default=100, help="Number of iterations to run")
    args = parser.parse_args()

    main(args)
