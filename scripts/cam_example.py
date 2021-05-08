# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

"""
CAM visualization
"""

import math
import argparse
from io import BytesIO

import matplotlib.pyplot as plt
import requests
from PIL import Image
import torch
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image

from torchcam import cams
from torchcam.utils import overlay_mask


def main(args):

    if args.device is None:
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    device = torch.device(args.device)

    # Pretrained imagenet model
    model = models.__dict__[args.model](pretrained=True).to(device=device)

    # Image
    if args.img.startswith('http'):
        img_path = BytesIO(requests.get(args.img).content)
    else:
        img_path = args.img
    pil_img = Image.open(img_path, mode='r').convert('RGB')

    # Preprocess image
    img_tensor = normalize(to_tensor(resize(pil_img, (224, 224))),
                           [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device=device)

    if isinstance(args.method, str):
        methods = [args.method]
    else:
        methods = [
            'CAM',
            'GradCAM', 'GradCAMpp', 'SmoothGradCAMpp',
            'ScoreCAM', 'SSCAM', 'ISCAM',
            'XGradCAM',
        ]
    # Hook the corresponding layer in the model
    cam_extractors = [cams.__dict__[name](model) for name in methods]

    # Don't trigger all hooks
    for extractor in cam_extractors:
        extractor._hooks_enabled = False

    # Homogenize number of elements in each row
    num_cols = math.ceil((len(cam_extractors) + 1) / args.rows)
    _, axes = plt.subplots(args.rows, num_cols, figsize=(6, 4))
    # Display input
    ax = axes[0][0] if args.rows > 1 else axes[0] if num_cols > 1 else axes
    ax.imshow(pil_img)
    ax.set_title("Input", size=8)

    for idx, extractor in zip(range(1, len(cam_extractors) + 1), cam_extractors):
        extractor._hooks_enabled = True
        model.zero_grad()
        scores = model(img_tensor.unsqueeze(0))

        # Select the class index
        class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx

        # Use the hooked data to compute activation map
        activation_map = extractor(class_idx, scores).cpu()

        # Clean data
        extractor.clear_hooks()
        extractor._hooks_enabled = False
        # Convert it to PIL image
        # The indexing below means first image in batch
        heatmap = to_pil_image(activation_map, mode='F')
        # Plot the result
        result = overlay_mask(pil_img, heatmap, alpha=args.alpha)

        ax = axes[idx // num_cols][idx % num_cols] if args.rows > 1 else axes[idx] if num_cols > 1 else axes

        ax.imshow(result)
        ax.set_title(extractor.__class__.__name__, size=8)

    # Clear axes
    if num_cols > 1:
        for _axes in axes:
            if args.rows > 1:
                for ax in _axes:
                    ax.axis('off')
            else:
                _axes.axis('off')

    else:
        axes.axis('off')

    plt.tight_layout()
    if args.savefig:
        plt.savefig(args.savefig, dpi=200, transparent=True, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Saliency Map comparison',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default='resnet18', help="The name of your training")
    parser.add_argument("--img", type=str,
                        default='https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg',
                        help="The image to extract CAM from")
    parser.add_argument("--class-idx", type=int, default=232, help='Index of the class to inspect')
    parser.add_argument("--device", type=str, default=None, help='Default device to perform computation on')
    parser.add_argument("--savefig", type=str, default=None, help="Path to save figure")
    parser.add_argument("--method", type=str, default=None, help="CAM method to use")
    parser.add_argument("--alpha", type=float, default=0.7, help="Transparency of the heatmap")
    parser.add_argument("--rows", type=int, default=1, help="Number of rows for the layout")
    args = parser.parse_args()

    main(args)
