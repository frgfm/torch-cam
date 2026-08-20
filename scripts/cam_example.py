# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""
CAM visualization
"""

import argparse
import math
from functools import partial
from io import BytesIO
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.models import get_model, get_model_weights
from torchvision.models.swin_transformer import SwinTransformer
from torchvision.models.vision_transformer import VisionTransformer
from torchvision.transforms.functional import to_pil_image, to_tensor

from torchcam import methods
from torchcam.utils import overlay_mask


def vit_reshape_transform(tensor, grid_size):
    patches = tensor[:, 1:, :].reshape(tensor.size(0), grid_size, grid_size, tensor.size(-1))
    return patches.permute(0, 3, 1, 2)


def swin_reshape_transform(tensor):
    return tensor.permute(0, 3, 1, 2)


def resolve_transformer_config(model, target_layer):
    reshape_transform = None
    if isinstance(model, VisionTransformer):
        grid_size = model.image_size // model.patch_size
        target_layer = target_layer or model.encoder.layers[-2].ln_1
        reshape_transform = partial(vit_reshape_transform, grid_size=grid_size)
    elif isinstance(model, SwinTransformer):
        target_layer = target_layer or model.features[-1][-1].norm2
        reshape_transform = swin_reshape_transform
    return target_layer, reshape_transform


def _load_image(img_path):
    if img_path.startswith(("http://", "https://")):
        request = Request(  # noqa: S310
            img_path,
            headers={"User-Agent": "TorchCAM (+https://github.com/frgfm/torch-cam)"},
        )
        with urlopen(request, timeout=5) as response:  # noqa: S310  # nosec B310
            img_path = BytesIO(response.read())
    return Image.open(img_path, mode="r").convert("RGB")


def main(args):  # noqa: PLR0912
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(args.device)

    # Pretrained imagenet model
    weights = get_model_weights(args.arch).DEFAULT
    model = get_model(args.arch, weights=weights).to(device=device).eval()
    # Freeze the model
    for p in model.parameters():
        p.requires_grad_(False)

    # Image
    pil_img = _load_image(args.img)
    preprocess = weights.transforms()

    # Preprocess image
    img_tensor = preprocess(to_tensor(pil_img)).to(device=device)
    img_tensor.requires_grad_(True)

    target_layer, reshape_transform = resolve_transformer_config(model, args.target)

    if isinstance(args.method, str):
        cam_methods = args.method.split(",")
    elif reshape_transform is not None:
        cam_methods = ["GradCAM"]
    else:
        cam_methods = [
            "CAM",
            "GradCAM",
            "GradCAMpp",
            "SmoothGradCAMpp",
            "ScoreCAM",
            "SSCAM",
            "ISCAM",
            "XGradCAM",
            "LayerCAM",
        ]
    # Hook the corresponding layer in the model
    cam_extractors = [
        methods.__dict__[name](
            model,
            target_layer=target_layer,
            enable_hooks=False,
            reshape_transform=reshape_transform,
        )
        for name in cam_methods
    ]

    # Homogenize number of elements in each row
    num_cols = math.ceil((len(cam_extractors) + 1) / args.rows)
    _, axes = plt.subplots(args.rows, num_cols, figsize=(6, 4))
    # Display input
    ax = axes[0][0] if args.rows > 1 else axes[0] if num_cols > 1 else axes
    ax.imshow(pil_img)
    ax.set_title("Input", size=8)

    for idx, extractor in zip(range(1, len(cam_extractors) + 1), cam_extractors, strict=True):
        extractor.enable_hooks()
        model.zero_grad()
        scores = model(img_tensor.unsqueeze(0))

        # Select the class index
        class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx
        class_name = weights.meta["categories"][class_idx]
        print(f"{extractor.__class__.__name__}: class {class_idx} ({class_name})")

        # Use the hooked data to compute activation map
        activation_map = extractor(class_idx, scores)[0].squeeze(0).cpu()

        # Clean data
        extractor.disable_hooks()
        extractor.remove_hooks()
        # Convert it to PIL image
        # The indexing below means first image in batch
        heatmap = to_pil_image(activation_map, mode="F")
        # Plot the result
        result = overlay_mask(pil_img, heatmap, alpha=args.alpha)

        ax = axes[idx // num_cols][idx % num_cols] if args.rows > 1 else axes[idx] if num_cols > 1 else axes

        ax.imshow(result)
        ax.set_title(f"{extractor.__class__.__name__}: {class_name}", size=8)

    # Clear axes
    if num_cols > 1:
        for axes_ in axes:
            if args.rows > 1:
                for ax in axes_:
                    ax.axis("off")
            else:
                axes_.axis("off")

    else:
        axes.axis("off")

    plt.tight_layout()
    if args.savefig:
        plt.savefig(args.savefig, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
    if not args.noblock:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Saliency Map comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--arch", type=str, default="resnet18", help="Name of the architecture")
    parser.add_argument(
        "--img",
        type=str,
        default="https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg",
        help="The image to extract CAM from",
    )
    parser.add_argument("--class-idx", type=int, default=None, help="Class index to inspect (defaults to prediction)")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Default device to perform computation on",
    )
    parser.add_argument("--savefig", type=str, default=None, help="Path to save figure")
    parser.add_argument("--method", type=str, default=None, help="CAM method to use")
    parser.add_argument("--target", type=str, default=None, help="the target layer")
    parser.add_argument("--alpha", type=float, default=0.5, help="Transparency of the heatmap")
    parser.add_argument("--rows", type=int, default=1, help="Number of rows for the layout")
    parser.add_argument(
        "--noblock",
        dest="noblock",
        help="Disables blocking visualization",
        action="store_true",
    )
    args = parser.parse_args()

    main(args)
