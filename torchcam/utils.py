# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Colormap
from PIL.Image import Image, Resampling, blend, fromarray


def overlay_mask(img: Image, mask: Image, colormap: Colormap | str = "jet", alpha: float = 0.7) -> Image:
    """Overlay a colormapped mask on a background image.

    Example:
        ```python
        from PIL import Image
        import matplotlib.pyplot as plt
        from torchcam.utils import overlay_mask
        img = ...
        cam = ...
        overlay = overlay_mask(img, cam)
        ```

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """
    if not isinstance(img, Image) or not isinstance(mask, Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    if len(img.getbands()) not in {1, 3}:
        raise ValueError("img argument needs to be a grayscale or RGB image")

    cmap = colormaps.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Resampling.BICUBIC)
    overlay = cmap(np.asarray(overlay) ** 2, bytes=True)[:, :, :3]
    if img.mode in {"L", "RGB"} and overlay.ndim == 3 and np.isfinite(alpha):
        return blend(fromarray(overlay), img.convert("RGB"), alpha)
    # Overlay the image with the mask
    bg_img = np.asarray(img) if len(img.getbands()) == 3 else np.asarray(img)[..., np.newaxis].repeat(3, axis=-1)
    return fromarray((alpha * bg_img + (1 - alpha) * overlay).astype(np.uint8))
