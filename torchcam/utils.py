# Copyright (C) 2020-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from typing import cast

import numpy as np
from matplotlib import colormaps as cm
from PIL.Image import Image, Resampling, fromarray


def overlay_mask(img: Image, mask: Image, colormap: str = "jet", alpha: float = 0.7) -> Image:
    """Overlay a colormapped mask on a background image

    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)

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

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Resampling.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    return fromarray((alpha * np.asarray(img) + (1 - alpha) * cast(np.ndarray, overlay)).astype(np.uint8))
