import numpy as np
from PIL import Image

from torchcam import utils


def test_overlay_mask():
    img = Image.fromarray(np.zeros((4, 4, 3)).astype(np.uint8))
    mask = Image.fromarray(255 * np.ones((4, 4)).astype(np.uint8))

    overlayed = utils.overlay_mask(img, mask, alpha=0.7)

    # Check object type
    assert isinstance(overlayed, Image.Image)
    # Verify value
    assert np.all(np.asarray(overlayed)[..., 0] == 0)
    assert np.all(np.asarray(overlayed)[..., 1] == 0)
    assert np.all(np.asarray(overlayed)[..., 2] == 39)
