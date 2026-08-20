import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import Colormap, ListedColormap
from PIL import Image
from PIL.Image import Resampling

from torchcam import utils


def _legacy_overlay_mask(
    img: Image.Image, mask: Image.Image, colormap: Colormap | str = "jet", alpha: float = 0.7
) -> Image.Image:
    cmap = colormaps.get_cmap(colormap)
    overlay = mask.resize(img.size, resample=Resampling.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    bg_img = np.asarray(img) if len(img.getbands()) == 3 else np.asarray(img)[..., np.newaxis].repeat(3, axis=-1)
    return Image.fromarray((alpha * bg_img + (1 - alpha) * overlay).astype(np.uint8))


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 3)])
def test_overlay_mask(shape: tuple[int, ...]):
    img = Image.fromarray(np.zeros(shape, dtype=np.uint8))
    mask = Image.fromarray(np.full((4, 4), 255, dtype=np.uint8))

    actual = utils.overlay_mask(img, mask, alpha=0.7)
    expected = np.zeros((4, 4, 3), dtype=np.uint8)
    expected[..., 2] = 39

    assert isinstance(actual, Image.Image)
    np.testing.assert_array_equal(np.asarray(actual), expected)


@pytest.mark.parametrize("shape", [(7, 9), (7, 9, 3)])
@pytest.mark.parametrize("alpha", [0.0, 0.5, 0.999])
@pytest.mark.parametrize("colormap", ["jet", pytest.param(ListedColormap(["black", "red", "white"]), id="custom")])
def test_overlay_mask_parity(shape: tuple[int, ...], alpha: float, colormap: Colormap | str):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
    mask = Image.fromarray(rng.integers(0, 256, (4, 5), dtype=np.uint8))

    expected = np.asarray(_legacy_overlay_mask(img, mask, colormap, alpha))
    actual = utils.overlay_mask(img, mask, colormap, alpha)
    difference = np.abs(expected.astype(np.int16) - np.asarray(actual).astype(np.int16))

    assert actual.mode == "RGB"
    assert actual.size == img.size
    assert np.asarray(actual).dtype == np.uint8
    assert difference.max() <= 1


@pytest.mark.parametrize(("mode", "shape"), [("P", (5, 6)), ("HSV", (5, 6, 3))])
def test_overlay_mask_preserves_raw_band_modes(mode: str, shape: tuple[int, ...]):
    values = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    img = Image.frombytes(mode, (shape[1], shape[0]), values.tobytes())
    mask = Image.fromarray(np.arange(12, dtype=np.uint8).reshape(3, 4))

    expected = np.asarray(_legacy_overlay_mask(img, mask))
    actual = np.asarray(utils.overlay_mask(img, mask))

    assert np.abs(expected.astype(np.int16) - actual.astype(np.int16)).max() <= 1


def test_overlay_mask_preserves_nan_alpha():
    img = Image.fromarray(np.arange(60, dtype=np.uint8).reshape(4, 5, 3))
    mask = Image.fromarray(np.arange(20, dtype=np.uint8).reshape(4, 5))

    with pytest.warns(RuntimeWarning):
        expected = np.asarray(_legacy_overlay_mask(img, mask, alpha=float("nan")))
    with pytest.warns(RuntimeWarning):
        actual = np.asarray(utils.overlay_mask(img, mask, alpha=float("nan")))

    np.testing.assert_array_equal(actual, expected)


def test_overlay_mask_errors():
    img = Image.new("RGB", (4, 4))
    mask = Image.new("L", (4, 4))

    with pytest.raises(TypeError, match=r"img and mask arguments need to be PIL\.Image"):
        utils.overlay_mask(np.zeros((4, 4, 3)), mask)  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match=r"img and mask arguments need to be PIL\.Image"):
        utils.overlay_mask(img, np.zeros((4, 4)))  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError):
        utils.overlay_mask(img, Image.new("RGB", (4, 4)))
    with pytest.raises(ValueError, match="img argument needs to be a grayscale or RGB image"):
        utils.overlay_mask(Image.new("RGBA", (4, 4)), mask)
    with pytest.raises(ValueError):
        utils.overlay_mask(img, mask, colormap="missing")


@pytest.mark.parametrize("alpha", [0, -0.1, 1.0, "0.5", None])
def test_overlay_mask_invalid_alpha(alpha: object):
    with pytest.raises(ValueError, match="alpha argument is expected to be of type float between 0 and 1"):
        utils.overlay_mask(
            Image.new("RGB", (4, 4)),
            Image.new("L", (4, 4)),
            alpha=alpha,  # ty: ignore[invalid-argument-type]
        )
