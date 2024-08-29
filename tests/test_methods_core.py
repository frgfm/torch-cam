import pytest
import torch

from torchcam.methods import core


def test_cam_constructor(mock_img_model):
    model = mock_img_model.eval()
    # Check that wrong target_layer raises an error
    with pytest.raises(ValueError):
        core._CAM(model, "3")

    # Wrong types
    with pytest.raises(TypeError):
        core._CAM(model, 3)
    with pytest.raises(TypeError):
        core._CAM(model, [3])

    # Unrelated module
    with pytest.raises(ValueError):
        core._CAM(model, torch.nn.ReLU())


def test_cam_context_manager(mock_img_model):
    model = mock_img_model.eval()
    with core._CAM(model):
        # Model is hooked
        assert sum(len(mod._forward_hooks) for mod in model.modules()) == 1
    # Exit should remove hooks
    assert all(len(mod._forward_hooks) == 0 for mod in model.modules())


def test_cam_precheck(mock_img_model, mock_img_tensor):
    model = mock_img_model.eval()
    with core._CAM(model, "0.3") as extractor, torch.no_grad():
        # Check missing forward raises Error
        with pytest.raises(AssertionError):
            extractor(0)

        # Correct forward
        model(mock_img_tensor)

        # Check incorrect class index
        with pytest.raises(ValueError):
            extractor(-1)

        # Check incorrect class index
        with pytest.raises(ValueError):
            extractor([-1])

        # Check missing score
        if extractor._score_used:
            with pytest.raises(ValueError):
                extractor(0)


@pytest.mark.parametrize(
    ("input_shape", "spatial_dims"),
    [
        ((8, 8), None),
        ((8, 8, 8), None),
        ((8, 8, 8), 2),
        ((8, 8, 8, 8), None),
        ((8, 8, 8, 8), 3),
    ],
)
def test_cam_normalize(input_shape, spatial_dims):
    input_tensor = torch.rand(input_shape)
    normalized_tensor = core._CAM._normalize(input_tensor, spatial_dims)
    # Shape check
    assert normalized_tensor.shape == input_shape
    # Value check
    assert not torch.any(torch.isnan(normalized_tensor))
    assert torch.all(normalized_tensor <= 1)
    assert torch.all(normalized_tensor >= 0)


def test_cam_remove_hooks(mock_img_model):
    model = mock_img_model.eval()
    with core._CAM(model, "0.3") as extractor:
        assert len(extractor.hook_handles) == 1
        # Check that there is only one hook on the model
        assert all(act is None for act in extractor.hook_a)
        with torch.no_grad():
            _ = model(torch.rand((1, 3, 32, 32)))
        assert all(isinstance(act, torch.Tensor) for act in extractor.hook_a)

        # Remove it
        extractor.remove_hooks()
        assert len(extractor.hook_handles) == 0
        # Reset the hooked values
        extractor.reset_hooks()
        with torch.no_grad():
            _ = model(torch.rand((1, 3, 32, 32)))
        assert all(act is None for act in extractor.hook_a)


def test_cam_repr(mock_img_model):
    model = mock_img_model.eval()
    with core._CAM(model, "0.3") as extractor:
        assert repr(extractor) == "_CAM(target_layer=['0.3'])"


def test_fuse_cams():
    with pytest.raises(TypeError):
        core._CAM.fuse_cams(torch.zeros((3, 32, 32)))

    with pytest.raises(ValueError):
        core._CAM.fuse_cams([])

    cams = [torch.rand((1, 32, 32)), torch.rand((1, 16, 16))]

    # Single CAM
    assert torch.equal(cams[0], core._CAM.fuse_cams(cams[:1]))

    # Fusion
    cam = core._CAM.fuse_cams(cams)
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == cams[0].ndim
    assert cam.shape == (1, 32, 32)

    # Specify target shape
    cam = core._CAM.fuse_cams(cams, (16, 16))
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == cams[0].ndim
    assert cam.shape == (1, 16, 16)
