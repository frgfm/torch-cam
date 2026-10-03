from operator import itemgetter

import pytest
import torch

from torchcam.methods import core


def test_cam_constructor(mock_img_model):
    model = mock_img_model.eval()
    # Check that wrong target_layer raises an error
    with pytest.raises(ValueError, match=r"\['3'\]") as exc_info:
        core._CAM(model, "3")
    assert "closest" not in str(exc_info.value)
    with pytest.raises(ValueError, match=r"'0\.30': \['0\.3'"):
        core._CAM(model, "0.30")
    with pytest.raises(ValueError, match="named_modules"):
        core._CAM(torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.Linear(12, 1)), input_shape=(3, 2, 2))

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


def test_cam_context_manager_cleans_up_on_exception(mock_img_model):
    model = mock_img_model.eval()
    with pytest.raises(RuntimeError), core._CAM(model, "0.3") as extractor:
        raise RuntimeError("boom")
    assert len(extractor.hook_handles) == 0
    assert all(len(mod._forward_hooks) == 0 for mod in model.modules())


def test_cam_hooks_off_restores_state(mock_img_model):
    model = mock_img_model.eval()
    with core._CAM(model, "0.3", enable_hooks=False) as extractor:
        assert not extractor._hooks_enabled
        with extractor._hooks_off():
            assert not extractor._hooks_enabled
        assert not extractor._hooks_enabled

        extractor.enable_hooks()
        with extractor._hooks_off():
            assert not extractor._hooks_enabled
        assert extractor._hooks_enabled


def test_cam_eval_mode_restores_state(mock_img_model, monkeypatch):
    model = mock_img_model.train()
    model[0][1].eval()
    with core._CAM(model, "0.3") as extractor:
        with extractor._eval_mode():
            assert all(not module.training for module in extractor.model.modules())
        assert extractor.model.training
        assert model[0][0].training
        assert not model[0][1].training

        modes = [module.training for module in model.modules()]

        def failing_eval():
            for module in model.modules():
                module.training = False
            raise RuntimeError("boom")

        monkeypatch.setattr(model, "eval", failing_eval)
        with pytest.raises(RuntimeError), extractor._eval_mode():
            pass
        assert [module.training for module in model.modules()] == modes


def test_cam_remove_hooks_idempotent(mock_img_model):
    model = mock_img_model.eval()
    with core._CAM(model, "0.3") as extractor:
        extractor.remove_hooks()
        extractor.remove_hooks()
        assert len(extractor.hook_handles) == 0


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


def test_output_target_validation(mock_img_model):
    with core._CAM(mock_img_model, "0.3") as extractor:
        with pytest.raises(ValueError, match="exactly one"):
            extractor(0, targets=itemgetter(0))
        with pytest.raises(ValueError, match="does not support"):
            extractor(targets=itemgetter(0))

    with pytest.raises(TypeError, match="callable"):
        core._resolve_targets(0, 1)
    with pytest.raises(TypeError, match="callable"):
        core._resolve_targets([0], 1)
    with pytest.raises(ValueError, match="batch size"):
        core._resolve_targets([itemgetter(0)], 2)

    with pytest.raises(ValueError, match="batch dimension"):
        core._target_scores(torch.tensor(1.0), itemgetter(0))
    with pytest.raises(TypeError, match="tensor or a per-sample list"):
        core._target_scores((torch.ones(1), torch.ones(1)), itemgetter(0))

    def number_target(_output):
        return 1

    with pytest.raises(TypeError, match="return a tensor"):
        core._target_scores(torch.ones(1, 1), number_target)


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
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_cam_normalize(input_shape, spatial_dims, noncontiguous):
    input_tensor = torch.rand(input_shape)
    if noncontiguous:
        input_tensor = input_tensor.transpose(-1, -2)
    expected = input_tensor.clone()
    dims = expected.ndim - 1 if spatial_dims is None else spatial_dims
    expected.sub_(expected.flatten(start_dim=-dims).min(-1).values[(...,) + (None,) * dims])
    expected.div_(expected.flatten(start_dim=-dims).max(-1).values[(...,) + (None,) * dims] + 1e-8)
    normalized_tensor = core._CAM._normalize(input_tensor, spatial_dims)
    torch.testing.assert_close(normalized_tensor, expected)
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


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("nonfinite", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fuse_cams_preserves_first_max(requires_grad, nonfinite, dtype):
    value = torch.nan if nonfinite else 2.0
    cams = [
        torch.tensor(values).view(1, 2, 2).requires_grad_(requires_grad)
        for values in ([1.0, value, 3.0, 0.0], [1.0, value, 4.0, value], [0.0, 2.0, 4.0, value])
    ]
    cams[1] = cams[1].to(dtype).detach().requires_grad_(requires_grad)
    resized = [
        torch.nn.functional.interpolate(cam.unsqueeze(1), (4, 4), mode="bilinear", align_corners=False) for cam in cams
    ]
    expected = torch.stack(resized).max(0).values.squeeze(1)
    fused = core._CAM.fuse_cams(cams, (4, 4))
    torch.testing.assert_close(fused, expected, equal_nan=True)
    if requires_grad:
        actual_grads = torch.autograd.grad(fused.sum(), cams)
        expected_grads = torch.autograd.grad(expected.sum(), cams)
        torch.testing.assert_close(actual_grads, expected_grads, equal_nan=True)
