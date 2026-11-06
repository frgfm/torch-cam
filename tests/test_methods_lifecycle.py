import weakref

import pytest
import torch
from torch import nn

from torchcam import methods


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [(methods.CAM, {}), (methods.SSCAM, {"std": -1}), (methods.SmoothGradCAMpp, {"std": -1})],
)
def test_failed_initialization_preserves_caller_hooks(method, kwargs):
    model = nn.Sequential(nn.Conv2d(3, 2, 1))  # CAM cannot find a classification head.
    with model[0].register_forward_hook(lambda *_args: None):
        original_hooks = model[0]._forward_hooks.copy()
        with pytest.raises(ValueError):
            method(model, "0", **kwargs)
        assert model[0]._forward_hooks == original_hooks
        assert not model._forward_pre_hooks
        assert not model._forward_hooks


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_registration_is_rolled_back(error_type, monkeypatch):
    model = nn.Sequential(nn.Identity(), nn.Identity())

    def fail(*_args, **_kwargs):
        raise error_type("registration interrupted")

    monkeypatch.setattr(model[1], "register_forward_hook", fail)
    with pytest.raises(error_type, match="registration interrupted"):
        methods.GradCAM(model, ["0", "1"])
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in model.modules())


@pytest.mark.parametrize("method", [methods.GradCAM, methods.EigenCAM, methods.ScoreCAM, methods.SmoothGradCAMpp])
def test_reset_and_close_release_captures(method, mock_img_model):
    model = mock_img_model.eval()
    input_tensor = torch.rand(1, 3, 8, 8, requires_grad=True)
    with method(model, "0.3") as extractor:
        model(input_tensor)
        capture = weakref.ref(extractor.hook_a[0])
        extractor.reset_hooks()
        assert capture() is None
        assert not extractor.closed
        # A fresh forward after reset still produces a valid CAM.
        assert torch.isfinite(extractor(0, model(input_tensor))[0]).all()
        captures = [weakref.ref(value) for value in extractor.hook_a]
        if hasattr(extractor, "_input"):
            captures.append(weakref.ref(extractor._input))
        with extractor._hooks_off():
            extractor.remove_hooks()
        assert extractor.closed
        assert not extractor._hooks_enabled
        assert all(capture() is None for capture in captures)
        for action in (
            extractor.enable_hooks,
            extractor.__enter__,
            lambda: extractor(0),
            lambda: extractor.compute_cams(0),
        ):
            with pytest.raises(RuntimeError, match="closed; create a new extractor"):
                action()
