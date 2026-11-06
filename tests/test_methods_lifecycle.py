import weakref

import pytest
import torch
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from torchcam import methods
from torchcam.methods.core import _CAM


def _setup(method):
    if method is methods.LeGrad:
        model = VisionTransformer(
            image_size=8, patch_size=4, num_layers=2, num_heads=2, hidden_dim=8, mlp_dim=16, num_classes=2
        ).eval()
        return model, {"target_layer": list(model.encoder.layers)}, torch.rand(1, 3, 8, 8)
    model = nn.Sequential(
        nn.Conv2d(3, 2, 1), nn.ReLU(), nn.Conv2d(2, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 2)
    ).eval()
    kwargs = {"target_layer": "2" if method is methods.CAM else ["0", "2"]}
    if method in {methods.SSCAM, methods.ISCAM, methods.SmoothGradCAMpp}:
        kwargs["num_samples"] = 2
    return model, kwargs, torch.rand(1, 3, 4, 4)


def _hook_counts(model):
    return [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in model.modules()]


def _captured_tensors(extractor):
    base = getattr(extractor, "base_cam", extractor)
    references = []
    for name in ("hook_a", "hook_g", "_hook_outputs", "hook_attn", "_input"):
        captured = getattr(base, name, None)
        values = captured if isinstance(captured, list) else [captured]
        references.extend(weakref.ref(value) for value in values if isinstance(value, torch.Tensor))
    return references


@pytest.mark.parametrize(
    "method",
    [
        methods.CAM,
        methods.EigenCAM,
        methods.GradCAM,
        methods.ScoreCAM,
        methods.SSCAM,
        methods.ISCAM,
        methods.SmoothGradCAMpp,
        methods.RefineCAM,
        methods.FinerCAM,
        methods.LeGrad,
    ],
)
@pytest.mark.parametrize("context_exit", [False, True])
def test_close_releases_captures_and_rejects_reuse(method, context_exit):
    model, kwargs, input_tensor = _setup(method)
    with model.register_forward_hook(lambda *_args: None):
        original_hooks = _hook_counts(model)
        extractor = method(model, **kwargs)
        assert not extractor.closed
        # Discard outputs and maps so only the extractor retains captures.
        extractor(0, model(input_tensor))
        references = _captured_tensors(extractor)
        assert references
        assert all(reference() is not None for reference in references)
        if context_exit:
            with extractor:
                pass
        else:
            extractor.remove_hooks()
        assert extractor.closed
        assert _hook_counts(model) == original_hooks
        assert all(reference() is None for reference in references)
        extractor.remove_hooks()
        extractor.reset_hooks()
        extractor.disable_hooks()
        for action in (extractor.enable_hooks, extractor.__enter__):
            with pytest.raises(RuntimeError, match="closed; create a new extractor"):
                action()
        scores = model(input_tensor)
        for action in (extractor, extractor.compute_cams):
            with pytest.raises(RuntimeError, match="closed; create a new extractor"):
                action(0, scores)
        assert _hook_counts(model) == original_hooks


@pytest.mark.parametrize("method", [methods.ScoreCAM, methods.SmoothGradCAMpp])
def test_reset_preserves_hook_ownership_and_allows_fresh_forward(method):
    model, kwargs, input_tensor = _setup(method)
    with method(model, **kwargs) as extractor:
        model(input_tensor)
        references = _captured_tensors(extractor)
        original_hooks = _hook_counts(model)
        extractor.reset_hooks()
        assert not extractor.closed
        assert _hook_counts(model) == original_hooks
        assert all(reference() is None for reference in references)
        with pytest.raises(AssertionError, match="forwarded"):
            extractor(0)
        scores = model(input_tensor)
        captured_input = extractor._input
        extractor.disable_hooks()
        assert not extractor.closed
        model(input_tensor + 1)
        assert extractor._input is captured_input
        assert not extractor._hooks_enabled
        extractor.enable_hooks()
        assert extractor._hooks_enabled
        assert all(torch.isfinite(cam).all() for cam in extractor(0, scores))


def test_close_inside_nested_capture_suspension_stays_closed():
    model, kwargs, _ = _setup(methods.GradCAM)
    extractor = methods.GradCAM(model, **kwargs)
    with extractor._hooks_off(), extractor._hooks_off():
        extractor.remove_hooks()
    assert extractor.closed
    assert not extractor._hooks_enabled
    with pytest.raises(RuntimeError, match="closed"), extractor._hooks_off():
        pass


@pytest.mark.parametrize(
    ("method", "fail_at"),
    [(_CAM, 2), (methods.GradCAM, 3), (methods.ScoreCAM, 3), (methods.SmoothGradCAMpp, 5), (methods.LeGrad, 6)],
)
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_hook_registration_is_rolled_back(method, fail_at, error_type, monkeypatch):
    model, kwargs, input_tensor = _setup(method)
    with model.register_forward_hook(lambda *_args: None), model.register_forward_pre_hook(lambda *_args: None):
        original_hooks = _hook_counts(model)
        expected = model(input_tensor)
        registered = []

        def failing_registration(register):
            def register_hook(module, *args, **options):
                if len(registered) + 1 == fail_at:
                    raise error_type("registration interrupted")
                handle = register(module, *args, **options)
                registered.append(handle)
                return handle

            return register_hook

        with monkeypatch.context() as patch:
            for name in ("register_forward_hook", "register_forward_pre_hook"):
                patch.setattr(nn.Module, name, failing_registration(getattr(nn.Module, name)))
            with pytest.raises(error_type, match="registration interrupted"):
                method(model, **kwargs)
        assert registered
        assert _hook_counts(model) == original_hooks
        torch.testing.assert_close(model(input_tensor), expected)


@pytest.mark.parametrize("fc_layer", [None, "missing", nn.Identity(), 123])
def test_cam_head_validation_failure_removes_hooks(fc_layer):
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1))
    with model[0].register_forward_hook(lambda *_args: None):
        original_hooks = _hook_counts(model)
        with pytest.raises((ValueError, KeyError, TypeError)):
            methods.CAM(model, "0", fc_layer=fc_layer)
        assert _hook_counts(model) == original_hooks


@pytest.mark.parametrize("method", [methods.SSCAM, methods.SmoothGradCAMpp])
def test_invalid_noise_distribution_removes_hooks(method):
    model, kwargs, _ = _setup(method)
    with pytest.raises(ValueError):
        method(model, **kwargs, std=-1)
    assert all(count == (0, 0) for count in _hook_counts(model))
