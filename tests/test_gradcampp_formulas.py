import pytest
import torch
from torch import nn

from torchcam.methods import GradCAMpp, SmoothGradCAMpp


class _SpatialClassifier(nn.Module):
    def __init__(self, spatial_shape):
        super().__init__()
        self.features = nn.Identity()
        self.register_buffer(
            "coefficients",
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 0.0, -1.0, 2.0]], dtype=torch.float64).reshape(
                1, 2, *spatial_shape
            ),
        )

    def forward(self, input_tensor):
        score = (self.features(input_tensor) * self.coefficients).flatten(1).sum(1)
        return torch.stack((score, -score), dim=1)


@pytest.mark.parametrize("spatial_shape", [(4,), (2, 2), (1, 2, 2)])
@pytest.mark.parametrize("method", [GradCAMpp, SmoothGradCAMpp])
def test_gradcampp_spatially_varying_gradient_formula(method, spatial_shape, monkeypatch):
    model = _SpatialClassifier(spatial_shape)
    input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]], dtype=torch.float64)
    input_tensor = input_tensor.reshape(1, 2, *spatial_shape).requires_grad_()
    with method(model, "features") as extractor:
        if isinstance(extractor, SmoothGradCAMpp):
            monkeypatch.setattr(extractor._distrib, "sample", torch.zeros)
        weights = extractor._get_weights(0, model(input_tensor))[0]

    # Both activation channels sum to 10. Each positive spatial derivative g
    # contributes g / (2 + 10*g); zero and negative derivatives contribute zero.
    expected = torch.tensor([[1 / 12 + 2 / 22 + 3 / 32 + 4 / 42, 1 / 12 + 2 / 22]], dtype=torch.float64)
    torch.testing.assert_close(weights, expected, atol=1e-9, rtol=1e-9)


class _ThresholdClassifier(nn.Module):
    def __init__(self, signed=False):
        super().__init__()
        self.features = nn.Identity()
        self.fail = False
        self.signed = signed

    def forward(self, input_tensor):
        features = self.features(input_tensor)
        if self.fail:
            raise RuntimeError("noisy forward failed")
        shifted = features - 1
        score = (shifted.abs() if self.signed else shifted.relu()).flatten(1).sum(1)
        return torch.stack((score, -score), dim=1)


@pytest.mark.parametrize(("signed", "expected_weight"), [(False, 0.25), (True, 0.0)])
def test_smoothgradcampp_uses_original_features_and_averaged_gradients(monkeypatch, signed, expected_weight):
    input_tensor = torch.tensor([[[[0.5, 1.5]]]], requires_grad=True)
    cams = []
    for noise_order in ([1.0, -1.0], [-1.0, 1.0]):
        model = _ThresholdClassifier(signed)
        noise = iter(noise_order)
        with SmoothGradCAMpp(model, "features", num_samples=2) as extractor:
            monkeypatch.setattr(extractor._distrib, "sample", lambda shape, noise=noise: torch.full(shape, next(noise)))
            model(input_tensor)
            original_activation = extractor.hook_a[0]
            original_output = extractor._hook_outputs[0]
            cams.append(extractor(0, normalized=False)[0])
            assert extractor.hook_a[0] is original_activation
            assert extractor._hook_outputs[0] is original_output
            torch.testing.assert_close(extractor._input, input_tensor)

    # For ReLU, noisy gradients are all ones or zeros: each moment averages to 1/2,
    # alpha is 1/4 and the channel weight is 1/4. For abs, gradients are +/-1,
    # so ReLU(mean(gradient)) is zero, unlike mean(ReLU(gradient)).
    expected = input_tensor.detach().squeeze(1) * expected_weight
    for cam in cams:
        torch.testing.assert_close(cam, expected)


@pytest.mark.parametrize("previous_ihook_enabled", [False, True])
def test_smoothgradcampp_restores_original_forward_after_error(previous_ihook_enabled):
    model = _ThresholdClassifier()
    input_tensor = torch.tensor([[[[0.5, 1.5]]]], requires_grad=True)
    with SmoothGradCAMpp(model, "features", num_samples=2) as extractor:
        model(input_tensor)
        original_activation = extractor.hook_a[0]
        original_output = extractor._hook_outputs[0]
        extractor._ihook_enabled = previous_ihook_enabled
        model.fail = True
        with pytest.raises(RuntimeError, match="noisy forward failed"):
            extractor(0)
        assert extractor.hook_a[0] is original_activation
        assert extractor._hook_outputs[0] is original_output
        assert extractor._ihook_enabled is previous_ihook_enabled
        torch.testing.assert_close(extractor._input, input_tensor)


def test_smoothgradcampp_disabled_hooks_preserve_cached_input():
    model = _ThresholdClassifier()
    input_tensor = torch.tensor([[[[0.5, 1.5]]]], requires_grad=True)
    with SmoothGradCAMpp(model, "features") as extractor:
        model(input_tensor)
        with extractor._hooks_off():
            model(torch.zeros_like(input_tensor))
        torch.testing.assert_close(extractor._input, input_tensor)
