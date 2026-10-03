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


class _ReducedPrecisionClassifier(nn.Module):
    def __init__(self, coefficients):
        super().__init__()
        self.features = nn.Identity()
        self.register_buffer("coefficients", coefficients)

    def forward(self, input_tensor):
        features = self.features(input_tensor.to(dtype=self.coefficients.dtype))
        score = (features * self.coefficients).flatten(1).sum(1)
        return torch.stack((score, -score), dim=1)


@pytest.mark.parametrize("method", [GradCAMpp, SmoothGradCAMpp])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["small_gradients", "activation_sum", "gradient_moments", "cancellation"])
def test_gradcampp_reduced_precision_matches_double_oracle(method, dtype, case):
    if case == "activation_sum":
        input_tensor = torch.linspace(250, 750, 256, dtype=dtype).reshape(1, 1, 16, 16)
        coefficients = torch.full_like(input_tensor, 0.02)
    elif case == "gradient_moments":
        input_tensor = torch.tensor([[[[1.0, 2.0]]]], dtype=dtype)
        coefficients = torch.full_like(input_tensor, 40.0)
    elif case == "cancellation":
        input_tensor = torch.tensor([[[[1.0, 1.0]], [[1.0, 3.0]]]], dtype=dtype)
        coefficients = torch.tensor([[[[-1.0, -1.0]], [[1.0, 1.0]]]], dtype=dtype)
    else:
        input_tensor = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], dtype=dtype)
        coefficients = torch.tensor([[[[1e-4, 2e-4]], [[0.0, 0.0]]]], dtype=dtype)

    model = _ReducedPrecisionClassifier(coefficients)
    input_tensor.requires_grad_()
    with method(model, "features") as extractor:
        weights = extractor._get_weights(0, model(input_tensor))[0]

    # Use the quantized inputs in a double-precision scalar formula. The model
    # is linear, so noise cannot change any derivative or its three moments.
    expected = []
    for activations, gradients in zip(input_tensor[0].double(), coefficients[0].double(), strict=True):
        activation_sum = activations.sum().item()
        expected.append(
            sum(g**2 / (2 * g**2 + g**3 * activation_sum + 1e-8) * max(g, 0) for g in gradients.flatten().tolist())
        )
    assert weights.dtype == torch.float32
    torch.testing.assert_close(weights, torch.tensor([expected]), rtol=2e-6, atol=1e-9)


class _UnusedChannelClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(1, 2, kernel_size=1, bias=False)
        with torch.no_grad():
            self.features.weight.fill_(1)

    def forward(self, input_tensor):
        features = self.features(input_tensor)
        score = features[:, 0].flatten(1).mean(1)
        return torch.stack((score, -score), dim=1)


@pytest.mark.parametrize("method", [GradCAMpp, SmoothGradCAMpp])
def test_gradcampp_autocast_zero_gradient_channel_keeps_valid_map(method):
    model = _UnusedChannelClassifier().eval()
    input_tensor = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
    with torch.autocast("cpu", dtype=torch.float16), method(model, "features") as extractor:
        cam = extractor(0, model(input_tensor))[0]

    # The second channel has no path to the score. It must contribute zero,
    # while the first channel must retain this linear model's spatial ranking.
    expected = torch.tensor([[[0.0, 1 / 3], [2 / 3, 1.0]]])
    assert cam.isfinite().all()
    torch.testing.assert_close(cam, expected, atol=5e-4, rtol=5e-4)
