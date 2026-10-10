import numpy as np
import pytest
import torch
from torch import nn
from torchvision.models import get_model

from torchcam.methods import EigenCAM


def _extract(features, *, normalized=True, **kwargs):
    model = nn.Sequential(nn.Identity()).eval()
    with EigenCAM(model, "0", **kwargs) as extractor:
        model(features)
        return extractor(normalized=normalized)[0]


@pytest.mark.parametrize("shape", [(2, 4, 3, 5), (2, 6, 2, 3, 4)])
@pytest.mark.parametrize("normalized", [False, True])
def test_eigencam_matches_numpy_projection(shape, normalized):
    features = np.random.default_rng(4).normal(size=shape)
    expected = []
    for sample in features:
        matrix = sample.reshape(shape[1], -1).T.copy()
        matrix -= matrix.mean(axis=0)
        _, _, vectors = np.linalg.svd(matrix, full_matrices=False)
        direction = vectors[0]
        direction *= np.sign(direction[np.argmax(np.abs(direction))])
        projection = np.maximum(matrix @ direction, 0).reshape(shape[2:])
        if normalized:
            projection -= projection.min()
            projection /= projection.max() + 1e-8
        expected.append(projection)
    actual = _extract(torch.from_numpy(features), normalized=normalized)
    torch.testing.assert_close(actual, torch.from_numpy(np.stack(expected)))


def test_eigencam_rank_one_centering_and_channel_axis():
    pattern = torch.tensor([[-3.0, -1.0, 0.0], [1.0, 2.0, 7.0]], dtype=torch.float64)
    loading = torch.tensor([1.0, 2.0, -1.0], dtype=torch.float64)
    offsets = torch.tensor([50.0, -30.0, 10.0], dtype=torch.float64)
    features = (loading[:, None, None] * pattern + offsets[:, None, None]).unsqueeze(0)
    expected = ((pattern - pattern.mean()) * loading.norm()).relu().unsqueeze(0)
    torch.testing.assert_close(_extract(features, normalized=False), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_eigencam_dtype_and_batch_independence(dtype):
    generator = torch.Generator().manual_seed(7)
    features = torch.randn(3, 4, 5, 7, generator=generator).to(dtype)
    expected_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    actual = _extract(features)
    assert actual.dtype == expected_dtype
    assert actual.device == features.device
    assert torch.isfinite(actual).all()
    assert actual.min() == 0
    assert actual.max() <= 1
    expected = torch.cat([_extract(sample.unsqueeze(0)) for sample in features])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, _extract(features.to(expected_dtype)))


@pytest.mark.parametrize("shape", [(2, 1, 3, 5), (2, 4, 1, 1), (2, 4, 3, 5)])
@pytest.mark.parametrize("value", [0.0, 7.0])
@pytest.mark.parametrize("normalized", [False, True])
def test_eigencam_constant_features(shape, value, normalized):
    actual = _extract(torch.full(shape, value), normalized=normalized)
    torch.testing.assert_close(actual, torch.zeros((shape[0], *shape[2:])))


def test_eigencam_single_channel_and_noncontiguous_features():
    features = torch.tensor([[[[1.0, 3.0, 0.0], [4.0, 7.0, 5.0]]]]).transpose(-1, -2)
    assert not features.is_contiguous()
    expected = (features[:, 0] - features.mean()).relu()
    torch.testing.assert_close(_extract(features, normalized=False), expected)


def test_eigencam_is_invariant_to_svd_sign(monkeypatch):
    features = torch.randn(2, 4, 3, 5, generator=torch.Generator().manual_seed(4))
    expected = _extract(features)
    svd = torch.linalg.svd

    def opposite_sign(*args, **kwargs):
        left, singular, right = svd(*args, **kwargs)
        return -left, singular, -right

    monkeypatch.setattr(torch.linalg, "svd", opposite_sign)
    torch.testing.assert_close(_extract(features), expected)


class _TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 4, 1)
        self.pool = nn.AvgPool2d(2)
        self.forwards = 0

    def forward(self, input_tensor):
        self.forwards += 1
        return self.pool(self.features(input_tensor)).mean((2, 3))


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_eigencam_one_forward_no_gradients_and_multiple_layers(context):
    model = _TinyCNN().eval()
    features = torch.randn(2, 3, 6, 8)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    with EigenCAM(model, ["features", "pool"]) as extractor, context():
        output = model(features)
        hooked = [activation.clone() for activation in extractor.hook_a]
        expected = extractor()
        for class_idx in (0, 3, [1, 2]):
            for actual, reference in zip(extractor(class_idx, output), expected, strict=True):
                torch.testing.assert_close(actual, reference)
        assert [cam.shape for cam in expected] == [(2, 6, 8), (2, 3, 4)]
        assert all(not cam.requires_grad for cam in expected)
        for activation, saved in zip(extractor.hook_a, hooked, strict=True):
            torch.testing.assert_close(activation, saved)
    assert model.forwards == 1
    assert not model.features._forward_hooks
    assert not model.pool._forward_hooks
    assert all(activation is None for activation in extractor.hook_a)
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)


def test_eigencam_transformer_reshape():
    features = torch.randn(2, 16, 4)

    def reshape(tokens):
        return tokens.transpose(1, 2).reshape(2, 4, 4, 4)

    torch.testing.assert_close(_extract(features, reshape_transform=reshape), _extract(reshape(features)))


def test_eigencam_torchvision_automatic_target():
    model = get_model("resnet18", weights=None).eval().requires_grad_(False)
    with EigenCAM(model, input_shape=(3, 32, 32)) as extractor, torch.inference_mode():
        model(torch.rand(1, 3, 32, 32))
        maps = extractor()
        assert len(maps) == 1
        assert maps[0].shape == (1, 2, 2)
        assert torch.isfinite(maps[0]).all()
        assert maps[0].min() >= 0
        assert maps[0].max() <= 1


@pytest.mark.parametrize(
    "features",
    [
        torch.zeros(2, 3),
        torch.zeros(0, 3, 4, 5),
        torch.zeros(2, 0, 4, 5),
        torch.zeros(2, 3, 0, 5),
        torch.ones(2, 3, 4, 5, dtype=torch.int64),
        torch.full((2, 3, 4, 5), float("nan")),
        torch.full((2, 3, 4, 5), float("inf")),
    ],
)
def test_eigencam_rejects_invalid_features(features):
    with pytest.raises(ValueError, match="EigenCAM requires"):
        _extract(features)


def test_eigencam_hook_and_target_validation():
    model = nn.Sequential(nn.Identity())
    with EigenCAM(model, "0") as extractor:
        with pytest.raises(AssertionError, match="forwarded"):
            extractor()
        model(torch.rand(2, 3, 4, 5))
        with pytest.raises(ValueError, match="does not support arbitrary"):
            extractor(targets=lambda output: output.sum())
        with pytest.raises(ValueError, match="length of `class_idx`"):
            extractor([0])
        with pytest.raises(ValueError, match="Incorrect `class_idx`"):
            extractor(-1)
        extractor.reset_hooks()
        with pytest.raises(AssertionError, match="forwarded"):
            extractor()


def test_eigencam_cpu_autocast():
    features = torch.randn(2, 4, 3, 5, generator=torch.Generator().manual_seed(4))
    expected = _extract(features)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = _extract(features)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)


def test_eigencam_does_not_hide_unsupported_cpu_svd(monkeypatch):
    def unsupported_svd(*_args, **_kwargs):
        raise NotImplementedError("unsupported SVD")

    monkeypatch.setattr(torch.linalg, "svd", unsupported_svd)
    with pytest.raises(NotImplementedError, match="unsupported SVD"):
        _extract(torch.rand(2, 4, 3, 5))


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_eigencam_accelerator(device):
    if (device == "cuda" and not torch.cuda.is_available()) or (
        device == "mps" and not torch.backends.mps.is_available()
    ):
        pytest.skip(f"{device} unavailable")
    features = torch.randn(2, 4, 3, 5, generator=torch.Generator().manual_seed(4))
    actual = _extract(features.to(device))
    assert actual.device.type == device
    torch.testing.assert_close(actual.cpu(), _extract(features), atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_eigencam_mps_svd_fallback(monkeypatch):
    svd = torch.linalg.svd
    devices = []

    def cpu_only_svd(features, **kwargs):
        devices.append(features.device.type)
        if features.device.type == "mps":
            raise NotImplementedError("SVD is unavailable on this MPS version")
        return svd(features, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", cpu_only_svd)
    features = torch.randn(2, 4, 3, 5, generator=torch.Generator().manual_seed(4))
    actual = _extract(features.to("mps"))
    assert devices == ["mps", "cpu"]
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), _extract(features), atol=2e-5, rtol=2e-5)
