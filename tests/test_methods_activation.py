from operator import itemgetter

import numpy as np
import pytest
import torch
from torch import nn
from torchvision.models import get_model

from torchcam.methods import activation


class _StructuredModel(nn.Module):
    def __init__(self, batched=False):
        super().__init__()
        self.features = nn.Conv2d(1, 2, 1)
        self.batched = batched

    def forward(self, input_tensor):
        scores = self.features(input_tensor).flatten(2).mean(-1)
        if self.batched:
            return {"primary": scores[:, 0], "secondary": scores[:, 1]}
        return [{"primary": row[0], "secondary": row[1]} for row in scores]


def test_base_cam_constructor():
    model = get_model("mobilenet_v2", weights=None).eval()
    model.requires_grad_(False)
    # Check that multiple target layers is disabled for base CAM
    with pytest.raises(ValueError):
        activation.CAM(model, ["classifier.1", "classifier.2"])

    # FC layer checks
    with pytest.raises(TypeError):
        activation.CAM(model, fc_layer=3)


def _verify_cam(activation_map, output_size):
    # Simple verifications
    assert isinstance(activation_map, torch.Tensor)
    assert activation_map.shape == output_size
    assert torch.isfinite(activation_map).all()


def _scorecam_inputs():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 5, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(5, 3),
    ).double()
    model.eval().requires_grad_(False)
    return model, torch.rand((2, 3, 8, 8), dtype=torch.float64)


def _scorecam_kwargs(cam_name, batch_size):
    return {"batch_size": batch_size, **({} if cam_name == "ScoreCAM" else {"num_samples": 2})}


@pytest.mark.parametrize(
    ("fc_layer", "batch_size"),
    [
        (None, 1),
        (None, 2),
        ("classifier.1", 1),
        (lambda m: m.classifier[1], 1),
    ],
)
def test_img_cams(fc_layer, batch_size, mock_img_tensor):
    model = get_model("mobilenet_v2", weights=None).eval()
    model.requires_grad_(False)
    kwargs = {}
    if fc_layer is not None:
        kwargs["fc_layer"] = fc_layer(model) if callable(fc_layer) else fc_layer

    # Hook the corresponding layer in the model
    with activation.CAM(model, **kwargs) as extractor, torch.no_grad():
        scores = model(mock_img_tensor.repeat((batch_size,) + (1,) * (mock_img_tensor.ndim - 1)))
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], (batch_size, 7, 7))
        # Multiple class indices
        _verify_cam(extractor(list(range(batch_size)), scores)[0], (batch_size, 7, 7))


def test_scorecam_torchvision_integration():
    model = get_model("mobilenet_v2", weights=None).eval().requires_grad_(False)
    input_tensor = torch.ones((1, 3, 16, 16))

    with activation.ScoreCAM(model, model.features[1].conv[2]) as extractor, torch.no_grad():
        scores = model(input_tensor)
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], (1, 8, 8))


def test_cam_conv1x1(mock_fullyconv_model):
    with activation.CAM(mock_fullyconv_model, fc_layer="1") as extractor, torch.no_grad():
        scores = mock_fullyconv_model(torch.rand((1, 3, 32, 32)))
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], (1, 32, 32))


@pytest.mark.parametrize("cam_name", ["ScoreCAM", "SSCAM", "ISCAM"])
@pytest.mark.parametrize("class_idx", [1, [1, 2]])
def test_scorecam_chunk_parity(cam_name, class_idx, monkeypatch):
    results = []
    for batch_size in (3, 64):
        model, input_tensor = _scorecam_inputs()
        with activation.__dict__[cam_name](
            model,
            ["1", "3"],
            **_scorecam_kwargs(cam_name, batch_size),
        ) as extractor:
            if batch_size == 64 and cam_name == "ScoreCAM":
                get_weights = extractor._get_score_weights
                monkeypatch.setattr(
                    extractor,
                    "_get_score_weights",
                    lambda acts, *args, get_weights=get_weights, extractor=extractor: get_weights(
                        [extractor._upsample(act) for act in acts], *args
                    ),
                )
            scores = model(input_tensor)
            torch.manual_seed(1)
            results.append(extractor(class_idx, scores))

    assert [cam.shape for cam in results[0]] == [(2, 8, 8), (2, 4, 4)]
    assert all(
        cam.dtype == torch.float64 and cam.device.type == "cpu" and torch.isfinite(cam).all() for cam in results[0]
    )
    for chunked, unchunked in zip(*results, strict=True):
        torch.testing.assert_close(chunked, unchunked)


def test_scorecam_bounds_mask_expansion(monkeypatch):
    model, input_tensor = _scorecam_inputs()
    interpolate = activation.F.interpolate
    expanded_batches = []

    def record_interpolation(input_, *args, **kwargs):
        expanded_batches.append(input_.shape[0] * input_.shape[1])
        return interpolate(input_, *args, **kwargs)

    monkeypatch.setattr(activation.F, "interpolate", record_interpolation)
    with activation.ScoreCAM(model, "3", batch_size=3) as extractor:
        extractor([1, 2], model(input_tensor))
    assert expanded_batches
    assert max(expanded_batches) <= 3


@pytest.mark.parametrize("cam_name", ["ScoreCAM", "SSCAM", "ISCAM"])
def test_scorecam_preserves_disabled_hooks(cam_name):
    model, input_tensor = _scorecam_inputs()
    with activation.__dict__[cam_name](model, ["1", "3"], **_scorecam_kwargs(cam_name, 3)) as extractor:
        scores = model(input_tensor)
        extractor.disable_hooks()
        torch.manual_seed(1)
        _ = extractor(1, scores)
        assert not extractor._hooks_enabled


@pytest.mark.parametrize("cam_name", ["ScoreCAM", "SSCAM", "ISCAM"])
def test_scorecam_restores_state_on_error(cam_name, monkeypatch):
    model, input_tensor = _scorecam_inputs()
    model.train()

    with activation.__dict__[cam_name](model, ["1", "3"], **_scorecam_kwargs(cam_name, 3)) as extractor:
        extractor.enable_hooks()
        scores = model(input_tensor)
        monkeypatch.setattr(
            extractor,
            "_masked_input_chunk",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError):
            extractor(1, scores)

        assert extractor._hooks_enabled
        assert model.training


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("cam_cls", [activation.ScoreCAM, activation.SSCAM, activation.ISCAM])
def test_score_cams_support_per_sample_output_targets(cam_cls, batched):
    model = _StructuredModel(batched).eval()
    kwargs = {"batch_size": 2}
    if cam_cls is not activation.ScoreCAM:
        kwargs["num_samples"] = 2

    with cam_cls(model, "features", **kwargs) as extractor:
        output = model(torch.rand(2, 1, 4, 4))
        cams = extractor(
            scores=output,
            targets=[itemgetter("primary"), itemgetter("secondary")],
        )

    _verify_cam(cams[0], (2, 4, 4))


@pytest.mark.parametrize(
    ("cam_name", "target_layer", "num_samples", "output_size"),
    [
        ("CAM", "0.3", None, (1, 8, 16, 16)),
        ("ScoreCAM", "0.3", None, (1, 8, 16, 16)),
        ("SSCAM", "0.3", 4, (1, 8, 16, 16)),
        ("ISCAM", "0.3", 4, (1, 8, 16, 16)),
    ],
)
def test_video_cams(
    cam_name,
    target_layer,
    num_samples,
    output_size,
    mock_video_model,
    mock_video_tensor,
):
    model = mock_video_model.eval()
    kwargs = {}
    # Speed up testing by reducing the number of samples
    if isinstance(num_samples, int):
        kwargs["num_samples"] = num_samples
    if cam_name != "CAM":
        kwargs["batch_size"] = 3

    # Hook the corresponding layer in the model
    with activation.__dict__[cam_name](model, target_layer, **kwargs) as extractor, torch.no_grad():
        scores = model(mock_video_tensor)
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], output_size)


def _extract_eigencam(features, *, normalized=True, **kwargs):
    model = nn.Sequential(nn.Identity()).eval()
    with activation.EigenCAM(model, "0", **kwargs) as extractor:
        model(features)
        return extractor(normalized=normalized)[0]


@pytest.mark.parametrize("shape", [(2, 4, 3, 5), (1, 6, 2, 3, 4)])
def test_eigencam_matches_numpy_projection(shape):
    features = np.random.default_rng(4).normal(size=shape)
    expected = []
    for sample in features:
        matrix = sample.reshape(shape[1], -1).T.copy()
        matrix -= matrix.mean(axis=0)
        _, _, vectors = np.linalg.svd(matrix, full_matrices=False)
        direction = vectors[0]
        direction *= np.sign(direction[np.argmax(np.abs(direction))])
        projection = np.maximum(matrix @ direction, 0).reshape(shape[2:])
        projection -= projection.min()
        projection /= projection.max() + 1e-8
        expected.append(projection)
    # The projection is centered per channel, oriented by the largest loading and min-max normalized
    torch.testing.assert_close(_extract_eigencam(torch.from_numpy(features)), torch.from_numpy(np.stack(expected)))


def test_eigencam_is_class_independent_for_frozen_encoders():
    model = nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU()), nn.Flatten(1)).eval()
    model.requires_grad_(False)

    with activation.EigenCAM(model, "0") as extractor, torch.inference_mode():
        embedding = model(torch.rand(2, 3, 8, 8))
        expected = extractor()
        # Class indices are accepted for API compatibility, but they do not change the map
        for class_idx in (0, 3, [1, 0]):
            for actual, reference in zip(extractor(class_idx, embedding), expected, strict=True):
                torch.testing.assert_close(actual, reference)

    assert embedding.shape == (2, 256)
    _verify_cam(expected[0], (2, 8, 8))
    assert not expected[0].requires_grad


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_eigencam_dtype(dtype):
    features = torch.randn(2, 4, 3, 5, generator=torch.Generator().manual_seed(4)).to(dtype)
    actual = _extract_eigencam(features)
    assert actual.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert actual.device == features.device
    torch.testing.assert_close(actual, _extract_eigencam(features.to(actual.dtype)), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("normalized", [False, True])
def test_eigencam_constant_features(normalized):
    actual = _extract_eigencam(torch.full((2, 3, 5, 7), 4.0), normalized=normalized)
    torch.testing.assert_close(actual, torch.zeros((2, 5, 7)))


@pytest.mark.parametrize(
    "features",
    [
        torch.zeros(2, 3),
        torch.zeros(0, 3, 4, 5),
        torch.ones(2, 3, 4, 5, dtype=torch.int64),
        torch.full((2, 3, 4, 5), float("nan")),
    ],
)
def test_eigencam_rejects_invalid_features(features):
    with pytest.raises(ValueError, match="EigenCAM requires"):
        _extract_eigencam(features)


def test_eigencam_falls_back_to_cpu_svd(monkeypatch):
    """Devices without an SVD kernel should decompose on CPU and move the vectors back."""
    matrix = torch.randn(2, 4, 3, 5, generator=torch.Generator().manual_seed(4)).flatten(2).transpose(1, 2)
    expected = activation._svd_right_vectors(matrix, cpu_fallback=False)
    svd, devices = torch.linalg.svd, []

    def no_svd_once(sample, **kwargs):
        devices.append(sample.device.type)
        if len(devices) == 1:
            raise NotImplementedError("SVD is unavailable on this device")
        return svd(sample, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", no_svd_once)
    actual = activation._svd_right_vectors(matrix, cpu_fallback=True)
    assert devices == ["cpu", "cpu"]
    torch.testing.assert_close(actual, expected)


def test_eigencam_propagates_unsupported_svd(monkeypatch):
    def unsupported_svd(*_args, **_kwargs):
        raise NotImplementedError("unsupported SVD")

    monkeypatch.setattr(torch.linalg, "svd", unsupported_svd)
    with pytest.raises(NotImplementedError, match="unsupported SVD"):
        _extract_eigencam(torch.rand(2, 4, 3, 5))
