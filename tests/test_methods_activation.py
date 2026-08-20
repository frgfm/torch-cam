import pytest
import torch
from torch import nn
from torchvision.models import get_model

from torchcam.methods import activation


def test_base_cam_constructor():
    model = get_model("mobilenet_v2", weights=None).eval()
    for p in model.parameters():
        p.requires_grad_(False)
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
    assert not torch.isnan(activation_map).any()


def _scorecam_inputs():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 5, 3, padding=1),
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
    (
        "cam_name",
        "target_layer",
        "fc_layer",
        "num_samples",
        "output_size",
        "batch_size",
    ),
    [
        ("CAM", None, None, None, (7, 7), 1),
        ("CAM", None, None, None, (7, 7), 2),
        ("CAM", None, "classifier.1", None, (7, 7), 1),
        ("CAM", None, lambda m: m.classifier[1], None, (7, 7), 1),
        ("ScoreCAM", "features.16.conv.3", None, None, (7, 7), 1),
        ("ScoreCAM", lambda m: m.features[16].conv[3], None, None, (7, 7), 1),
        ("SSCAM", "features.16.conv.3", None, 4, (7, 7), 1),
        ("ISCAM", "features.16.conv.3", None, 4, (7, 7), 1),
    ],
)
def test_img_cams(
    cam_name,
    target_layer,
    fc_layer,
    num_samples,
    output_size,
    batch_size,
    mock_img_tensor,
):
    model = get_model("mobilenet_v2", weights=None).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    kwargs = {}
    # Speed up testing by reducing the number of samples
    if isinstance(num_samples, int):
        kwargs["num_samples"] = num_samples

    if fc_layer is not None:
        kwargs["fc_layer"] = fc_layer(model) if callable(fc_layer) else fc_layer

    target_layer = target_layer(model) if callable(target_layer) else target_layer
    # Hook the corresponding layer in the model
    with activation.__dict__[cam_name](model, target_layer, **kwargs) as extractor, torch.no_grad():
        scores = model(mock_img_tensor.repeat((batch_size,) + (1,) * (mock_img_tensor.ndim - 1)))
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], (batch_size, *output_size))
        # Multiple class indices
        _verify_cam(extractor(list(range(batch_size)), scores)[0], (batch_size, *output_size))


def test_cam_conv1x1(mock_fullyconv_model):
    with activation.CAM(mock_fullyconv_model, fc_layer="1") as extractor, torch.no_grad():
        scores = mock_fullyconv_model(torch.rand((1, 3, 32, 32)))
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], (1, 32, 32))


@pytest.mark.parametrize("cam_name", ["ScoreCAM", "SSCAM", "ISCAM"])
@pytest.mark.parametrize("class_idx", [1, [1, 2]])
def test_scorecam_chunk_parity(cam_name, class_idx):
    results = []
    for batch_size in (3, 64):
        model, input_tensor = _scorecam_inputs()
        with activation.__dict__[cam_name](
            model,
            ["1", "3"],
            **_scorecam_kwargs(cam_name, batch_size),
        ) as extractor:
            scores = model(input_tensor)
            torch.manual_seed(1)
            results.append(extractor(class_idx, scores))

    assert [cam.shape for cam in results[0]] == [(2, 8, 8), (2, 8, 8)]
    assert all(cam.dtype == torch.float64 and cam.device.type == "cpu" for cam in results[0])
    for chunked, unchunked in zip(*results, strict=True):
        torch.testing.assert_close(chunked, unchunked)


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
