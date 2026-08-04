from functools import partial

import pytest
import torch
from torch import nn
from torchvision.models import get_model

from torchcam.methods import activation, gradient
from torchcam.metrics import ClassificationMetric


def _verify_cam(activation_map, output_size):
    # Simple verifications
    assert isinstance(activation_map, torch.Tensor)
    assert activation_map.shape == output_size
    assert not torch.isnan(activation_map).any()


def _tiny_model():
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(4, 3),
    ).eval()


@pytest.mark.parametrize(
    ("cam_name", "target_layer", "output_size", "batch_size"),
    [
        ("GradCAM", "features.18.0", (7, 7), 1),
        ("GradCAMpp", "features.18.0", (7, 7), 1),
        ("SmoothGradCAMpp", lambda m: m.features[18][0], (7, 7), 1),
        ("SmoothGradCAMpp", "features.18.0", (7, 7), 1),
        ("XGradCAM", "features.18.0", (7, 7), 1),
        ("LayerCAM", "features.18.0", (7, 7), 1),
        ("LayerCAM", "features.18.0", (7, 7), 2),
    ],
)
def test_img_cams(cam_name, target_layer, output_size, batch_size, mock_img_tensor):
    model = get_model("mobilenet_v2", weights=None).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    target_layer = target_layer(model) if callable(target_layer) else target_layer
    # Hook the corresponding layer in the model
    with gradient.__dict__[cam_name](model, target_layer) as extractor:
        scores = model(mock_img_tensor.repeat((batch_size,) + (1,) * (mock_img_tensor.ndim - 1)))
        # Use the hooked data to compute activation map
        _verify_cam(
            extractor(scores[0].argmax().item(), scores, retain_graph=True)[0],
            (batch_size, *output_size),
        )
        # Multiple class indices
        _verify_cam(extractor(list(range(batch_size)), scores)[0], (batch_size, *output_size))

    # Inplace model
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(8, 10),
    )
    for p in model.parameters():
        p.requires_grad_(False)

    # Hook before the inplace ops
    with gradient.__dict__[cam_name](model, "2") as extractor:
        scores = model(mock_img_tensor)
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], (1, 224, 224))


@pytest.mark.parametrize(
    ("cam_name", "target_layer", "output_size"),
    [
        ("GradCAM", "0.3", (1, 8, 16, 16)),
        ("GradCAMpp", "0.3", (1, 8, 16, 16)),
        ("SmoothGradCAMpp", "0.3", (1, 8, 16, 16)),
        ("XGradCAM", "0.3", (1, 8, 16, 16)),
        ("LayerCAM", "0.3", (1, 8, 16, 16)),
    ],
)
def test_video_cams(cam_name, target_layer, output_size, mock_video_model, mock_video_tensor):
    model = mock_video_model.eval()
    # Hook the corresponding layer in the model
    with gradient.__dict__[cam_name](model, target_layer) as extractor:
        scores = model(mock_video_tensor)
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], output_size)


def test_smoothgradcampp_repr():
    model = get_model("mobilenet_v2", weights=None).eval()

    # Hook the corresponding layer in the model
    with gradient.SmoothGradCAMpp(model, "features.18.0") as extractor:
        assert repr(extractor) == "SmoothGradCAMpp(target_layer=['features.18.0'], num_samples=4, std=0.3)"


def test_layercam_fuse_cams():
    with pytest.raises(TypeError):
        gradient.LayerCAM.fuse_cams(torch.zeros((3, 32, 32)))

    with pytest.raises(ValueError):
        gradient.LayerCAM.fuse_cams([])

    cams = [torch.rand((1, 32, 32)), torch.rand((1, 16, 16))]

    # Single CAM
    assert torch.equal(cams[0], gradient.LayerCAM.fuse_cams(cams[:1]))

    # Fusion
    cam = gradient.LayerCAM.fuse_cams(cams)
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == cams[0].ndim
    assert cam.shape == (1, 32, 32)

    # Specify target shape
    cam = gradient.LayerCAM.fuse_cams(cams, (16, 16))
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == cams[0].ndim
    assert cam.shape == (1, 16, 16)


def test_refinecam_fuse_cams():
    cams = [
        torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
        torch.tensor([[[3.0, 2.0], [1.0, 0.0]]]),
    ]
    originals = [cam.clone() for cam in cams]

    assert torch.allclose(
        gradient.RefineCAM.fuse_cams(cams, normalized=False),
        torch.tensor([[[0.0, 2 / 9], [2 / 9, 0.0]]]),
    )
    assert torch.allclose(
        gradient.RefineCAM.fuse_cams(cams),
        torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
    )
    assert all(map(torch.equal, cams, originals))

    larger_cam = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    assert gradient.RefineCAM.fuse_cams([cams[0], larger_cam]).shape == (1, 4, 4)
    assert gradient.RefineCAM.fuse_cams([cams[0], larger_cam], (2, 2)).shape == (1, 2, 2)
    assert gradient.RefineCAM.fuse_cams(cams[:1]).shape == cams[0].shape

    volume = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    assert gradient.RefineCAM.fuse_cams([volume, volume.flip(-1)]).shape == volume.shape

    with pytest.raises(TypeError):
        gradient.RefineCAM.fuse_cams(torch.zeros((1, 2, 2)))
    with pytest.raises(ValueError):
        gradient.RefineCAM.fuse_cams([])


@pytest.mark.parametrize("base_method", [gradient.GradCAMpp, gradient.LayerCAM, activation.ScoreCAM])
def test_refinecam_base_methods(base_method):
    model = _tiny_model()
    input_tensor = torch.rand((2, 3, 8, 8))

    with gradient.RefineCAM(model, ["0", "2"], base_method=base_method) as extractor:
        scores = model(input_tensor)
        cams = extractor([0, 1], scores)
        assert len(cams) == 1
        _verify_cam(cams[0], (2, 8, 8))
        assert extractor.model is model
        assert extractor.target_names == ["0", "2"]
        assert repr(extractor) == f"RefineCAM(base_method={base_method.__name__}, target_layer=['0', '2'])"

    assert extractor.base_cam.hook_handles == []


def test_refinecam_metric_compatibility():
    model = _tiny_model()
    input_tensor = torch.rand((1, 3, 8, 8))

    with gradient.RefineCAM(model, ["0", "2"]) as extractor:
        metric = ClassificationMetric(extractor, partial(torch.softmax, dim=-1))
        metric.update(input_tensor)
        summary = metric.summary()

    assert set(summary) == {"avg_drop", "conf_increase"}


def test_refinecam_rejects_invalid_configuration():
    model = _tiny_model()

    with pytest.raises(ValueError, match="at least two target layers"):
        gradient.RefineCAM(model, ["0"])
    with pytest.raises(TypeError, match="CAM extractor class"):
        gradient.RefineCAM(model, ["0", "2"], base_method=nn.Module)


def test_gradcam_does_not_accumulate_hook_handles(mock_img_tensor):
    model = get_model("mobilenet_v2", weights=None).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with gradient.GradCAM(model, "features.18.0") as extractor:
        initial_handles = len(extractor.hook_handles)
        for _ in range(3):
            scores = model(mock_img_tensor)
            extractor(scores[0].argmax().item(), scores, retain_graph=True)
        assert len(extractor.hook_handles) == initial_handles
        assert len(extractor._grad_hook_handles) == len(extractor.target_names)


def test_smoothgradcampp_restores_input_hook_on_error(mock_img_tensor, monkeypatch):
    model = get_model("mobilenet_v2", weights=None).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with gradient.SmoothGradCAMpp(model, "features.18.0", num_samples=1) as extractor:
        scores = model(mock_img_tensor)
        monkeypatch.setattr(
            extractor,
            "_backprop",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError):
            extractor(scores[0].argmax().item(), scores)

        assert extractor._ihook_enabled
