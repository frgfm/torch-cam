import pytest
import torch
from torch import nn
from torchvision.models import mobilenet_v2

from torchcam.methods import gradient


def _verify_cam(activation_map, output_size):
    # Simple verifications
    assert isinstance(activation_map, torch.Tensor)
    assert activation_map.shape == output_size
    assert not torch.isnan(activation_map).any()


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
    model = mobilenet_v2(weights=None).eval()
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
    model = mobilenet_v2(weights=None).eval()

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
