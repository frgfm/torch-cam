# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest
import torch
from torch import nn
from torchvision.models import mobilenet_v2

from torchcam.methods import gradient


def _verify_cam(activation_map, output_size):
    # Simple verifications
    assert isinstance(activation_map, torch.Tensor)
    assert activation_map.shape == output_size
    assert not torch.any(torch.isnan(activation_map))


@pytest.mark.parametrize(
    "cam_name, target_layer, output_size",
    [
        ["GradCAM", 'features.18.0', (7, 7)],
        ["GradCAMpp", 'features.18.0', (7, 7)],
        ["SmoothGradCAMpp", lambda m: m.features[18][0], (7, 7)],
        ["SmoothGradCAMpp", 'features.18.0', (7, 7)],
        ["XGradCAM", 'features.18.0', (7, 7)],
        ["LayerCAM", 'features.18.0', (7, 7)],
    ],
)
def test_img_cams(cam_name, target_layer, output_size, mock_img_tensor):
    model = mobilenet_v2(pretrained=False).eval()

    target_layer = target_layer(model) if callable(target_layer) else target_layer
    # Hook the corresponding layer in the model
    extractor = gradient.__dict__[cam_name](model, target_layer)

    scores = model(mock_img_tensor)
    # Use the hooked data to compute activation map
    _verify_cam(extractor(scores[0].argmax().item(), scores)[0], output_size)

    # Inplace model
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(8, 10)
    )

    # Hook before the inplace ops
    extractor = gradient.__dict__[cam_name](model, '2')
    scores = model(mock_img_tensor)
    # Use the hooked data to compute activation map
    _verify_cam(extractor(scores[0].argmax().item(), scores)[0], (224, 224))


@pytest.mark.parametrize(
    "cam_name, target_layer, output_size",
    [
        ["GradCAM", '0.3', (8, 16, 16)],
        ["GradCAMpp", '0.3', (8, 16, 16)],
        ["SmoothGradCAMpp", '0.3', (8, 16, 16)],
        ["XGradCAM", '0.3', (8, 16, 16)],
        ["LayerCAM", '0.3', (8, 16, 16)],
    ],
)
def test_video_cams(cam_name, target_layer, output_size, mock_video_model, mock_video_tensor):
    model = mock_video_model.eval()
    # Hook the corresponding layer in the model
    extractor = gradient.__dict__[cam_name](model, target_layer)

    scores = model(mock_video_tensor)
    # Use the hooked data to compute activation map
    _verify_cam(extractor(scores[0].argmax().item(), scores)[0], output_size)


def test_smoothgradcampp_repr():
    model = mobilenet_v2(pretrained=False).eval()

    # Hook the corresponding layer in the model
    extractor = gradient.SmoothGradCAMpp(model, 'features.18.0')

    assert repr(extractor) == "SmoothGradCAMpp(target_layer=['features.18.0'], num_samples=4, std=0.3)"


def test_layercam_fuse_cams(mock_img_model):

    with pytest.raises(TypeError):
        gradient.LayerCAM.fuse_cams(torch.zeros((3, 32, 32)))

    with pytest.raises(ValueError):
        gradient.LayerCAM.fuse_cams([])

    cams = [torch.rand((32, 32)), torch.rand((16, 16))]

    # Single CAM
    assert torch.equal(cams[0], gradient.LayerCAM.fuse_cams(cams[:1]))

    # Fusion
    cam = gradient.LayerCAM.fuse_cams(cams)
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == cams[0].ndim
    assert cam.shape == (32, 32)

    # Specify target shape
    cam = gradient.LayerCAM.fuse_cams(cams, (16, 16))
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == cams[0].ndim
    assert cam.shape == (16, 16)
