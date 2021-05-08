# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest
import torch
from torchvision.models import mobilenet_v2

from torchcam.cams import gradcam


def _verify_cam(activation_map, output_size):
    # Simple verifications
    assert isinstance(activation_map, torch.Tensor)
    assert activation_map.shape == output_size
    assert not torch.any(torch.isnan(activation_map))


@pytest.mark.parametrize(
    "cam_name, target_layer, output_size",
    [
        ["GradCAM", 'features.18.0', (7, 7)],
        ["GradCAMpp", 'features.18.0', (7, 7)],
        ["SmoothGradCAMpp", 'features.18.0', (7, 7)],
        ["XGradCAM", 'features.18.0', (7, 7)],
    ],
)
def test_img_cams(cam_name, target_layer, output_size, mock_img_tensor):
    model = mobilenet_v2(pretrained=False).eval()

    # Hook the corresponding layer in the model
    extractor = gradcam.__dict__[cam_name](model, target_layer)

    scores = model(mock_img_tensor)
    # Use the hooked data to compute activation map
    _verify_cam(extractor(scores[0].argmax().item(), scores), output_size)


@pytest.mark.parametrize(
    "cam_name, target_layer, output_size",
    [
        ["GradCAM", '0.3', (8, 16, 16)],
        ["GradCAMpp", '0.3', (8, 16, 16)],
        ["SmoothGradCAMpp", '0.3', (8, 16, 16)],
        ["XGradCAM", '0.3', (8, 16, 16)],
    ],
)
def test_video_cams(cam_name, target_layer, output_size, mock_video_model, mock_video_tensor):
    model = mock_video_model.eval()
    # Hook the corresponding layer in the model
    extractor = gradcam.__dict__[cam_name](model, target_layer)

    scores = model(mock_video_tensor)
    # Use the hooked data to compute activation map
    _verify_cam(extractor(scores[0].argmax().item(), scores), output_size)


def test_smoothgradcampp_repr():
    model = mobilenet_v2(pretrained=False).eval()

    # Hook the corresponding layer in the model
    extractor = gradcam.SmoothGradCAMpp(model, 'features.18.0')

    assert repr(extractor) == "SmoothGradCAMpp(target_layer='features.18.0', num_samples=4, std=0.3)"
