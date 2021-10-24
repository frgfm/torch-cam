# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest
import torch
from torchvision.models import mobilenet_v2

from torchcam.cams import cam


def test_base_cam_constructor(mock_img_model):
    model = mobilenet_v2(pretrained=False).eval()
    # Check that multiple target layers is disabled for base CAM
    with pytest.raises(TypeError):
        _ = cam.CAM(model, ['classifier.1'])

    # FC layer checks
    with pytest.raises(TypeError):
        _ = cam.CAM(model, fc_layer=3)


def _verify_cam(activation_map, output_size):
    # Simple verifications
    assert isinstance(activation_map, torch.Tensor)
    assert activation_map.shape == output_size
    assert not torch.any(torch.isnan(activation_map))


@pytest.mark.parametrize(
    "cam_name, target_layer, fc_layer, num_samples, output_size",
    [
        ["CAM", None, None, None, (7, 7)],
        ["CAM", None, 'classifier.1', None, (7, 7)],
        ["CAM", None, lambda m: m.classifier[1], None, (7, 7)],
        ["ScoreCAM", 'features.16.conv.3', None, None, (7, 7)],
        ["ScoreCAM", lambda m: m.features[16].conv[3], None, None, (7, 7)],
        ["SSCAM", 'features.16.conv.3', None, 4, (7, 7)],
        ["ISCAM", 'features.16.conv.3', None, 4, (7, 7)],
    ],
)
def test_img_cams(cam_name, target_layer, fc_layer, num_samples, output_size, mock_img_tensor):
    model = mobilenet_v2(pretrained=False).eval()
    kwargs = {}
    # Speed up testing by reducing the number of samples
    if isinstance(num_samples, int):
        kwargs['num_samples'] = num_samples

    if fc_layer is not None:
        kwargs['fc_layer'] = fc_layer(model) if callable(fc_layer) else fc_layer

    target_layer = target_layer(model) if callable(target_layer) else target_layer
    # Hook the corresponding layer in the model
    extractor = cam.__dict__[cam_name](model, target_layer, **kwargs)

    with torch.no_grad():
        scores = model(mock_img_tensor)
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], output_size)


def test_cam_conv1x1(mock_fullyconv_model):
    extractor = cam.CAM(mock_fullyconv_model, fc_layer='1')
    with torch.no_grad():
        scores = mock_fullyconv_model(torch.rand((1, 3, 32, 32)))
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], (32, 32))


@pytest.mark.parametrize(
    "cam_name, target_layer, num_samples, output_size",
    [
        ["CAM", '0.3', None, (8, 16, 16)],
        ["ScoreCAM", '0.3', None, (8, 16, 16)],
        ["SSCAM", '0.3', 4, (8, 16, 16)],
        ["ISCAM", '0.3', 4, (8, 16, 16)],
    ],
)
def test_video_cams(cam_name, target_layer, num_samples, output_size, mock_video_model, mock_video_tensor):
    model = mock_video_model.eval()
    kwargs = {}
    # Speed up testing by reducing the number of samples
    if isinstance(num_samples, int):
        kwargs['num_samples'] = num_samples

    # Hook the corresponding layer in the model
    extractor = cam.__dict__[cam_name](model, target_layer, **kwargs)

    with torch.no_grad():
        scores = model(mock_video_tensor)
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], output_size)
