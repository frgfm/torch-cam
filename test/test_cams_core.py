# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import pytest
import torch

from torchcam.cams import core


def test_cam_constructor(mock_img_model):
    model = mock_img_model.eval()
    # Check that wrong target_layer raises an error
    with pytest.raises(ValueError):
        _ = core._CAM(model, '3')


def test_cam_precheck(mock_img_model, mock_img_tensor):
    model = mock_img_model.eval()
    extractor = core._CAM(model, '0.3')
    with torch.no_grad():
        # Check missing forward raises Error
        with pytest.raises(AssertionError):
            extractor(0)
        # Check that a batch of 2 cannot be accepted
        _ = model(torch.cat((mock_img_tensor, mock_img_tensor)))
        with pytest.raises(ValueError):
            extractor(0)
        # Correct forward
        _ = model(mock_img_tensor)

        # Check incorrect class index
        with pytest.raises(ValueError):
            extractor(-1)

        # Check missing score
        if extractor._score_used:
            with pytest.raises(ValueError):
                extractor(0)


@pytest.mark.parametrize(
    "input_shape, spatial_dims",
    [
        [(8, 8), None],
        [(8, 8, 8), None],
        [(8, 8, 8), 2],
        [(8, 8, 8, 8), None],
        [(8, 8, 8, 8), 3],
    ],
)
def test_cam_normalize(input_shape, spatial_dims):
    input_tensor = torch.rand(input_shape)
    normalized_tensor = core._CAM._normalize(input_tensor, spatial_dims)
    # Shape check
    assert normalized_tensor.shape == input_shape
    # Value check
    assert not torch.any(torch.isnan(normalized_tensor))
    assert torch.all(normalized_tensor <= 1) and torch.all(normalized_tensor >= 0)


def test_cam_clear_hooks(mock_img_model):
    model = mock_img_model.eval()
    extractor = core._CAM(model, '0.3')

    assert len(extractor.hook_handles) == 1
    # Check that there is only one hook on the model
    assert extractor.hook_a is None
    with torch.no_grad():
        _ = model(torch.rand((1, 3, 32, 32)))
    assert extractor.hook_a is not None

    # Remove it
    extractor.clear_hooks()
    assert len(extractor.hook_handles) == 0
    # Check that there is no hook anymore
    extractor.hook_a = None
    with torch.no_grad():
        _ = model(torch.rand((1, 3, 32, 32)))
    assert extractor.hook_a is None


def test_cam_repr(mock_img_model):
    model = mock_img_model.eval()
    extractor = core._CAM(model, '0.3')

    assert repr(extractor) == "_CAM(target_layer='0.3')"
