# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from torchvision.models import resnet18

from torchcam.cams import utils


def test_locate_candidate_layer(mock_img_model):
    # ResNet-18
    mod = resnet18().eval()
    assert utils.locate_candidate_layer(mod) == 'layer4'

    # Custom model
    mod = mock_img_model.train()

    assert utils.locate_candidate_layer(mod) == '0.3'
    # Check that the model is switched back to its origin mode afterwards
    assert mod.training


def test_locate_linear_layer(mock_img_model):

    # ResNet-18
    mod = resnet18().eval()
    assert utils.locate_linear_layer(mod) == 'fc'

    # Custom model
    mod = mock_img_model
    assert utils.locate_linear_layer(mod) == '2'
