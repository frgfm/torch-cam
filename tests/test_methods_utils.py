from torchvision.models import resnet18

from torchcam.methods import _utils


def test_locate_candidate_layer(mock_img_model):
    # ResNet-18
    mod = resnet18().eval()
    assert _utils.locate_candidate_layer(mod) == "layer4"

    # Custom model
    mod = mock_img_model.train()

    assert _utils.locate_candidate_layer(mod) == "0.3"
    # Check that the model is switched back to its origin mode afterwards
    assert mod.training


def test_locate_linear_layer(mock_img_model):

    # ResNet-18
    mod = resnet18().eval()
    assert _utils.locate_linear_layer(mod) == "fc"

    # Custom model
    mod = mock_img_model
    assert _utils.locate_linear_layer(mod) == "2"
