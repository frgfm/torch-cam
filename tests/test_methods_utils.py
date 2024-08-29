from torchvision.models import mobilenet_v3_large, resnet18

from torchcam.methods import _utils


def test_locate_candidate_layer(mock_img_model):
    # ResNet-18
    mod = resnet18().eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    assert _utils.locate_candidate_layer(mod) == "layer4"

    # Mobilenet V3 Large
    mod = mobilenet_v3_large().eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    assert _utils.locate_candidate_layer(mod) == "features"

    # Custom model
    mod = mock_img_model.train()

    assert _utils.locate_candidate_layer(mod) == "0.3"
    # Check that the model is switched back to its origin mode afterwards
    assert mod.training


def test_locate_linear_layer(mock_img_model):
    # ResNet-18
    mod = resnet18().eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    assert _utils.locate_linear_layer(mod) == "fc"

    # Custom model
    mod = mock_img_model
    assert _utils.locate_linear_layer(mod) == "2"
