from torchvision.models import get_model

from torchcam.methods import _utils


def test_locate_candidate_layer(mock_img_model):
    # ResNet-18
    mod = get_model("resnet18", weights=None).eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    assert _utils.locate_candidate_layer(mod) == "layer4"

    # Mobilenet V3 Large
    mod = get_model("mobilenet_v3_large", weights=None).eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    assert _utils.locate_candidate_layer(mod) == "features"

    # Custom model
    mod = mock_img_model.train()

    assert _utils.locate_candidate_layer(mod) == "0.3"
    # Check that the model is switched back to its origin mode afterwards
    assert mod.training
