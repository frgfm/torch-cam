import pytest
from torchvision.models import get_model

from torchcam.methods import _utils


def test_locate_candidate_layer(mock_img_model, monkeypatch):
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
    mod[0][1].eval()

    assert _utils.locate_candidate_layer(mod) == "0.3"
    # Check that the model is switched back to its origin mode afterwards
    assert mod.training
    assert mod[0][0].training
    assert not mod[0][1].training

    modes = [module.training for module in mod.modules()]

    def failing_eval():
        for module in mod.modules():
            module.training = False
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "eval", failing_eval)
    with pytest.raises(RuntimeError):
        _utils.locate_candidate_layer(mod)
    assert [module.training for module in mod.modules()] == modes
