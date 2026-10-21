import pytest
import torch
from torch import nn
from torchvision.models import get_model

from torchcam.methods import GradCAM, _utils


def test_locate_candidate_layer_skips_non_tensor_outputs():
    class DictOutput(nn.Sequential):
        def forward(self, x):
            return {"logits": super().forward(x)}

    assert _utils.locate_candidate_layer(DictOutput(nn.Conv2d(3, 4, 3)), (3, 8, 8)) == "0"


def test_locate_candidate_layer_skips_region_and_shared_outputs():
    class Detector(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Conv2d(3, 4, 3)
            self.head = nn.Conv2d(4, 4, 1)
            self.rois = nn.Identity()

        def forward(self, x):
            x = self.backbone(x)
            self.head(self.head(x))
            return self.rois(x.expand(2, -1, -1, -1))

    assert _utils.locate_candidate_layer(Detector(), (3, 8, 8)) == "backbone"


@pytest.mark.parametrize(
    ("arch", "expected"),
    [("fasterrcnn_mobilenet_v3_large_320_fpn", "backbone.body.16"), ("lraspp_mobilenet_v3_large", "backbone.16")],
)
def test_locate_candidate_layer_stops_at_feature_dicts(arch, expected):
    # Per-level FPN outputs and single-proposal ROI heads come after the backbone's feature dict
    mod = get_model(arch, weights=None, weights_backbone=None).eval()
    assert _utils.locate_candidate_layer(mod) == expected


def test_locate_candidate_layer(mock_img_model, monkeypatch):
    # ResNet-18
    mod = get_model("resnet18", weights=None).eval()
    mod.requires_grad_(False)
    assert _utils.locate_candidate_layer(mod) == "layer4"

    # Mobilenet V3 Large
    mod = get_model("mobilenet_v3_large", weights=None).eval()
    mod.requires_grad_(False)
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


@pytest.mark.parametrize(
    ("dtype", "default_dtype"),
    [
        (torch.float16, torch.float32),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
    ],
)
def test_locate_candidate_layer_model_dtype(dtype, default_dtype):
    model = nn.Sequential(nn.Conv2d(3, 4, 3, dtype=dtype), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        assert _utils.locate_candidate_layer(model, (3, 8, 8)) == "1"
    finally:
        torch.set_default_dtype(original_dtype)


def test_gradcam_automatic_layer_double_model():
    model = (
        nn
        .Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(1), nn.Linear(4, 2))
        .double()
        .eval()
    )
    with GradCAM(model, input_shape=(3, 8, 8)) as extractor:
        scores = model(torch.ones((1, 3, 8, 8), dtype=torch.float64))
        (cam,) = extractor(0, scores)
        assert cam.dtype == torch.float64
    assert all(not module._forward_hooks for module in model.modules())
