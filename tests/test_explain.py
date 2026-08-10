import importlib
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from torchcam.explain import explain
from torchcam.methods import LeGrad

explain_module = importlib.import_module("torchcam.explain")


class _TinyCNN(nn.Module):
    def __init__(self, *, count_forwards=False):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 3)
        self.count_forwards = count_forwards
        self.forward_count = 0

    def forward(self, input_tensor):
        if self.count_forwards:
            self.forward_count += 1
        return self.classifier(self.pool(self.features(input_tensor)).flatten(1))


class _TupleModel(_TinyCNN):
    def forward(self, input_tensor):
        output = super().forward(input_tensor)
        return output, output


class _NonFiniteModel(_TinyCNN):
    def forward(self, input_tensor):
        output = super().forward(input_tensor)
        return output * torch.tensor(float("nan"), device=output.device)


def _tiny_vit():
    return VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=4,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=3,
        dropout=0,
        attention_dropout=0,
    )


def test_explain_predicted_class_with_automatic_cnn_target():
    model = _TinyCNN().eval()
    result = explain(model, torch.randn(1, 3, 8, 8), class_names=["a", "b", "c"])

    assert result.logits.shape == (1, 3)
    assert result.logits.device.type == "cpu"
    assert result.logits.dtype == torch.float32
    assert not result.logits.requires_grad
    assert set(result.cams) == {result.predicted_class_idx}
    assert result.cams[result.predicted_class_idx][0].shape == (8, 8)
    assert result.cams[result.predicted_class_idx][0].dtype == torch.float32
    assert result.target_layers == ("features",)
    assert result.method == "GradCAM"
    assert result.class_names == ("a", "b", "c")


def test_explain_runs_fresh_forward_for_distinct_expected_class():
    model = _TinyCNN(count_forwards=True).eval()
    input_tensor = torch.randn(1, 3, 8, 8)
    predicted = model(input_tensor).argmax().item()
    expected = (predicted + 1) % 3
    model.forward_count = 0

    result = explain(model, input_tensor, expected_class_idx=expected, target_layer="features.1")

    assert model.forward_count == 2
    assert set(result.cams) == {predicted, expected}


def test_explain_reuses_map_when_expected_class_is_predicted():
    model = _TinyCNN(count_forwards=True).eval()
    input_tensor = torch.randn(1, 3, 8, 8)
    predicted = model(input_tensor).argmax().item()
    model.forward_count = 0

    result = explain(model, input_tensor, expected_class_idx=predicted, target_layer="features.1")

    assert model.forward_count == 1
    assert set(result.cams) == {predicted}


def test_explain_supports_torchvision_vit_with_legrad(tmp_path):
    model = _tiny_vit().eval()
    result = explain(
        model,
        torch.randn(1, 3, 32, 32),
        method=LeGrad,
        target_layer=list(model.encoder.layers)[-2:],
    )

    assert result.method == "LeGrad"
    assert result.target_layers == ("encoder.layers.encoder_layer_0", "encoder.layers.encoder_layer_1")
    assert result.cams[result.predicted_class_idx][0].shape == (4, 4)
    assert torch.isfinite(result.cams[result.predicted_class_idx][0]).all()
    bundle = result.save(tmp_path / "vit", Image.new("RGB", (32, 32)))
    artifact = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))["classes"][
        str(result.predicted_class_idx)
    ]["artifacts"][0]
    assert artifact["target_layers"] == list(result.target_layers)


def test_explain_rejects_invalid_inputs_and_outputs(monkeypatch):
    model = _TinyCNN()
    input_tensor = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="no CAMs"):
        explain_module._prepare_maps([])
    with pytest.raises(ValueError, match="evaluation mode"):
        explain(model, input_tensor, target_layer="features.1")

    model.eval()
    with pytest.raises(ValueError, match="shape"):
        explain(model, torch.randn(2, 3, 8, 8), target_layer="features.1")
    with pytest.raises(ValueError, match="floating-point"):
        explain(model, torch.zeros(1, 3, 8, 8, dtype=torch.uint8), target_layer="features.1")
    with pytest.raises(ValueError, match="output range"):
        explain(model, input_tensor, expected_class_idx=3, target_layer="features.1")
    with pytest.raises(ValueError, match="class_names"):
        explain(model, input_tensor, class_names=["a"], target_layer="features.1")
    with pytest.raises(TypeError, match="tensor"):
        explain(_TupleModel().eval(), input_tensor, target_layer="features.1")
    with pytest.raises(ValueError, match="non-finite logits"):
        explain(_NonFiniteModel().eval(), input_tensor, target_layer="features.1")
    prepare_maps = explain_module._prepare_maps

    def inject_non_finite(maps):
        maps[0][0, 0, 0] = float("nan")
        return prepare_maps(maps)

    monkeypatch.setattr(explain_module, "_prepare_maps", inject_non_finite)
    with pytest.raises(ValueError, match="non-finite"):
        explain(model, input_tensor, target_layer="features.1")

    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        explain(model, input_tensor, target_layer="features.1")


def test_explain_cleans_hooks_and_preserves_model_state():
    model = _TinyCNN().eval()
    input_tensor = torch.randn(1, 3, 8, 8)
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    gradient = parameter.grad
    state = {name: value.clone() for name, value in model.state_dict().items()}
    flags = [parameter.requires_grad for parameter in model.parameters()]
    modes = [module.training for module in model.modules()]
    hooks = (len(model.features[1]._forward_hooks), len(model.features[1]._forward_pre_hooks))

    explain(model, input_tensor, target_layer="features.1")

    assert hooks == (len(model.features[1]._forward_hooks), len(model.features[1]._forward_pre_hooks))
    assert parameter.grad is gradient
    assert flags == [parameter.requires_grad for parameter in model.parameters()]
    assert modes == [module.training for module in model.modules()]
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in state.items())
    assert input_tensor.grad is None


def test_save_writes_complete_deterministic_bundle(tmp_path):
    model = _TinyCNN().eval()
    result = explain(
        model,
        torch.randn(1, 3, 8, 8),
        expected_class_idx=0,
        class_names=["a", "b", "c"],
        target_layer="features.1",
    )
    image = Image.fromarray(np.zeros((13, 19, 3), dtype=np.uint8))
    bundle = result.save(tmp_path / "bundle", image)

    expected_files = {"manifest.json"}
    for class_idx in result.cams:
        expected_files.update({
            f"class-{class_idx}-layer-0.npy",
            f"class-{class_idx}-layer-0-heatmap.png",
            f"class-{class_idx}-layer-0-overlay.png",
        })
    assert {path.name for path in bundle.iterdir()} == expected_files
    for class_idx, maps in result.cams.items():
        stored = np.load(bundle / f"class-{class_idx}-layer-0.npy", allow_pickle=False)
        assert stored.dtype == np.float32
        assert np.array_equal(stored, maps[0].numpy())
        with Image.open(bundle / f"class-{class_idx}-layer-0-heatmap.png") as heatmap:
            assert heatmap.mode == "L"
        with Image.open(bundle / f"class-{class_idx}-layer-0-overlay.png") as overlay:
            assert overlay.size == image.size

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["prediction"]["class_idx"] == result.predicted_class_idx
    assert manifest["expected"] == {"class_idx": 0, "class_name": "a"}
    assert manifest["image_size"] == [19, 13]
    assert manifest["target_layers"] == ["features.1"]
    for class_data in manifest["classes"].values():
        assert all("/" not in path for path in class_data["artifacts"][0].values() if isinstance(path, str))

    with pytest.raises(FileExistsError):
        result.save(bundle, image)


def test_save_writes_manifest_only_after_artifacts(tmp_path, monkeypatch):
    model = _TinyCNN().eval()
    result = explain(model, torch.randn(1, 3, 8, 8), target_layer="features.1")
    output_dir = tmp_path / "incomplete"
    image = Image.new("RGB", (8, 8))

    def fail_overlay(*_args, **_kwargs):
        raise RuntimeError("render failed")

    with monkeypatch.context() as patch:
        patch.setattr(explain_module, "overlay_mask", fail_overlay)
        with pytest.raises(RuntimeError, match="render failed"):
            result.save(output_dir, image)

    assert not output_dir.exists()
    assert result.save(output_dir, image) == output_dir
