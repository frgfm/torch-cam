import importlib
import json
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models.vision_transformer import VisionTransformer

from torchcam.explain import explain
from torchcam.methods import LeGrad
from torchcam.utils import overlay_mask

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


def test_explain_rejects_invalid_requests():
    model = _TinyCNN()
    input_tensor = torch.randn(1, 3, 8, 8)
    with pytest.raises(ValueError, match="model output must have shape"):
        explain_module._validate_logits(torch.zeros(3))
    with pytest.raises(ValueError, match="no CAMs"):
        explain_module._prepare_maps([])
    with pytest.raises(ValueError, match="CAMs must have shape"):
        explain_module._prepare_maps([torch.zeros(2, 2)])
    with pytest.raises(TypeError, match="Module"):
        explain(cast(Any, None), input_tensor)
    with pytest.raises(ValueError, match="evaluation mode"):
        explain(model, input_tensor, target_layer="features.1")

    model.eval()
    with pytest.raises(TypeError, match="Tensor"):
        explain(model, cast(Any, None), target_layer="features.1")
    with pytest.raises(ValueError, match="shape"):
        explain(model, torch.randn(2, 3, 8, 8), target_layer="features.1")
    with pytest.raises(ValueError, match="floating-point"):
        explain(model, torch.zeros(1, 3, 8, 8, dtype=torch.uint8), target_layer="features.1")
    with pytest.raises(TypeError, match="integer or None"):
        explain(model, input_tensor, expected_class_idx=cast(Any, "0"), target_layer="features.1")
    with pytest.raises(TypeError, match="extractor class"):
        explain(model, input_tensor, method=cast(Any, nn.Module), target_layer="features.1")
    with pytest.raises(TypeError, match="mapping"):
        explain(model, input_tensor, method_kwargs=cast(Any, []), target_layer="features.1")
    with pytest.raises(ValueError, match="output range"):
        explain(model, input_tensor, expected_class_idx=3, target_layer="features.1")
    with pytest.raises(ValueError, match="class_names"):
        explain(model, input_tensor, class_names=["a"], target_layer="features.1")
    with pytest.raises(TypeError, match="class name"):
        explain(model, input_tensor, class_names=cast(Any, ["a", "b", 3]), target_layer="features.1")
    with pytest.raises(ValueError, match="directly"):
        explain(model, input_tensor, method_kwargs={"target_layer": "features.1"})


def test_explain_rejects_invalid_outputs(monkeypatch):
    model = _TinyCNN().eval()
    input_tensor = torch.randn(1, 3, 8, 8)
    with pytest.raises(TypeError, match="tensor"):
        explain(_TupleModel().eval(), input_tensor, target_layer="features.1")
    with pytest.raises(ValueError, match="non-finite logits"):
        explain(_NonFiniteModel().eval(), input_tensor, target_layer="features.1")
    prepare_maps = explain_module._prepare_maps

    def inject_non_finite(maps):
        maps[0][0, 0, 0] = float("nan")
        return prepare_maps(maps)

    with monkeypatch.context() as patch:
        patch.setattr(explain_module, "_prepare_maps", inject_non_finite)
        with pytest.raises(ValueError, match="non-finite"):
            explain(model, input_tensor, target_layer="features.1")

    predicted = int(model(input_tensor).argmax().item())
    expected = (predicted + 1) % 3
    original_forward = model.forward
    forward_count = 0

    def change_output_shape(tensor):
        nonlocal forward_count
        forward_count += 1
        output = original_forward(tensor)
        return output if forward_count == 1 else output[:, :-1]

    with monkeypatch.context() as patch:
        patch.setattr(model, "forward", change_output_shape)
        with pytest.raises(ValueError, match="shape changed"):
            explain(model, input_tensor, expected_class_idx=expected, target_layer="features.1")

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
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(TypeError, match="PIL image"):
        result.save(tmp_path / "bad-image", cast(Any, None))
    bundle = result.save(tmp_path / "bundle", image)

    expected_files = {"manifest.json", "input.png"}
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
        assert torch.equal(torch.from_numpy(stored), maps[0])
        with Image.open(bundle / f"class-{class_idx}-layer-0-heatmap.png") as heatmap:
            assert heatmap.mode == "L"
        with Image.open(bundle / f"class-{class_idx}-layer-0-overlay.png") as overlay:
            assert overlay.size == image.size
            assert np.array_equal(overlay, overlay_mask(image, Image.fromarray(stored), alpha=0.5))

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["prediction"]["class_idx"] == result.predicted_class_idx
    assert manifest["expected"] == {"class_idx": 0, "class_name": "a"}
    assert manifest["image_size"] == [8, 8]
    assert manifest["input_image"] == "input.png"
    assert "context" not in manifest
    assert manifest["target_layers"] == ["features.1"]
    for class_data in manifest["classes"].values():
        assert all("/" not in path for path in class_data["artifacts"][0].values() if isinstance(path, str))

    with pytest.raises(FileExistsError):
        result.save(bundle, image)


def test_save_requires_model_view_and_preserves_caller_context(tmp_path):
    original = Image.fromarray(np.arange(12 * 20 * 3, dtype=np.uint8).reshape(12, 20, 3))
    model_image = original.crop((4, 2, 16, 10))
    model = _TinyCNN().eval()
    input_tensor = torch.from_numpy(np.array(model_image)).permute(2, 0, 1).float().unsqueeze(0) / 255
    result = explain(model, input_tensor, target_layer="features.1")
    output_dir = tmp_path / "evidence"
    context = {
        "checkpoint_id": "checkpoint-sha256:example",
        "preprocessing_id": "crop=(4,2,16,10); scale=1/255",
        "sample_id": "sample-7",
        "split_id": "validation-v1",
        "group_id": "camera-b",
    }

    with pytest.raises(ValueError, match="model-view"):
        result.save(output_dir, original, context=context)
    assert not output_dir.exists()

    for invalid in ([], {"sample_id": 7}, {7: "sample"}):
        with pytest.raises(TypeError, match="strings to strings"):
            result.save(output_dir, model_image, context=cast(Any, invalid))
        assert not output_dir.exists()

    bundle = result.save(output_dir, model_image, context=context)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["image_size"] == [12, 8]
    assert manifest["context"] == context
    assert manifest["input_shape"][-2:] == [8, 12]
    with Image.open(bundle / manifest["input_image"]) as stored_input:
        assert np.array_equal(stored_input, model_image)
    for class_idx, maps in result.cams.items():
        with Image.open(bundle / f"class-{class_idx}-layer-0-overlay.png") as overlay:
            assert np.array_equal(overlay, overlay_mask(model_image, Image.fromarray(maps[0].numpy()), alpha=0.5))


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


@pytest.mark.parametrize("mode", ["F", "I"])
def test_save_preserves_float_and_signed_integer_grayscale(tmp_path, mode):
    values = np.arange(96).reshape(8, 12)
    values = (values / 100 - 0.25).astype(np.float32) if mode == "F" else (values * 4096 - 10000).astype(np.int32)
    image = Image.fromarray(values)
    model = nn.Sequential(nn.Conv2d(1, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 2)).eval()
    input_tensor = torch.from_numpy(values.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    result = explain(model, input_tensor, target_layer="0")

    bundle = result.save(tmp_path / mode, image)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_image"] == "input.tiff"
    with Image.open(bundle / manifest["input_image"]) as stored_input:
        assert stored_input.mode == mode
        np.testing.assert_array_equal(stored_input, values)
