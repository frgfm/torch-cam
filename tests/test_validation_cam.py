import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights
from torchvision.transforms import Compose, Resize, ToTensor

from scripts import validation_cam
from torchcam.methods import GradCAM


def test_selection_is_ranked_bounded_deduplicated_and_repeatable():
    logits = torch.tensor([
        [8.0, 0, 0],
        [3.0, 0, 0],
        [1.01, 1, 0],
        [0, 1.1, 1],
        [0, 0, 6],
        [6, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ])
    model = nn.Embedding.from_pretrained(logits).eval()
    dataset = TensorDataset(torch.arange(8), torch.tensor([1, 2, 0, 1, 2, 0, 1, 0]))
    python_rng = random.getstate()
    first, timing = validation_cam.validate(model, DataLoader(dataset, batch_size=3), per_bucket=2)
    second, _ = validation_cam.validate(model, DataLoader(dataset, batch_size=5), per_bucket=2)

    assert first == second
    assert random.getstate() == python_rng
    assert timing["accuracy"] == pytest.approx(0.75)
    assert timing["validation_seconds"] >= timing["selection_seconds"] >= 0
    assert [row["index"] for row in first if "wrong_high" in row["buckets"]] == [0, 1]
    assert [row["index"] for row in first if "correct_low" in row["buckets"]] == [7, 2]
    control_labels = [row["label"] for row in first if "control" in row["buckets"]]
    assert len(control_labels) == len(set(control_labels)) == 2
    assert len(first) == len({row["index"] for row in first}) <= 6
    assert all(row["prediction"] != row["runner_up"] for row in first)
    with torch.no_grad():
        model.weight.neg_()
    changed, _ = validation_cam.validate(model, DataLoader(dataset, batch_size=3), per_bucket=2)
    assert sorted(row["index"] for row in changed if "control" in row["buckets"]) == sorted(
        row["index"] for row in first if "control" in row["buckets"]
    )


def test_validation_rejects_unusable_inputs():
    model = nn.Embedding(2, 3).eval()
    dataset = TensorDataset(torch.arange(2), torch.tensor([0, 1]))
    with pytest.raises(TypeError, match="sequential"):
        validation_cam.validate(model, DataLoader(dataset, shuffle=True))
    with pytest.raises(ValueError, match=r"model\.eval"):
        validation_cam.validate(model.train(), DataLoader(dataset))
    model.eval()
    for labels in (torch.tensor([-1, 0]), torch.tensor([0, 3])):
        with pytest.raises(ValueError, match="output indices"):
            validation_cam.validate(model, DataLoader(TensorDataset(torch.arange(2), labels)))
    with pytest.raises(ValueError, match="empty"):
        validation_cam.validate(model, DataLoader(TensorDataset(torch.empty(0, dtype=torch.long), torch.empty(0))))
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        validation_cam.validate(model, DataLoader(dataset))


class _ColorClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 2, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.features.weight.copy_(torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]).reshape(2, 3, 1, 1))
            self.head.weight.copy_(torch.tensor([[1.0, -1], [-1, 1], [0, 0]]))
        self.batch_sizes = []

    def forward(self, inputs):
        self.batch_sizes.append(len(inputs))
        return self.head(self.pool(self.features(inputs)).flatten(1))


@pytest.fixture
def evidence_inputs(tmp_path):
    root = tmp_path / "images"
    for label in (0, 1):
        folder = root / str(label)
        folder.mkdir(parents=True)
        for idx, color in enumerate(("red", "green", "blue")):
            image = Image.new("RGB", (12, 8), color)
            ImageDraw.Draw(image).rectangle((0, 0, 3, 7), fill="black")
            image.save(folder / f"{idx}.png")
    dataset = ImageFolder(root, transform=Compose([Resize((8, 8)), ToTensor()]))
    model = _ColorClassifier().eval()
    cases, validation = validation_cam.validate(model, DataLoader(dataset, batch_size=4), per_bucket=3)
    return model, dataset, cases, validation


def _save(model, dataset, cases, output, **kwargs):
    return validation_cam.save_evidence(
        model,
        dataset,
        cases,
        output,
        class_names=["red", "green", "other"],
        model_view=lambda image: image.resize((8, 8)),
        target_layer="features",
        batch_size=2,
        **kwargs,
    )


def test_batched_evidence_matches_singletons_and_preserves_training_state(evidence_inputs, tmp_path):
    model, dataset, cases, validation = evidence_inputs
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    gradients = [parameter.grad.clone() for parameter in model.parameters()]
    model.batch_sizes.clear()
    output = tmp_path / "epoch-001"
    with torch.no_grad():
        report = _save(model, dataset, cases, output, context={"checkpoint_id": "test"}, validation=validation)
    assert model.batch_sizes == [min(2, len(cases) - start) for start in range(0, len(cases), 2)]
    assert not model.features._forward_hooks
    assert all(not module.training for module in model.modules())
    for parameter, gradient in zip(model.parameters(), gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient)
    assert json.loads((output / "report.json").read_text()) == report
    assert report["cam_seconds"] > 0
    assert report["artifact_seconds"] > 0
    assert report["validation"] == validation
    with Image.open(output / report["contact_sheet"]) as sheet:
        assert sheet.size == (672, 290 * len(cases))

    for record in report["samples"]:
        bundle = output / record["bundle"]
        manifest = json.loads((bundle / "manifest.json").read_text())
        assert manifest["expected"]["class_idx"] == record["label"]
        assert manifest["context"]["sample_id"] == record["sample_id"]
        assert record["comparison"] == (
            record["label"] if record["prediction"] != record["label"] else record["runner_up"]
        )
        with Image.open(bundle / manifest["input_image"]) as image:
            expected_image = dataset.loader(dataset.samples[record["index"]][0]).resize((8, 8))
            np.testing.assert_array_equal(np.asarray(image), np.asarray(expected_image))
        for class_idx, saved_class in manifest["classes"].items():
            artifact = saved_class["artifacts"][0]
            saved_cam = np.load(bundle / artifact["map"], allow_pickle=False)
            with GradCAM(model, "features") as extractor:
                inputs = dataset[record["index"]][0].unsqueeze(0).requires_grad_(True)
                expected_cam = extractor(int(class_idx), model(inputs))[0][0]
            torch.testing.assert_close(torch.from_numpy(saved_cam), expected_cam)
            assert saved_cam.dtype == np.float32
            assert (bundle / artifact["overlay"]).is_file()
    assert any(record["blank_classes"] for record in report["samples"])


def test_failure_cleans_partial_evidence_and_never_overwrites(evidence_inputs, tmp_path):
    model, dataset, cases, _ = evidence_inputs
    output = tmp_path / "evidence"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        _save(model, dataset, cases, output)
    assert sentinel.read_text() == "existing evidence"
    output = tmp_path / "failed"
    with torch.no_grad():
        model.head.weight.zero_()
    with pytest.raises(ValueError, match="inference changed"):
        _save(model, dataset, cases, output)
    assert not output.exists()
    assert not model.features._forward_hooks
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        _save(model, dataset, cases, output)


def test_artifact_failure_removes_all_partial_bundles(evidence_inputs, tmp_path, monkeypatch):
    model, dataset, cases, _ = evidence_inputs

    def fail_contact_sheet(directory, _records):
        assert list(directory.glob("sample-*/manifest.json"))
        raise OSError("render failed")

    monkeypatch.setattr(validation_cam, "_contact_sheet", fail_contact_sheet)
    output = tmp_path / "failed-render"
    with pytest.raises(OSError, match="render failed"):
        _save(model, dataset, cases, output)
    assert not output.exists()
    assert not model.features._forward_hooks


def test_cli_uses_numeric_class_indices_and_writes_one_complete_report(evidence_inputs, tmp_path, monkeypatch, capsys):
    model, dataset, _, _ = evidence_inputs
    (dataset.root / "1").rename(dataset.root / "2")
    weights = SimpleNamespace(meta={"categories": ["red", "green", "other"]}, url="test-checkpoint")
    weights.transforms = lambda: ResNet18_Weights.DEFAULT.transforms(crop_size=8, resize_size=8)
    monkeypatch.setattr(validation_cam, "get_model_weights", lambda _: SimpleNamespace(DEFAULT=weights))
    monkeypatch.setattr(validation_cam, "get_model", lambda *_args, **_kwargs: model)
    output = tmp_path / "cli"
    args = validation_cam._build_parser().parse_args([
        str(dataset.root),
        str(output),
        "--target",
        "features",
        "--batch-size",
        "2",
    ])
    validation_cam.main(args)
    report = json.loads((output / "report.json").read_text())
    assert {row["label"] for row in report["samples"]} == {0, 2}
    assert report["context"]["checkpoint_id"] == "test-checkpoint"
    assert report["validation"]["samples"] == 6
    assert "including selection" in capsys.readouterr().out
