import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from torchcam.explain import explain

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / ".agents/skills/torchcam-debug-prediction/scripts/validate_bundle.py"
SPEC = importlib.util.spec_from_file_location("validate_bundle", VALIDATOR_PATH)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


@pytest.fixture
def bundle(tmp_path):
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
    image = Image.new("RGB", (16, 16))
    result = explain(
        model, torch.rand(1, 3, 16, 16), expected_class_idx=1, class_names=["vertical", "horizontal"], target_layer="0"
    )
    result.save(tmp_path / "bundle", image)
    return tmp_path / "bundle"


def test_validator_reads_real_artifact_list_and_blank_maps(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    entry = next(iter(manifest["classes"].values()))
    path = bundle / entry["artifacts"][0]["map"]
    cam = np.load(path, allow_pickle=False)
    np.save(path, np.zeros_like(cam), allow_pickle=False)
    result = VALIDATOR.validate_bundle(bundle, ["vertical", "horizontal"])
    assert result["schema_version"] == 1
    assert any(row["blank"] for row in result["maps"])


@pytest.mark.parametrize("fault", ["missing_manifest", "outside_path", "labels", "nonfinite", "overlay_size"])
def test_validator_rejects_incomplete_or_misleading_evidence(bundle, tmp_path, fault):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    artifact = next(iter(manifest["classes"].values()))["artifacts"][0]
    labels = ["vertical", "horizontal"]
    if fault == "missing_manifest":
        path.unlink()
    elif fault == "outside_path":
        np.save(tmp_path / "outside.npy", np.zeros((2, 2), dtype=np.float32))
        artifact["map"] = "../outside.npy"
        path.write_text(json.dumps(manifest))
    elif fault == "labels":
        labels.reverse()
    elif fault == "nonfinite":
        np.save(bundle / artifact["map"], np.full((2, 2), np.nan, dtype=np.float32))
    else:
        Image.new("RGB", (8, 8)).save(bundle / artifact["overlay"])
    with pytest.raises((ValueError, FileNotFoundError)):
        VALIDATOR.validate_bundle(bundle, labels)
