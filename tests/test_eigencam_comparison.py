from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from PIL import Image
from torch import nn

from scripts import eigencam_comparison as comparison


@pytest.mark.parametrize("method", comparison.METHODS)
@pytest.mark.parametrize("scope", ["extractor", "end-to-end"])
def test_comparison_times_the_correct_scope_and_autograd_mode(method, scope, monkeypatch):
    events = []

    def forward(_tensor):
        events.append(("forward", torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return torch.tensor([[1.0]])

    def extract(*args):
        events.append(("extract", torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        assert len(args) == (0 if method == "EigenCAM" else 2)
        return [torch.ones(1, 2, 2)]

    ticks = iter((1.0, 1.25))
    monkeypatch.setattr(comparison, "_synchronize", lambda _device: events.append("sync"))
    monkeypatch.setattr(comparison.time, "perf_counter", lambda: (events.append("clock"), next(ticks))[1])
    elapsed = comparison._time_sample(
        Mock(side_effect=forward), torch.ones(1, 3, 2, 2), extract, method, 0, torch.device("cpu"), scope
    )
    gradient = method in {"GradCAM", "LayerCAM"}
    forward_event = ("forward", gradient, not gradient)
    extraction_event = ("extract", gradient, not gradient)
    expected = (
        [forward_event, "sync", "clock", extraction_event, "sync", "clock"]
        if scope == "extractor"
        else ["sync", "clock", forward_event, extraction_event, "sync", "clock"]
    )
    assert events == expected
    assert elapsed == 250


def test_comparison_rejects_modified_cached_image(tmp_path):
    (tmp_path / "chelsea.png").write_bytes(b"modified image")
    with pytest.raises(ValueError, match="Unexpected image checksum: chelsea"):
        comparison._load_images(tmp_path)


def test_comparison_headless_example_preserves_features_and_exposes_baseline_constraints(tmp_path, monkeypatch):
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer4 = nn.Sequential(nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d((4, 4)))
            self.fc = nn.Linear(4, 2)

        def forward(self, tensor):
            return self.fc(self.layer4(tensor).mean(dim=(2, 3)))

    model = Encoder().eval().requires_grad_(False)
    image = Image.new("RGB", (224, 224), color="orange")
    image.paste(Image.new("RGB", (112, 224), color="blue"), (0, 0))
    original_cam = comparison._extract(model, comparison._input(image, torch.device("cpu")), "EigenCAM", 0)
    assert original_cam.max() > 0
    monkeypatch.setattr(comparison, "_save_grid", lambda *_args: None)
    report = comparison._headless_evidence(model, image, original_cam, SimpleNamespace(device="cpu", output=tmp_path))
    assert report["embedding_shape"] == [1, 4]
    assert report["map_max_abs_difference"] == 0
    assert "unable to resolve" in report["baseline_errors"]["CAM"]
    assert "LayerCAM_under_inference_mode" in report["baseline_errors"]
