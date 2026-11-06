import pytest
import torch
from torch import nn

from torchcam import methods


class _ConditionalClassifier(nn.Sequential):
    def forward(self, input_tensor, *, skip=False, fail_at=None):
        for name, layer in self.named_children():
            if name == fail_at:
                raise RuntimeError("forward failed")
            if not (skip and name == "2"):
                input_tensor = layer(input_tensor)
        return input_tensor


@pytest.mark.parametrize("method", ["CAM", "EigenCAM", "GradCAM", "ScoreCAM", "SmoothGradCAMpp", "RefineCAM"])
@pytest.mark.parametrize("interruption", [{"skip": True}, {"fail_at": "0"}, {"fail_at": "2"}, {"fail_at": "3"}])
def test_captures_belong_to_one_complete_forward(method, interruption):
    model = _ConditionalClassifier(
        nn.Conv2d(3, 4, 1), nn.ReLU(), nn.Identity(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)
    ).eval()
    input_tensor = torch.rand(1, 3, 4, 4, requires_grad=True)
    options = {"num_samples": 2} if method == "SmoothGradCAMpp" else {}
    layers = "2" if method == "CAM" else ["1", "2"]
    with getattr(methods, method)(model, layers, **options) as extractor:
        scores = model(input_tensor)
        torch.manual_seed(0)
        expected = extractor(0, scores)
        if "fail_at" in interruption:
            with pytest.raises(RuntimeError, match="forward failed"):
                model(input_tensor * 2, **interruption)
        else:
            model(input_tensor * 2, **interruption)
        for extract in (extractor, extractor.compute_cams):
            with pytest.raises(AssertionError, match="forward"):
                extract(0, scores)

        # A complete forward recovers; an unrelated paused forward preserves its captures.
        scores = model(input_tensor)
        with extractor._hooks_off():
            model(input_tensor * 2, skip=True)
        torch.manual_seed(0)
        actual = extractor(0, scores)
        for before, after in zip(expected, actual, strict=True):
            assert torch.equal(before, after)
