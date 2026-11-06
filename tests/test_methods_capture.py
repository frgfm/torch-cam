import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

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


@pytest.mark.parametrize("method", [methods.CAM, methods.EigenCAM])
@pytest.mark.parametrize("checkpoint_root", [False, True])
def test_backward_recomputation_preserves_completed_capture(method, checkpoint_root, monkeypatch):
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2)).eval()
    input_tensor = torch.rand(1, 3, 4, 4, requires_grad=True)
    with method(model, "0") as extractor:
        expected = extractor(0, model(input_tensor))[0]
        if checkpoint_root:
            scores = checkpoint(model, input_tensor, use_reentrant=False)
            extractor.disable_hooks()
        else:
            forward = model.forward
            monkeypatch.setattr(model, "forward", lambda value: checkpoint(forward, value, use_reentrant=False))
            scores = model(input_tensor)
        scores.sum().backward()
        extractor.enable_hooks()
        assert torch.equal(extractor(0, scores)[0], expected)


@pytest.mark.parametrize("method", [methods.EigenCAM, methods.GradCAM])
def test_existing_forward_callback_can_extract(method, mock_img_model):
    model = mock_img_model.eval()
    captures = []

    def capture(_model, _inputs, scores):
        captures.extend(extractor(0, scores, retain_graph=True))

    with model.register_forward_hook(capture), method(model, "0.3") as extractor:
        scores = model(torch.rand(1, 3, 4, 4, requires_grad=True))
        assert torch.equal(captures[0], extractor(0, scores)[0])
