import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchcam.methods import EigenCAM, GradCAM


def test_skipped_target_rejects_previous_capture(monkeypatch):
    model = nn.Sequential(nn.Identity())
    image = torch.rand(1, 2, 3, 3)
    with EigenCAM(model, "0") as extractor:
        model(image)
        monkeypatch.setattr(model, "forward", lambda image: image)
        model(image * 2)
        with pytest.raises(AssertionError, match="not captured"):
            extractor()


def test_failed_forward_rejects_partial_capture():
    model = nn.Sequential(nn.Identity())

    def fail(*_args):
        raise RuntimeError("forward failed")

    with EigenCAM(model, "0") as extractor, model[0].register_forward_hook(fail):
        with pytest.raises(RuntimeError, match="forward failed"):
            model(torch.rand(1, 2, 3, 3))
        with pytest.raises(AssertionError, match="successful model forward"):
            extractor()


def test_checkpoint_backward_preserves_capture(mock_img_model, monkeypatch):
    model = mock_img_model.eval()
    forward = model.forward
    monkeypatch.setattr(model, "forward", lambda image: checkpoint(forward, image, use_reentrant=False))
    with EigenCAM(model, "0.3") as extractor:
        scores = model(torch.rand(1, 3, 4, 4, requires_grad=True))
        expected = extractor()[0]
        scores.sum().backward()
        assert torch.equal(extractor()[0], expected)


def test_existing_forward_callback_can_extract(mock_img_model):
    captures = []

    def capture(_model, _inputs, scores):
        captures.extend(extractor(0, scores))

    with mock_img_model.register_forward_hook(capture), GradCAM(mock_img_model, "0.3") as extractor:
        mock_img_model(torch.rand(1, 3, 4, 4, requires_grad=True))
        assert captures[0].shape == (1, 4, 4)
