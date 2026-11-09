from contextlib import suppress

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchcam.methods import EigenCAM, GradCAM, ScoreCAM


def test_skipped_target_rejects_previous_capture(monkeypatch):
    model = nn.Sequential(nn.Identity())
    image = torch.rand(1, 2, 3, 3)
    with EigenCAM(model, "0") as extractor:
        model(image)
        monkeypatch.setattr(model, "forward", lambda image: image)
        model(image * 2)
        with pytest.raises(AssertionError, match="not captured"):
            extractor()


@pytest.mark.parametrize("prehook", [True, False])
def test_failed_forward_rejects_partial_capture(prehook):
    model = nn.Sequential(nn.Identity())

    def fail(*_args):
        raise RuntimeError("forward failed")

    image = torch.rand(1, 2, 3, 3)
    with EigenCAM(model, "0") as extractor:
        model(image)
        register = model.register_forward_pre_hook if prehook else model.register_forward_hook
        with register(fail, prepend=prehook), pytest.raises(RuntimeError, match="forward failed"):
            model(image * 2)
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


def test_whole_model_checkpoint_preserves_capture(mock_img_model):
    model = mock_img_model.eval()
    with GradCAM(model, "0.3") as extractor:
        scores = checkpoint(model, torch.rand(1, 3, 4, 4, requires_grad=True), use_reentrant=False)
        expected = extractor(0, scores, retain_graph=True)[0]
        scores.sum().backward(retain_graph=True)
        assert torch.equal(extractor(0, scores)[0], expected)


def test_recursive_failure_is_not_completed_by_inner_call(monkeypatch):
    model = nn.Sequential(nn.Identity())

    def forward(image, *, nested=False, fail=False):
        if nested:
            return model[0](image)
        result = model(image, nested=True)
        if fail:
            raise RuntimeError("outer failed")
        return result

    monkeypatch.setattr(model, "forward", forward)
    image = torch.rand(1, 2, 3, 3)
    with EigenCAM(model, "0") as extractor:
        with pytest.raises(RuntimeError, match="outer failed"):
            model(image, fail=True)
        with pytest.raises(AssertionError, match="successful model forward"):
            extractor()
        model(image)
        assert extractor()[0].shape == (1, 3, 3)


def test_caught_recursive_failure_preserves_input(mock_img_model, monkeypatch):
    model = mock_img_model.eval()
    forward = model.forward

    def recover(image, *, nested=False):
        if nested:
            raise RuntimeError("inner failed")
        with suppress(RuntimeError):
            model(image, nested=True)
        return forward(image)

    monkeypatch.setattr(model, "forward", recover)
    with ScoreCAM(model, "0.3") as extractor:
        scores = model(torch.rand(1, 3, 4, 4, requires_grad=True))
        assert torch.isfinite(extractor(0, scores)[0]).all()
