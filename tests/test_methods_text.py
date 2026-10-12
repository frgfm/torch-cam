from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from torchcam.methods import TokenIG


class TextClassifier(nn.Module):
    """Separable polynomial classifier with analytically known token contributions."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2, dtype=torch.float64)
        self.dropout = nn.Dropout()
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor([[0, 0], [2, 1], [-1, 3], [4, -2]], dtype=torch.float64))
        self.calls = 0

    def forward(self, input_ids, attention_mask):
        """Compute the classifier output.

        Returns:
            two independent class scores per sample
        """
        self.calls += 1
        states = self.dropout(self.embedding(input_ids)) * attention_mask.unsqueeze(-1)
        x, y = states.unbind(-1)
        return torch.stack(((x.square() + 2 * y).sum(-1), (-x + y.pow(3)).sum(-1)), -1)


def _inputs():
    return {"input_ids": torch.tensor([[1, 2, 3], [3, 1, 0]]), "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]])}


def test_token_ig_class_specific_analytic_contributions():
    model = TextClassifier()
    inputs = _inputs()
    with torch.no_grad():
        result = TokenIG(model, "embedding")(inputs, steps=4, max_steps=8)
    x, y = model.embedding(inputs["input_ids"]).detach().unbind(-1)
    expected = torch.stack((x.square() + 2 * y, -x + y.pow(3)), 1)
    torch.testing.assert_close(result.attributions, expected)
    torch.testing.assert_close(result.scores, expected.sum(-1))
    torch.testing.assert_close(result.baseline_scores, torch.zeros_like(result.scores))
    torch.testing.assert_close(result.completeness_error, torch.zeros_like(result.completeness_error))
    assert result.class_indices == (0, 1)
    assert result.steps == 8
    assert model.calls == 2 + 4 + 8  # One forward per node, shared by all classes.
    assert not result.attributions.requires_grad
    assert (result.attributions < 0).any()
    assert not torch.equal(result.attributions[:, 0].argsort(), result.attributions[:, 1].argsort())


def test_token_ig_reference_mask_class_order_and_model_preservation():  # noqa: PLR0914
    model = TextClassifier()
    model.dropout.eval()  # Mixed submodule modes must also survive attribution.
    modes = [module.training for module in model.modules()]
    model.embedding.weight.grad = torch.ones_like(model.embedding.weight)
    original_grad = model.embedding.weight.grad.clone()
    original_weights = model.embedding.weight.detach().clone()
    inputs = _inputs()
    baseline = torch.ones(2, 3, 2, dtype=torch.float64)
    mask = torch.tensor([[False, True, True], [True, False, False]])
    prediction = model(**inputs).detach()
    result = TokenIG(model, model.embedding)(inputs, [1, 0], baseline=baseline, token_mask=mask, steps=4, max_steps=8)
    states = model.embedding(inputs["input_ids"]).detach()
    reference = torch.where(mask.unsqueeze(-1), baseline, states) * inputs["attention_mask"].unsqueeze(-1)
    x, y = states.unbind(-1)
    bx, by = reference.unbind(-1)
    expected = torch.stack((-x + y.pow(3) + bx - by.pow(3), x.square() + 2 * y - bx.square() - 2 * by), 1)
    torch.testing.assert_close(result.attributions, expected)
    torch.testing.assert_close(result.attributions.sum(-1), (result.scores - result.baseline_scores)[:, [1, 0]])
    torch.testing.assert_close(model(**inputs), prediction)
    torch.testing.assert_close(model.embedding.weight, original_weights)
    torch.testing.assert_close(model.embedding.weight.grad, original_grad)
    assert model.embedding.weight.requires_grad
    assert [module.training for module in model.modules()] == modes
    assert not model.embedding._forward_hooks


def test_token_ig_structured_outputs_and_frozen_model():
    model = TextClassifier().requires_grad_(False)
    forward = model.forward
    model.forward = lambda **kwargs: SimpleNamespace(logits=forward(**kwargs))
    result = TokenIG(model, model.embedding, lambda output: output.logits)(_inputs(), 1, steps=4, max_steps=8)
    assert result.attributions.shape == (2, 1, 3)
    assert result.class_indices == (1,)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_token_ig_interactions_keep_context_fixed():
    model = TextClassifier()
    model.forward = lambda input_ids, attention_mask: (model.embedding(input_ids)[..., 0] * attention_mask).prod(
        -1, keepdim=True
    )
    inputs = {"input_ids": torch.tensor([[1, 3, 2]]), "attention_mask": torch.ones(1, 3)}
    result = TokenIG(model, model.embedding)(
        inputs, token_mask=torch.tensor([[True, True, False]]), steps=4, max_steps=8
    )
    # The third token is fixed context: IG splits a two-token product equally.
    torch.testing.assert_close(result.attributions, torch.tensor([[[-4, -4, 0]]], dtype=torch.float64))


@pytest.mark.parametrize("checkpointed", [False, True])
def test_token_ig_checkpointed_cancellation(checkpointed):
    model = TextClassifier()
    with torch.no_grad():
        model.embedding.weight[1:, 0].copy_(torch.tensor([1, -1, 0], dtype=torch.float64))
    model.embedding.weight.grad = torch.ones_like(model.embedding.weight)

    def forward(input_ids, attention_mask):
        def cubic(ids):
            return model.embedding(ids)[..., 0].pow(3)

        states = checkpoint(cubic, input_ids, use_reentrant=False) if checkpointed else cubic(input_ids)
        score = (states * attention_mask).sum(-1)
        return torch.stack((score, 2 * score), -1)

    model.forward = forward
    inputs = {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2)}
    result = TokenIG(model, model.embedding)(inputs, steps=4, max_steps=8)
    # Incorrect [3, -3] maps also pass completeness: assert the individual analytical contributions.
    torch.testing.assert_close(result.attributions, torch.tensor([[[1, -1], [2, -2]]], dtype=torch.float64))
    torch.testing.assert_close(result.completeness_error, torch.zeros(1, 2, dtype=torch.float64))
    torch.testing.assert_close(model.embedding.weight.grad, torch.ones_like(model.embedding.weight))
    assert model.training
    assert model.dropout.training
    assert not model.embedding._forward_hooks


def test_token_ig_refuses_unconverged_maps_and_cleans_up():
    model = TextClassifier()
    forward = model.forward
    model.forward = lambda **kwargs: forward(**kwargs).sin()
    with pytest.raises(RuntimeError, match="did not converge"):
        TokenIG(model, model.embedding)(_inputs(), steps=2, max_steps=4, tolerance=1e-12)
    assert model.training
    assert model.dropout.training
    assert not model.embedding._forward_hooks


@pytest.mark.parametrize("failure", ["no_grad", "disconnected", "reused", "exception", "shape", "nonfinite"])
def test_token_ig_forward_failures_restore_model(failure):
    model = TextClassifier()
    forward = model.forward
    if failure == "no_grad":
        model.forward = torch.no_grad()(forward)
    elif failure == "disconnected":
        model.forward = lambda **kwargs: forward(**kwargs).detach() + nn.Parameter(torch.ones(2, 2))
    elif failure == "reused":
        model.forward = lambda **kwargs: forward(**kwargs) + forward(**kwargs)
    elif failure == "shape":
        model.forward = lambda **kwargs: forward(**kwargs)[0]
    elif failure == "nonfinite":
        model.forward = lambda **kwargs: forward(**kwargs) * float("nan")
    else:

        def broken(**kwargs):
            forward(**kwargs)
            raise RuntimeError("failed model forward")

        model.forward = broken
    with pytest.raises((ValueError, RuntimeError)):
        TokenIG(model, model.embedding)(_inputs(), steps=2, max_steps=4)
    assert model.training
    assert not model.embedding._forward_hooks
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 0},
        {"steps": True},
        {"max_steps": 32},
        {"tolerance": float("nan")},
        {"stability_tolerance": -1},
        {"class_idx": True},
        {"class_idx": [1, 1]},
        {"class_idx": []},
        {"class_idx": 2},
        {"token_mask": torch.ones(2, 3)},
        {"token_mask": torch.ones(3, dtype=torch.bool)},
        {"baseline": torch.ones(2, 3, 2, dtype=torch.float32)},
        {"baseline": torch.ones(2, 2, 2, dtype=torch.float64)},
        {"baseline": torch.full((2, 3, 2), float("nan"), dtype=torch.float64)},
    ],
)
def test_token_ig_invalid_requests(kwargs):
    model = TextClassifier()
    with pytest.raises(ValueError):
        TokenIG(model, model.embedding)(_inputs(), **kwargs)
    assert model.training
    assert not model.embedding._forward_hooks


def test_token_ig_inference_mode_and_missing_layer():
    model = TextClassifier()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference mode"):
        TokenIG(model, model.embedding)(_inputs())
    with pytest.raises(ValueError, match="belong"):
        TokenIG(model, nn.Embedding(4, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_token_ig_zero_path_and_single_class(dtype):
    model = TextClassifier().to(dtype)
    result = TokenIG(model, model.embedding)(
        _inputs(), 0, token_mask=torch.zeros(2, 3, dtype=torch.bool), steps=2, max_steps=4
    )
    assert result.attributions.count_nonzero() == 0
    torch.testing.assert_close(result.scores, result.baseline_scores)


def test_token_ig_resolves_gradient_near_zero_baseline():
    model = TextClassifier()

    def sharp(input_ids, attention_mask):
        states = model.embedding(input_ids) * attention_mask.unsqueeze(-1)
        return (states / (states.square() + 1e-4).sqrt()).sum((1, 2)).unsqueeze(-1)

    model.forward = sharp
    result = TokenIG(model, model.embedding)(_inputs(), steps=8, max_steps=512, tolerance=1e-6)
    expected = sharp(**_inputs()).detach()
    torch.testing.assert_close(result.attributions.sum(-1), expected, atol=1e-6, rtol=0)
    assert result.steps > 16
