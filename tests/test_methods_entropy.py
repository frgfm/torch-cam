import pytest
import torch
from torch import nn

from torchcam.methods import EntropyGradient


def _inputs(dtype=torch.float64):
    generator = torch.Generator().manual_seed(42)
    embeddings = torch.randn(2, 8, 3, generator=generator, dtype=dtype).requires_grad_()
    weights = torch.randn(8, 3, 4, generator=generator, dtype=dtype) / 5
    logits = torch.einsum("ntc,tcv->nv", embeddings, weights)
    return embeddings, weights, logits


def _analytic_map(logits, weights):
    # Independent chain rule: dH/dz_j = -p_j * (log(p_j) + H).
    probs = logits.detach().exp()
    probs /= probs.sum(-1, keepdim=True)
    entropy = -(probs * probs.log()).sum(-1, keepdim=True)
    gradient = torch.einsum("nv,tcv->ntc", -probs * (probs.log() + entropy), weights)
    return gradient.square().sum(-1).sqrt()


@pytest.mark.parametrize(
    "tokens",
    [None, torch.tensor([6, 1, 4, 7, 3, 0]), torch.tensor([True, True, False, True, True, False, True, True])],
)
def test_entropy_gradient_matches_chain_rule_and_spatial_order(tokens):
    embeddings, weights, logits = _inputs()
    expected = _analytic_map(logits, weights)
    if tokens is not None:
        expected = expected[:, tokens]
    grid = (2, 4) if tokens is None else (2, 3)
    expected = expected.reshape(2, *grid)
    extractor = EntropyGradient(grid)
    raw = extractor(logits, embeddings, tokens, normalized=False, retain_graph=True)
    torch.testing.assert_close(raw, expected)
    assert raw.dtype == torch.float64
    assert not raw.requires_grad
    expected -= expected.amin((1, 2), keepdim=True)
    expected /= expected.amax((1, 2), keepdim=True)
    torch.testing.assert_close(extractor(logits, embeddings, tokens), expected)


def test_entropy_gradient_mixed_precision():
    embeddings, weights, logits = _inputs(torch.bfloat16)
    expected = _analytic_map(logits.float(), weights.float()).bfloat16().float().reshape(2, 2, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        maps = EntropyGradient((2, 4))(logits, embeddings, normalized=False)
    assert maps.dtype == torch.float32
    torch.testing.assert_close(maps, expected, atol=2e-3, rtol=2e-2)


def test_entropy_gradient_ignores_caller_autocast():
    embeddings = torch.zeros(1, 6, 1, requires_grad=True)
    weights = (torch.arange(1, 7)[:, None] * 1000.0).expand(6, 3).clone()
    weights[:, 0] += torch.arange(6, 0, -1)
    logits = embeddings.flatten(1) @ weights + torch.tensor([1.0, -1.0, 0.0])
    extractor = EntropyGradient((2, 3))
    expected = extractor(logits, embeddings, normalized=False, retain_graph=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = extractor(logits, embeddings, normalized=False)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("offset", [0.0, 10000.0])
def test_entropy_gradient_uniform_and_saturated_distributions(offset):
    embeddings = torch.zeros(1, 6, 2, requires_grad=True)
    score = (embeddings * torch.arange(1, 13).reshape(6, 2)).sum((1, 2))
    logits = torch.stack([score + offset, 2 * score, 3 * score], dim=-1)
    extractor = EntropyGradient((2, 3))
    raw = extractor(logits, embeddings, normalized=False, retain_graph=True)
    torch.testing.assert_close(raw, torch.zeros(1, 2, 3), atol=0, rtol=0)
    torch.testing.assert_close(extractor(logits, embeddings), raw, atol=0, rtol=0)


def test_entropy_gradient_preserves_parameter_gradients_and_controls_retention():
    projector, head = nn.Linear(2, 3), nn.Linear(18, 4)
    embeddings = projector(torch.randn(1, 6, 2))  # Non-leaf projected inputs.
    logits = head(embeddings.flatten(1))
    head.weight.grad = torch.full_like(head.weight, 7)
    saved = head.weight.grad.clone()
    extractor = EntropyGradient((2, 3))
    first = extractor(logits, embeddings, retain_graph=True)
    torch.testing.assert_close(head.weight.grad, saved)
    assert projector.weight.grad is None
    logits.sum().backward(retain_graph=True)  # Training remains possible after attribution.
    assert projector.weight.grad is not None
    torch.testing.assert_close(extractor(logits, embeddings), first)
    with pytest.raises(RuntimeError, match=r"second time|freed"):
        extractor(logits, embeddings)


@pytest.mark.parametrize("tokens", [torch.arange(5), torch.tensor([0, 1, 2, 3, 4, 4])])
def test_entropy_gradient_rejects_wrong_count_and_duplicate_tokens(tokens):
    embeddings, _, logits = _inputs()
    with pytest.raises(ValueError, match="visual"):
        EntropyGradient((2, 3))(logits, embeddings, tokens)


def test_entropy_gradient_rejects_post_forward_slice():
    embeddings, _, logits = _inputs()
    with pytest.raises(RuntimeError, match="disconnected"):
        EntropyGradient((2, 4))(logits, embeddings[:, :])
