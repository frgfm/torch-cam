import pytest
import torch
from torch import nn

from torchcam.methods import EntropyGradient


def _inputs(dtype=torch.float64, tokens=6):
    generator = torch.Generator().manual_seed(42)
    embeddings = torch.randn(2, tokens, 3, generator=generator, dtype=dtype).requires_grad_()
    weights = torch.randn(tokens, 3, 4, generator=generator, dtype=dtype) / 5
    logits = torch.einsum("ntc,tcv->nv", embeddings, weights)
    return embeddings, weights, logits


def _analytic_map(logits, weights):
    # dH/dz_j = -p_j * (log(p_j) + H); derive embedding gradients using the linear chain rule.
    probs = logits.detach().exp()
    probs /= probs.sum(-1, keepdim=True)
    entropy = -(probs * probs.log()).sum(-1, keepdim=True)
    gradient = torch.einsum("nv,tcv->ntc", -probs * (probs.log() + entropy), weights)
    return gradient.square().sum(-1).sqrt()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("grid", [(2, 3), (3, 2), (1, 6), (6, 1)])
def test_entropy_gradient_analytic(dtype, normalized, grid):
    embeddings, weights, logits = _inputs(dtype)
    expected = _analytic_map(logits, weights).reshape(2, *grid)
    if normalized:
        expected -= expected.amin((1, 2), keepdim=True)
        expected /= expected.amax((1, 2), keepdim=True)
    maps = EntropyGradient(grid)(logits, embeddings, normalized=normalized)
    torch.testing.assert_close(maps, expected)
    assert maps.device == embeddings.device
    assert maps.dtype == dtype
    assert not maps.requires_grad


@pytest.mark.parametrize("selection", ["mask", "int32", "int64"])
def test_entropy_gradient_selects_original_tensor_and_spatial_order(selection):
    embeddings, weights, logits = _inputs(tokens=8)
    tokens = torch.tensor([6, 1, 4, 7, 3, 0])
    if selection == "mask":
        tokens = torch.tensor([True, True, False, True, True, False, True, True])
        indices = tokens.nonzero().squeeze(-1)
    else:
        tokens = tokens.to(getattr(torch, selection))
        indices = tokens.long()
    expected = _analytic_map(logits, weights)[:, indices].reshape(2, 2, 3)
    maps = EntropyGradient((2, 3))(logits, embeddings, tokens, normalized=False)
    torch.testing.assert_close(maps, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_entropy_gradient_mixed_precision(dtype):
    embeddings, weights, _ = _inputs(torch.float32)
    embeddings = embeddings.to(dtype).detach().requires_grad_()
    weights = weights.to(dtype)
    # Construct a genuine low-precision graph, including half-precision backward into the embeddings.
    logits = torch.einsum("ntc,tcv->nv", embeddings, weights)
    expected = _analytic_map(logits.float(), weights.float()).to(dtype).float().reshape(2, 2, 3)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        maps = EntropyGradient((2, 3))(logits, embeddings, normalized=False)
    assert maps.dtype == torch.float32
    torch.testing.assert_close(maps, expected, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize("scale", [1e-30, 1.0, 1e30])
def test_entropy_gradient_stable_norms_and_small_map_normalization(scale):
    embeddings = torch.zeros(1, 6, 2, requires_grad=True)
    weights = torch.arange(1, 13).reshape(1, 6, 2) * scale
    score = (embeddings * weights).sum((1, 2)) + 1
    logits = torch.stack([score, torch.zeros_like(score)], dim=-1)
    maps = EntropyGradient((2, 3))(logits, embeddings, normalized=False, retain_graph=True)
    expected = _analytic_map(
        torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.stack([weights[0].double(), torch.zeros(6, 2)], dim=-1)
    )
    torch.testing.assert_close(maps.double() / scale, expected.reshape(1, 2, 3) / scale, rtol=1e-6, atol=0)
    normalized = EntropyGradient((2, 3))(logits, embeddings)
    assert normalized.max() == 1
    assert normalized.min() == 0
    assert normalized.isfinite().all()


@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_entropy_gradient_constant_maps(offset):
    embeddings = torch.zeros(1, 6, 2, requires_grad=True)
    score = embeddings.sum((1, 2)) + offset
    logits = torch.stack([score, torch.zeros_like(score)], dim=-1)
    maps = EntropyGradient((2, 3))(logits, embeddings, retain_graph=True)
    assert maps.count_nonzero() == 0
    raw = EntropyGradient((2, 3))(logits, embeddings, normalized=False)
    assert raw.isfinite().all()
    assert bool(raw.count_nonzero()) == bool(offset)


@pytest.mark.parametrize("value", [10000.0, torch.finfo(torch.float32).max])
def test_entropy_gradient_extreme_finite_logits(value):
    embeddings = torch.zeros(1, 6, 2, requires_grad=True)
    score = embeddings.sum((1, 2))
    logits = torch.stack([score + value, score - value], dim=-1)
    maps = EntropyGradient((2, 3))(logits, embeddings)
    assert maps.isfinite().all()
    assert maps.count_nonzero() == 0


def test_entropy_gradient_retention_and_parameter_gradients():
    projector = nn.Linear(2, 3, dtype=torch.float64)
    head = nn.Linear(18, 4, dtype=torch.float64)
    inputs = torch.randn(1, 6, 2, dtype=torch.float64)
    embeddings = projector(inputs)  # Non-leaf projected embeddings, not final language-model states.
    logits = head(embeddings.flatten(1))
    parameters = list(projector.parameters()) + list(head.parameters())
    parameters[0].grad = torch.full_like(parameters[0], 7)
    parameters[-1].grad = torch.full_like(parameters[-1], 3)
    saved = [None if param.grad is None else param.grad.clone() for param in parameters]
    extractor = EntropyGradient((2, 3))
    first = extractor(logits, embeddings, retain_graph=True)
    second = extractor(logits, embeddings, retain_graph=True)
    torch.testing.assert_close(first, second)
    for parameter, original in zip(parameters, saved, strict=True):
        if original is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, original)
    logits.sum().backward(retain_graph=True)  # Retention permits training after attribution.
    assert parameters[1].grad is not None
    torch.testing.assert_close(extractor(logits, embeddings), first)  # Default releases the graph.
    with pytest.raises(RuntimeError, match=r"second time|freed"):
        extractor(logits, embeddings)


@pytest.mark.parametrize("grid", [(0, 3), (2, -1), (2,), (2, 3, 4), [2, 3], (True, 3), (2.0, 3)])
def test_entropy_gradient_invalid_grid(grid):
    with pytest.raises(ValueError, match="grid_shape"):
        EntropyGradient(grid)


@pytest.mark.parametrize(
    "selection",
    [
        torch.ones(6),
        torch.ones(2, 6, dtype=torch.bool),
        torch.ones(5, dtype=torch.bool),
        torch.tensor([0, 1, 2, 3, 4]),
        torch.tensor([0, 1, 2, 3, 4, 6]),
        torch.tensor([0, 1, 2, 3, 4, 4]),
        torch.ones(6, dtype=torch.bool, device="meta"),
    ],
)
def test_entropy_gradient_invalid_selection(selection):
    embeddings, _, logits = _inputs()
    with pytest.raises(ValueError, match=r"visual|indices|mask"):
        EntropyGradient((2, 3))(logits, embeddings, selection)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("logits", torch.empty(2, 0)),
        ("logits", torch.empty(1, 4)),
        ("logits", torch.ones(2, 4, dtype=torch.long)),
        ("logits", torch.full((2, 4), float("inf"))),
        ("logits", torch.empty(2, 4, device="meta")),
        ("embeddings", torch.empty(2, 6)),
        ("embeddings", torch.zeros(2, 5, 3)),
        ("embeddings", torch.ones(2, 6, 3, dtype=torch.long)),
        ("embeddings", torch.full((2, 6, 3), float("nan"))),
    ],
)
def test_entropy_gradient_invalid_inputs(argument, value):
    embeddings, _, logits = _inputs()
    kwargs = {"logits": logits, "embeddings": embeddings, argument: value}
    with pytest.raises(ValueError):
        EntropyGradient((2, 3))(**kwargs)


@pytest.mark.parametrize("argument", ["logits", "embeddings"])
def test_entropy_gradient_detached_inputs(argument):
    embeddings, _, logits = _inputs()
    kwargs = {"logits": logits, "embeddings": embeddings}
    kwargs[argument] = kwargs[argument].detach()
    with pytest.raises(RuntimeError, match="differentiable"):
        EntropyGradient((2, 3))(**kwargs)


@pytest.mark.parametrize("replacement", ["independent", "slice"])
def test_entropy_gradient_disconnected_inputs(replacement):
    embeddings, _, logits = _inputs()
    if replacement == "independent":
        logits = torch.randn(2, 4, requires_grad=True)
    else:
        embeddings = embeddings[:, :6]  # Even an unchanged slice created after the forward is disconnected.
    with pytest.raises(RuntimeError, match="disconnected"):
        EntropyGradient((2, 3))(logits, embeddings)


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_entropy_gradient_disabled_autograd(context):
    embeddings, _, logits = _inputs()
    with context(), pytest.raises(RuntimeError, match="gradient tracking"):
        EntropyGradient((2, 3))(logits, embeddings)


def test_entropy_gradient_nonfinite_backward():
    embeddings = torch.zeros(1, 6, 2, requires_grad=True)
    logits = torch.stack([embeddings.sqrt().sum((1, 2)) + 1, embeddings.sum((1, 2))], dim=-1)
    with pytest.raises(ValueError, match="gradients must be finite"):
        EntropyGradient((2, 3))(logits, embeddings)


def test_entropy_gradient_unrepresentable_channel_norm():
    embeddings = torch.zeros(1, 6, 64, requires_grad=True)
    score = (embeddings * torch.finfo(torch.float32).max).sum((1, 2)) + 1
    logits = torch.stack([score, torch.zeros_like(score)], dim=-1)
    with pytest.raises(ValueError, match="channel norms must be finite"):
        EntropyGradient((2, 3))(logits, embeddings)
