import pytest
import torch
from torch import nn

from torchcam.methods import TAM


def _inputs(dtype=torch.float64):
    head = nn.Linear(3, 3, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.eye(3, dtype=dtype))
        head.bias.fill_(10)  # Equation (1) uses weights only, even when a head has a bias.
    visual = torch.tensor(
        [[[1, 4, 2], [2, 1, 3], [3, 2, 1]], [[1, 2, 5], [4, 1, 2], [2, 3, 4]]], dtype=dtype
    ).unsqueeze(0)
    context = torch.tensor([[[0, 0, 1], [1, 0, 2], [100, 0, 1000]]], dtype=dtype)
    return head, visual, context, torch.tensor([[0, 1, 2]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("kernel_size", [1, 3])
def test_tam_reference_parity(dtype, normalized, kernel_size):
    head, visual, context, ids = _inputs(dtype)
    # Golden maps: equation (4), NumPy lstsq for equation (5), and the authors' rank Gaussian filter.
    expected = torch.tensor(
        [
            [[0, 1.4354243542435425, 0], [3.0442804428044283, 0, 0.8708487084870851]],
            [[0, 0.3756827102529987, 2.4495668984910495], [0, 2.9153230058500093, 0]],
        ]
        if kernel_size == 1
        else [
            [
                [0.5176772746780427, 0.6563606179452193, 0.2958181911398182],
                [0.9985765672761355, 0.3794879965792799, 0.7838797934139747],
            ],
            [
                [1.1174212020398895, 0.4024199081279833, 2.0459338533987887],
                [0.4082483142106481, 0.4719554141066802, 0.9443170239398895],
            ],
        ],
        dtype=dtype,
    )
    if normalized:
        expected -= expected.amin((1, 2), keepdim=True)
        expected /= expected.amax((1, 2), keepdim=True) + 1e-8
    maps = TAM(head, kernel_size)(
        [2, 0], visual.repeat(2, 1, 1, 1), context.repeat(2, 1, 1), ids.repeat(2, 1), normalized
    )
    torch.testing.assert_close(maps, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_tam_low_precision_and_autocast(dtype):
    head, visual, context, ids = _inputs()
    expected = TAM(head)(2, visual, context, ids)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        maps = TAM(head.to(dtype))(2, visual.to(dtype), context.to(dtype), ids)
    assert maps.dtype == dtype
    torch.testing.assert_close(maps, expected.to(dtype), atol=0, rtol=0)


def test_tam_without_autocast_backend(monkeypatch):
    head, visual, context, ids = _inputs()
    expected = TAM(head)(2, visual, context, ids)
    monkeypatch.setattr(torch.amp, "is_autocast_available", lambda _device: False)
    monkeypatch.setattr(torch, "autocast", lambda **_kwargs: pytest.fail("This backend has no autocast"))
    torch.testing.assert_close(TAM(head)(2, visual, context, ids), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("kernel_size", [1, 3])
def test_tam_complete_cancellation(dtype, normalized, kernel_size):
    head = nn.Linear(1, 2, bias=False, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.1], [0.3]], dtype=dtype))
    visual = torch.arange(10, 70, 10, dtype=dtype).reshape(1, 2, 3, 1)
    maps = TAM(head, kernel_size)(0, visual, torch.ones(1, 1, 1, dtype=dtype), torch.tensor([[1]]), normalized)
    assert maps.count_nonzero() == 0


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e6])
@pytest.mark.parametrize(("dtype", "delta"), [(torch.float32, 1e-4), (torch.float64, 1e-10)])
def test_tam_preserves_weak_residual(scale, dtype, delta):
    head = nn.Linear(2, 2, bias=False, dtype=dtype)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1, 0], [1, delta]], dtype=dtype))
    visual = torch.tensor([[[[1, 0], [1, 0]], [[1, 0], [1, 1]]]], dtype=dtype) * scale
    context = torch.tensor([[[1.0, 0]]], dtype=dtype)
    maps = TAM(head, kernel_size=1)(1, visual, context, torch.tensor([[0]]), normalized=False)
    expected = torch.tensor([[[0, 0], [0, 0.75 * delta]]], dtype=dtype)
    torch.testing.assert_close(maps / scale, expected, rtol=1e-3, atol=delta * 1e-3)


def test_tam_smoothing_preserves_tiny_maps():
    head = nn.Linear(1, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        head.weight.fill_(1)
    visual = torch.tensor([[[[0], [1], [0]], [[3], [0], [1]]]], dtype=torch.float64)
    context, ids = torch.empty(1, 0, 1, dtype=torch.float64), torch.empty(1, 0, dtype=torch.long)
    extractor = TAM(head)
    expected = extractor(0, visual, context, ids, normalized=False)
    actual = extractor(0, visual * 1e-10, context, ids, normalized=False)
    torch.testing.assert_close(actual / 1e-10, expected)


@pytest.mark.parametrize("shape", [(1, 1), (1, 4), (2, 3)])
@pytest.mark.parametrize("value", [0.0, 2.0])
def test_tam_constant_maps_and_empty_context(shape, value):
    head = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        head.weight.fill_(1)
    visual = torch.full((1, *shape, 1), value)
    context, ids = torch.empty(1, 0, 1), torch.empty(1, 0, dtype=torch.long)
    extractor = TAM(head, kernel_size=5)
    torch.testing.assert_close(extractor(0, visual, context, ids, normalized=False), visual.squeeze(-1))
    assert extractor(0, visual, context, ids).count_nonzero() == 0


def test_tam_repeated_tokens_no_grad_or_forward(monkeypatch):
    head, visual, context, ids = _inputs()
    original = visual.clone()
    visual.requires_grad_()
    monkeypatch.setattr(head, "forward", lambda *_args: pytest.fail("TAM must not project the full vocabulary"))
    maps = TAM(head, kernel_size=1)(2, visual, context, torch.full_like(ids, 2), normalized=False)
    torch.testing.assert_close(maps, visual[..., 2])
    torch.testing.assert_close(visual, original)
    assert not maps.requires_grad
    assert head.weight.grad is None


@pytest.mark.parametrize("token_id", [-1, 3, [2, 1], [0.5], "cat", True])
def test_tam_invalid_target(token_id):
    head, visual, context, ids = _inputs()
    with pytest.raises(ValueError, match=r"token|vocabulary"):
        TAM(head)(token_id, visual, context, ids)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("visual_features", torch.empty(1, 0, 3, 3)),
        ("visual_features", torch.empty(1, 2, 3)),
        ("visual_features", torch.empty(1, 2, 3, 4)),
        ("context_features", torch.empty(1, 3, 4)),
        ("context_features", torch.empty(2, 3, 3)),
        ("context_features", torch.ones(1, 3, 3, dtype=torch.long)),
        ("context_features", torch.empty(1, 3, 3, device="meta")),
        ("context_ids", torch.tensor([[0, -1, 2]])),
        ("context_ids", torch.tensor([[0, 3, 2]])),
        ("context_ids", torch.ones(1, 3)),
        ("context_ids", torch.zeros(1, 2, dtype=torch.long)),
    ],
)
def test_tam_invalid_states(argument, value):
    head, visual, context, ids = _inputs()
    kwargs = {"visual_features": visual, "context_features": context, "context_ids": ids, argument: value}
    with pytest.raises(ValueError):
        TAM(head)(2, **kwargs)


@pytest.mark.parametrize("kernel_size", [0, 2, -1, 1.5, True])
def test_tam_invalid_constructor(kernel_size):
    with pytest.raises((ValueError, TypeError)):
        TAM(nn.Linear(3, 3), kernel_size)
    with pytest.raises(TypeError, match="head"):
        TAM(nn.Identity())
