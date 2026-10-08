import pytest
import torch
from torch import nn

from torchcam.methods import DEXAR


def _layer(rows, dtype=torch.float64, scale=1):
    rows = torch.tensor(rows, dtype=dtype) * scale
    attention = torch.zeros(rows.shape[0], rows.shape[1], 3, rows.shape[2], dtype=dtype, requires_grad=True)
    scores = (attention[:, :, -1] * rows).sum((1, 2))
    return scores, attention


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("scale", [1, 1e-4])
def test_dexar_layer_logits_head_filter_and_rectangular_grid(dtype, scale):
    # Visual keys are offset from the beginning; negative gradients must not become absolute relevance.
    scores0, attention0 = _layer([[[1, 0, 1, 2, 3, 4, 5, 2], [-8, -9, -1, -2, -3, -4, -5, -1]]], dtype, scale)
    scores1, attention1 = _layer([[[1, 6, 5, 4, 3, 2, 1, 1]]], dtype, scale)
    mask = torch.tensor([False, True, True, True, True, True, True, False])
    maps, weights = DEXAR((2, 3))([scores0, scores1], [attention0, attention1], mask)
    # Head weights are 5-2=3 and 6-1=5. Raw sum: [30, 28, 26, 24, 22, 20] * scale**2.
    expected = torch.tensor([[[1, 0.8, 0.6], [0.4, 0.2, 0]]], dtype=maps.dtype)
    torch.testing.assert_close(maps, expected, atol=max(2 * torch.finfo(dtype).eps, 1e-6), rtol=0)
    torch.testing.assert_close(weights, torch.tensor([4 * scale], dtype=weights.dtype), atol=scale * 0.02, rtol=0)
    assert maps.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert not maps.requires_grad
    assert not weights.requires_grad
    assert attention0.grad is None
    assert attention1.grad is None


def test_dexar_token_weight_uses_separate_global_maxima():
    score, attention = _layer([[[0, 5, 1, 1], [12, 10, 2, 0]]])
    maps, weights = DEXAR((1, 2))([score], [attention], torch.tensor([False, True, True, False]))
    # First head survives, but the global text maximum exceeds the global visual maximum.
    torch.testing.assert_close(maps, torch.tensor([[[1.0, 0.0]]], dtype=maps.dtype))
    torch.testing.assert_close(weights, torch.zeros_like(weights))


def test_dexar_all_heads_filtered_and_batch_independence():
    score, attention = _layer([[[0, 2, 1, 0]], [[3, 2, 1, 0]]])
    maps, weights = DEXAR((1, 2))([score], [attention], torch.tensor([False, True, True, False]))
    torch.testing.assert_close(maps, torch.tensor([[[1, 0]], [[0, 0]]], dtype=maps.dtype))
    torch.testing.assert_close(weights, torch.tensor([2, 0], dtype=weights.dtype))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_dexar_sequence_uses_normalized_token_maps(dtype):
    extractor = DEXAR((1, 3))
    maps = torch.tensor([[[[0.01, 0, 0]], [[0, 100, 0]], [[0, 0, 999]]]], dtype=dtype)
    weights = torch.tensor([[3, 1, 0]], dtype=dtype)
    original = maps.clone()
    sequence = extractor.aggregate(maps, weights)
    # Weight normalized maps, not raw gradient sums; the huge filler-token map has zero weight.
    torch.testing.assert_close(sequence, torch.tensor([[[1, 1 / 3, 0]]], dtype=sequence.dtype))
    torch.testing.assert_close(maps, original)
    assert not sequence.requires_grad


def test_dexar_constant_maps_and_zero_sequence_weights():
    extractor = DEXAR((2, 3))
    maps = torch.ones(2, 3, 2, 3)
    assert not extractor.aggregate(maps, torch.ones(2, 3)).any()
    maps[0, 0, 0, 0] = 2
    assert not extractor.aggregate(maps, torch.zeros(2, 3)).any()


@pytest.mark.parametrize("values", [[1e38] * 4, [1e-8, 1e38, 0, 0]])
def test_dexar_finite_weights_do_not_overflow_or_mutate(values):
    maps = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]]])
    weights = torch.tensor([values])
    original = weights.clone()
    torch.testing.assert_close(DEXAR((1, 2)).aggregate(maps, weights), maps[:, 0])
    torch.testing.assert_close(weights, original)


def test_dexar_retain_graph_does_not_change_parameter_gradients():
    score, attention = _layer([[[0, 2, 1]]])
    parameter = nn.Parameter(torch.ones((), dtype=score.dtype))
    parameter.grad = torch.tensor(7, dtype=score.dtype)
    score *= parameter
    extractor = DEXAR((1, 2))
    mask = torch.tensor([False, True, True])
    first = extractor([score], [attention], mask, retain_graph=True)
    second = extractor([score], [attention], mask)
    for actual, expected in zip(second, first, strict=True):
        torch.testing.assert_close(actual, expected)
    assert parameter.grad == 7
    assert attention.grad is None


def test_dexar_disconnected_returned_attention():
    score, attention = _layer([[[0, 2, 1]]])
    disconnected = attention.detach().requires_grad_()
    with pytest.raises(RuntimeError, match="not connected"):
        DEXAR((1, 2))([score], [disconnected], torch.tensor([False, True, True]))


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
def test_dexar_requires_autograd(mode):
    score, attention = _layer([[[0, 2, 1]]])
    with mode(), pytest.raises(RuntimeError, match="gradient tracking"):
        DEXAR((1, 2))([score], [attention], torch.tensor([False, True, True]))
