# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Sequence
from contextlib import nullcontext

import torch
from torch import Tensor, nn

from .core import _CAM

__all__ = ["DEXAR", "TAM"]


class TAM:
    """Extract the visual Token Activation Map from
    ["Token Activation Map to Visually Explain Multimodal LLMs"](https://arxiv.org/abs/2506.23270).

    TAM projects final language-model states onto output-head weights, removes the least-squares estimate of
    earlier text's visual interference, and applies the paper's rank Gaussian filter. Identical context token IDs
    are excluded from interference. The output-head bias is unused, following equation (1).

    This extractor consumes states from an existing forward; it installs no hooks and runs no model forwards or
    backward passes. Unlike class CAM extractors, it returns one tensor, shaped ``(N, H, W)``. Only visual maps are
    returned, with optional normalization over each image rather than joint image/text normalization.

    Args:
        token_classifier: the model's linear vocabulary output head, usually ``model.get_output_embeddings()``
        kernel_size: positive odd rank Gaussian window size; use 1 to disable smoothing

    Raises:
        TypeError: if the output head or kernel size has an invalid type
        ValueError: if the kernel size is not positive and odd
    """

    def __init__(self, token_classifier: nn.Linear, kernel_size: int = 3) -> None:
        if not isinstance(token_classifier, nn.Linear):
            raise TypeError("`token_classifier` must be an nn.Linear vocabulary head")
        if not isinstance(kernel_size, int) or isinstance(kernel_size, bool):
            raise TypeError("`kernel_size` must be an integer")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("`kernel_size` must be positive and odd")
        self.token_classifier = token_classifier
        self.kernel_size = kernel_size

    @torch.no_grad()
    def __call__(
        self,
        token_id: int | list[int],
        visual_features: Tensor,
        context_features: Tensor,
        context_ids: Tensor,
        normalized: bool = True,
    ) -> Tensor:
        """Explain one vocabulary token per image using final language-model states.

        Args:
            token_id: vocabulary ID shared by the batch, or one ID per sample
            visual_features: image-token states reshaped to ``(N, H, W, C)`` in spatial order
            context_features: earlier text states shaped ``(N, S, C)``, aligned with ``context_ids``
            context_ids: earlier text vocabulary IDs shaped ``(N, S)``; exclude image, padding and special tokens
            normalized: whether to normalize each visual map to [0, 1]

        Returns:
            visual activation maps shaped ``(N, H, W)``, on the visual features' device and dtype

        Raises:
            ValueError: if shapes, devices, feature dtypes or vocabulary IDs are incompatible
        """  # noqa: DOC502
        self._validate_inputs(token_id, visual_features, context_features, context_ids)
        ids = [token_id] * visual_features.shape[0] if isinstance(token_id, int) else token_id
        target_ids = torch.tensor(ids, device=visual_features.device, dtype=torch.long)
        # Accumulate in float32 for half-precision models, without converting the full vocabulary head.
        dtype = torch.float64 if visual_features.dtype == torch.float64 else torch.float32
        device_type = visual_features.device.type
        autocast = (
            torch.autocast(device_type=device_type, enabled=False)
            if torch.amp.is_autocast_available(device_type)
            else nullcontext()
        )
        with autocast:
            visual = visual_features.flatten(1, 2).to(dtype)
            weights = self.token_classifier.weight[target_ids].to(dtype)
            maps = (visual * weights.unsqueeze(1)).sum(-1).relu()
            tolerance = maps.amax(-1, keepdim=True) * (16 * torch.finfo(dtype).eps)
            relevance = (context_features.to(dtype) * weights.unsqueeze(1)).sum(-1).relu()
            relevance.masked_fill_(context_ids == target_ids.unsqueeze(1), 0)
            relevance /= relevance.sum(-1, keepdim=True) + 1e-8
            context_weights = self.token_classifier.weight[context_ids].to(dtype)
            context_maps = visual.bmm(context_weights.transpose(1, 2)).relu()
            interference = context_maps.bmm(relevance.unsqueeze(-1)).squeeze(-1)
            denominator = interference.square().sum(-1, keepdim=True)
            scale = (maps * interference).sum(-1, keepdim=True) / denominator.masked_fill(denominator == 0, 1)
            maps = (maps - scale * interference).relu()
            # Avoid amplifying round-off from complete cancellation into a strong heatmap.
            maps.masked_fill_(maps.amax(-1, keepdim=True) <= tolerance, 0)
            maps = maps.reshape(visual_features.shape[:3])
            maps = self._filter(maps)
            if normalized:
                maps = _CAM._normalize(maps)  # noqa: SLF001
        return maps.to(visual_features.dtype)

    def _validate_inputs(self, token_id: int | list[int], visual: Tensor, context: Tensor, ids: Tensor) -> None:
        """Validate spatial states and aligned context."""  # noqa: DOC501
        if visual.ndim != 4 or any(dim == 0 for dim in visual.shape):
            raise ValueError("`visual_features` must have non-empty shape (N, H, W, C)")
        if context.ndim != 3 or context.shape[0] != visual.shape[0] or context.shape[2] != visual.shape[3]:
            raise ValueError("`context_features` must have shape (N, S, C) matching the visual states")
        if ids.shape != context.shape[:2] or ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("`context_ids` must be an integer tensor of shape (N, S)")
        weight = self.token_classifier.weight
        if visual.shape[-1] != weight.shape[-1]:
            raise ValueError("feature width must match the vocabulary head")
        if any(tensor.device != visual.device for tensor in (context, ids, weight)):
            raise ValueError("states, token IDs and vocabulary head must be on the same device")
        if not visual.is_floating_point() or not context.is_floating_point():
            raise ValueError("feature states must be floating-point tensors")
        targets = [token_id] * visual.shape[0] if isinstance(token_id, int) else token_id
        if not isinstance(targets, list) or len(targets) != visual.shape[0]:
            raise ValueError("`token_id` must be an integer or one vocabulary ID per sample")
        if any(not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < weight.shape[0] for idx in targets):
            raise ValueError("target token IDs must be within the vocabulary")
        if ((ids < 0) | (ids >= weight.shape[0])).any():
            raise ValueError("context token IDs must be within the vocabulary")

    def _filter(self, maps: Tensor) -> Tensor:
        """Apply a reflected rank Gaussian filter, including singleton spatial dimensions."""  # noqa: DOC201
        if self.kernel_size == 1:
            return maps
        radius = self.kernel_size // 2
        indices = []
        for length in maps.shape[-2:]:
            positions = torch.arange(-radius, length + radius, device=maps.device)
            reflected = positions.remainder(max(2 * (length - 1), 1))
            indices.append(length - 1 - (reflected - (length - 1)).abs())
        padded = maps[:, indices[0]][:, :, indices[1]]
        windows = padded.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).flatten(-2).sort(-1).values
        mean = windows.mean(-1)
        variation = windows.std(-1, correction=0) / mean.masked_fill(mean == 0, 1)
        ranks = torch.arange(self.kernel_size**2, device=maps.device) - self.kernel_size**2 // 2
        weights = torch.exp(-ranks.square() / (2 * variation.square().unsqueeze(-1)).clamp_min(1e-8))
        return (windows * weights).sum(-1) / weights.sum(-1)


class DEXAR:
    """Token and sequence attribution from [DEX-AR](https://arxiv.org/abs/2603.06302), equations (4)-(6).

    Differentiate each layer's own token logit against its attention. Weight positive visual gradients by
    each head's positive visual-minus-text maximum. Aggregate normalized token maps by visual relevance.
    Model extraction stays with the caller; no hooks are installed.

    Args:
        grid_shape: explicit visual grid ``(height, width)``, including rectangular grids

    Raises:
        ValueError: if the grid is not a pair of positive integers
    """

    def __init__(self, grid_shape: tuple[int, int]) -> None:
        if (
            not isinstance(grid_shape, tuple)
            or len(grid_shape) != 2
            or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in grid_shape)
        ):
            raise ValueError("`grid_shape` must be a tuple of two positive integers")
        self.grid_shape = grid_shape

    def __call__(
        self,
        scores: Sequence[Tensor],
        attentions: Sequence[Tensor],
        visual_mask: Tensor,
        *,
        retain_graph: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Explain a token from the prefix that predicts it, before appending that token.

        Args:
            scores: each layer's selected vocabulary logits ``(N,)`` from the last query state, with the
                model's final norm applied once. Use raw logits, not probabilities.
            attentions: matching downstream-used post-softmax tensors ``(N, heads, queries, keys)``
            visual_mask: boolean ``(keys,)``, shared across the batch, selecting visual keys in spatial order.
                Use unpadded prefixes; other keys, including special tokens, are textual context.
            retain_graph: whether to keep the graph for another attribution on the same forward

        Returns:
            detached normalized maps ``(N, height, width)`` and weights ``(N,)`` on the attention device,
            accumulated in float32 (float64 for double attention)

        Raises:
            ValueError: if layer tensors, scores, visual keys or grid shapes are incompatible
            RuntimeError: if autograd is disabled or the logits are disconnected from their attention tensors
        """  # noqa: DOC502
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("DEXAR requires a forward and attribution with gradient tracking enabled")
        self._validate_inputs(scores, attentions, visual_mask)
        dtype = torch.float64 if attentions[0].dtype == torch.float64 else torch.float32
        maps, image_scores, text_scores = [], [], []
        for idx, (score, attention) in enumerate(zip(scores, attentions, strict=True)):
            try:
                grad = torch.autograd.grad(score.sum(), attention, retain_graph=retain_graph or idx < len(scores) - 1)[
                    0
                ]
            except RuntimeError as exc:
                raise RuntimeError(f"layer {idx} token logit is not connected to its attention probabilities") from exc
            # ReLU follows the official implementation and the paper's Appendix G.1.
            row = grad[:, :, -1].detach().to(dtype).relu()
            visual = row[:, :, visual_mask]
            image, text = visual.amax(-1), row[:, :, ~visual_mask].amax(-1)
            maps.append((visual * (image - text).relu().unsqueeze(-1)).sum(1))
            image_scores.append(image.amax(1))
            text_scores.append(text.amax(1))
        maps = torch.stack(maps).sum(0).reshape(scores[0].shape[0], *self.grid_shape)
        weights = (torch.stack(image_scores).amax(0) - torch.stack(text_scores).amax(0)).relu()
        return self._normalize(maps), weights

    def _validate_inputs(self, scores: Sequence[Tensor], attentions: Sequence[Tensor], visual_mask: Tensor) -> None:
        if not scores or len(scores) != len(attentions):
            raise ValueError("provide one selected token logit for each attention layer, with at least one layer")
        first = attentions[0]
        if first.ndim != 4 or any(dim == 0 for dim in first.shape):
            raise ValueError("attention probabilities must have non-empty shape (N, heads, queries, keys)")
        if visual_mask.ndim != 1 or visual_mask.shape[0] != first.shape[-1] or visual_mask.dtype != torch.bool:
            raise ValueError("`visual_mask` must be a boolean tensor shaped (keys,)")
        count = int(visual_mask.sum())
        if count != self.grid_shape[0] * self.grid_shape[1] or count == visual_mask.numel():
            raise ValueError("visual keys must match `grid_shape` and leave at least one textual key")
        for score, attention in zip(scores, attentions, strict=True):
            if (
                attention.ndim != 4
                or any(dim == 0 for dim in attention.shape)
                or attention.shape[0] != first.shape[0]
                or attention.shape[-1] != first.shape[-1]
                or score.shape != (first.shape[0],)
            ):
                raise ValueError(
                    "layer scores must have shape (N,) and attention layers must share batch and key counts"
                )
            if any(tensor.device != first.device for tensor in (score, attention, visual_mask)):
                raise ValueError("scores, attention probabilities and visual mask must be on the same device")
            if not score.is_floating_point() or not attention.is_floating_point():
                raise ValueError("scores and attention probabilities must be floating-point tensors")
            if not score.requires_grad or not attention.requires_grad:
                raise RuntimeError("DEXAR requires differentiable scores and attention probabilities")

    @torch.no_grad()
    def aggregate(self, maps: Tensor, weights: Tensor) -> Tensor:
        """Combine individually normalized token maps, following paper equation (6).

        Args:
            maps: token maps shaped ``(N, tokens, height, width)``; min-max normalized per token before weighting
            weights: non-negative token relevance weights shaped ``(N, tokens)`` from ``__call__``

        Returns:
            detached maps ``(N, height, width)``; constant maps and all-zero weights yield zeros

        Raises:
            ValueError: if shapes, devices, dtypes or finite non-negative values are incompatible
        """
        if (
            maps.ndim != 4
            or any(dim == 0 for dim in maps.shape)
            or maps.shape[-2:] != self.grid_shape
            or weights.shape != maps.shape[:2]
        ):
            raise ValueError("`maps` must have shape (N, tokens, height, width) and `weights` shape (N, tokens)")
        if maps.device != weights.device or not maps.is_floating_point() or not weights.is_floating_point():
            raise ValueError("maps and weights must be floating-point tensors on the same device")
        if not maps.isfinite().all() or not weights.isfinite().all() or (maps < 0).any() or (weights < 0).any():
            raise ValueError("maps and weights must be finite and non-negative")
        dtype = torch.float64 if torch.float64 in {maps.dtype, weights.dtype} else torch.float32
        normalized = self._normalize(maps.to(dtype).clone())
        weights = weights.to(dtype).clone()
        weights.masked_fill_(normalized.amax((-2, -1)) == 0, 0)
        weights /= weights.amax(1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
        return self._normalize((normalized * weights[..., None, None]).sum(1))

    @staticmethod
    def _normalize(maps: Tensor) -> Tensor:
        # Gradient products can be very small; a fixed epsilon would erase their contrast.
        return _CAM._normalize(maps, spatial_dims=2, eps=torch.finfo(maps.dtype).tiny)  # noqa: SLF001
