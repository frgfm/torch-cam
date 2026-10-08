# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Sequence

import torch
from torch import Tensor

from .core import _CAM

__all__ = ["DEXAR"]


class DEXAR:
    """Explain generated tokens with
    ["DEX-AR: A Dynamic Explainability Method for Autoregressive Vision-Language Models"](https://arxiv.org/abs/2603.06302).

    Each selected layer's token logit is differentiated with respect to that layer's post-softmax attention.
    Positive gradients in the last query row are weighted by the head's maximum visual gradient minus its
    maximum textual gradient, clamped at zero. The weighted sum over heads and layers yields a token map.
    ``aggregate`` combines normalized token maps using their visual relevance weights (paper equation (6)).

    Like TAM, this extractor consumes an existing forward and installs no hooks. Model-specific extraction,
    vocabulary projection and spatial ordering belong to the caller. Unlike TAM, it requires autograd and
    attention probabilities that actually participate in the computation of the layer logits.

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
        """Explain the next token from the prefix that predicts it, before appending that token.

        Args:
            scores: one selected vocabulary logit per sample, shaped ``(N,)``, for each selected layer.
                Project that layer's last query state through the model's final norm and vocabulary head;
                do not normalize a state that already includes the final norm a second time.
            attentions: corresponding downstream-used probability tensors shaped ``(N, heads, queries, keys)``.
                Cached forwards with one query are allowed if their attention tensors remain differentiable.
            visual_mask: boolean tensor shaped ``(keys,)``, shared across the batch, selecting visual keys in
                spatial order. All other keys are textual context, including special tokens. Supply unpadded
                prefixes containing the prompt and only previously generated tokens.
            retain_graph: whether to keep the graph for another attribution on the same forward

        Returns:
            normalized token maps ``(N, height, width)`` and visual relevance weights ``(N,)``. Both are detached,
            on the attention device, accumulated in float32 (float64 for double-precision attention).

        Raises:
            ValueError: if layer tensors, scores, visual keys or grid shapes are incompatible
            RuntimeError: if autograd is disabled or the logits are disconnected from their attention tensors
        """  # noqa: DOC502
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("DEXAR requires a forward and attribution with gradient tracking enabled")
        self._validate_inputs(scores, attentions, visual_mask)
        dtype = torch.float64 if attentions[0].dtype == torch.float64 else torch.float32
        visual_grads, image_scores, text_scores = [], [], []
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
            visual_grads.append(visual)
            image_scores.append(visual.amax(-1))
            text_scores.append(row[:, :, ~visual_mask].amax(-1))
        # Concatenation permits different head counts at different layers.
        visual = torch.cat(visual_grads, dim=1)
        image = torch.cat(image_scores, dim=1)
        text = torch.cat(text_scores, dim=1)
        maps = (visual * (image - text).relu().unsqueeze(-1)).sum(1)
        maps = maps.reshape(maps.shape[0], *self.grid_shape)
        weights = (image.amax(1) - text.amax(1)).relu()
        return self._normalize(maps), weights

    def _validate_inputs(self, scores: Sequence[Tensor], attentions: Sequence[Tensor], visual_mask: Tensor) -> None:
        if not scores or len(scores) != len(attentions):
            raise ValueError("provide one selected token logit for each attention layer, with at least one layer")
        first = attentions[0]
        if first.ndim != 4 or any(dim == 0 for dim in first.shape):
            raise ValueError("attention probabilities must have non-empty shape (N, heads, queries, keys)")
        if visual_mask.ndim != 1 or visual_mask.shape[0] != first.shape[-1] or visual_mask.dtype != torch.bool:
            raise ValueError("`visual_mask` must be a boolean tensor shaped (keys,)")
        if visual_mask.device != first.device:
            raise ValueError("scores, attention probabilities and visual mask must be on the same device")
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
            if score.device != first.device or attention.device != first.device:
                raise ValueError("scores, attention probabilities and visual mask must be on the same device")
            if not score.is_floating_point() or not attention.is_floating_point():
                raise ValueError("scores and attention probabilities must be floating-point tensors")
            if not score.requires_grad or not attention.requires_grad:
                raise RuntimeError("DEXAR requires differentiable scores and attention probabilities")

    @torch.no_grad()
    def aggregate(self, maps: Tensor, weights: Tensor) -> Tensor:
        """Combine token maps using visual relevance weights, following paper equation (6).

        Args:
            maps: token maps shaped ``(N, tokens, height, width)``; min-max normalized per token before weighting
            weights: non-negative token relevance weights shaped ``(N, tokens)`` from ``__call__``

        Returns:
            detached sequence maps shaped ``(N, height, width)``; constant maps and all-zero weights yield zeros

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
        return self._normalize((normalized * weights.to(dtype)[..., None, None]).sum(1))

    @staticmethod
    def _normalize(maps: Tensor) -> Tensor:
        # Gradient products can be very small; a fixed epsilon would erase their contrast.
        return _CAM._normalize(maps, spatial_dims=2, eps=torch.finfo(maps.dtype).tiny)  # noqa: SLF001
