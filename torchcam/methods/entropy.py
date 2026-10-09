# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import torch
from torch import Tensor

from .core import _CAM

__all__ = ["EntropyGradient"]


class EntropyGradient:
    """Extract the map primitive from
    ["Entropy-Gradient Grounding"](https://arxiv.org/abs/2604.08456), equations (1)-(4).

    Take the channel L2 norm of next-token Shannon entropy gradients with respect to projected visual
    embeddings entering the language model. This measures uncertainty sensitivity, not word attribution.
    Region selection and crop-and-refine are excluded. The caller supplies tensors from an existing
    differentiable forward and their spatial order; this extractor installs no hooks.

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
        logits: Tensor,
        embeddings: Tensor,
        visual_tokens: Tensor | None = None,
        *,
        normalized: bool = True,
        retain_graph: bool = False,
    ) -> Tensor:
        """Map uncertainty sensitivity at one caller-selected decoding step.

        Args:
            logits: full vocabulary logits ``(N, vocabulary)`` at the chosen query, e.g. ``output.logits[:, -1]``
                for an unpadded prefix; no temperature or sampling filters are applied
            embeddings: exact projected or mixed visual/text input embeddings ``(N, tokens, channels)`` used
                by the forward; final hidden states and post-forward slices/copies are not valid substitutes
            visual_tokens: optional boolean mask ``(tokens,)`` or unique int32/int64 indices in row-major
                order, shared across the batch; omitted means all tokens are visual
            normalized: min-max normalize each image to [0, 1]; constant maps become zero
            retain_graph: keep the graph for another attribution/backward; default releases it

        Returns:
            detached ``(N, height, width)`` maps on the embeddings' device; accumulation uses float32 or
            float64 if either input is double precision. Entropy uses natural logs.

        Raises:
            ValueError: if shapes, devices, dtypes, finite values or token selection are incompatible
            RuntimeError: if autograd is disabled, inputs are disconnected/non-differentiable, or the graph is freed

        Note:
            Sample entropies are summed, assuming independent samples. Parameter gradients are preserved.
            Accumulation cannot recover gradients already lost inside a low-precision model.
        """
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("EntropyGradient requires a forward and attribution with gradient tracking enabled")
        indices = self._validate_inputs(logits, embeddings, visual_tokens)
        if not logits.requires_grad or not embeddings.requires_grad:
            raise RuntimeError("EntropyGradient requires differentiable logits and embeddings")
        dtype = torch.float64 if torch.float64 in {logits.dtype, embeddings.dtype} else torch.float32
        log_probs = logits.to(dtype).log_softmax(-1)
        probs = log_probs.exp()
        # Zero-probability terms contribute zero, including finite extreme logits whose subtraction overflows.
        entropy = -(probs * log_probs.masked_fill(probs == 0, 0)).sum(-1)
        # Differentiate the original tensor, then select tokens: a post-forward slice is not a graph ancestor.
        grad = torch.autograd.grad(entropy.sum(), embeddings, retain_graph=retain_graph, allow_unused=True)[0]
        if grad is None:
            raise RuntimeError("logits are disconnected from `embeddings`; pass the exact tensor used by the forward")
        grad = grad.detach().to(dtype)
        if indices is not None:
            grad = grad.index_select(1, indices)
        if not grad.isfinite().all():
            raise ValueError("selected embedding gradients must be finite")
        # Rescaling prevents the sum of squared channels from underflowing or overflowing.
        scale = grad.abs().amax(-1, keepdim=True)
        maps = (grad / scale.masked_fill(scale == 0, 1)).norm(p=2, dim=-1) * scale.squeeze(-1)
        if not maps.isfinite().all():
            raise ValueError("gradient channel norms must be finite")
        maps = maps.reshape(embeddings.shape[0], *self.grid_shape)
        if normalized:
            # A fixed epsilon would erase contrast in small uncertainty gradients.
            maps = _CAM._normalize(maps, spatial_dims=2, eps=torch.finfo(dtype).tiny)  # noqa: SLF001
        return maps

    def _validate_inputs(self, logits: Tensor, embeddings: Tensor, visual_tokens: Tensor | None) -> Tensor | None:
        if logits.ndim != 2 or any(dim == 0 for dim in logits.shape):
            raise ValueError("`logits` must have non-empty shape (N, vocabulary)")
        if embeddings.ndim != 3 or any(dim == 0 for dim in embeddings.shape):
            raise ValueError("`embeddings` must have non-empty shape (N, tokens, channels)")
        if logits.shape[0] != embeddings.shape[0]:
            raise ValueError("logits and embeddings must have the same batch size")
        if logits.device != embeddings.device:
            raise ValueError("logits and embeddings must be on the same device")
        if not logits.is_floating_point() or not embeddings.is_floating_point():
            raise ValueError("logits and embeddings must be floating-point tensors")
        if not logits.isfinite().all() or not embeddings.isfinite().all():
            raise ValueError("logits and embeddings must be finite")
        return self._visual_indices(embeddings, visual_tokens)

    def _visual_indices(self, embeddings: Tensor, visual_tokens: Tensor | None) -> Tensor | None:
        count = self.grid_shape[0] * self.grid_shape[1]
        if visual_tokens is None:
            if embeddings.shape[1] != count:
                raise ValueError("visual token count must match `grid_shape`")
            return None
        if not isinstance(visual_tokens, Tensor) or visual_tokens.ndim != 1:
            raise ValueError("`visual_tokens` must be a one-dimensional boolean mask or integer indices")
        if visual_tokens.device != embeddings.device:
            raise ValueError("`visual_tokens` must be on the embeddings' device")
        if visual_tokens.dtype == torch.bool:
            if visual_tokens.shape[0] != embeddings.shape[1]:
                raise ValueError("visual mask length must match the embedding token count")
            indices = visual_tokens.nonzero().squeeze(-1)
        elif visual_tokens.dtype in {torch.int32, torch.int64}:
            indices = visual_tokens.to(torch.long)
            if ((indices < 0) | (indices >= embeddings.shape[1])).any() or indices.unique().numel() != indices.numel():
                raise ValueError("visual indices must be unique and within the embedding token range")
        else:
            raise ValueError("`visual_tokens` must contain booleans or int32/int64 indices")
        if indices.numel() != count:
            raise ValueError("selected visual token count must match `grid_shape`")
        return indices
