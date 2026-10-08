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

    Differentiate the Shannon entropy of the full next-token distribution with respect to projected visual
    embeddings entering the language model, then take the gradient's L2 norm over embedding channels.
    This measures sensitivity of predictive uncertainty, rather than support for a particular generated word.
    Region selection, smoothing, cropping and iterative refinement are outside this extractor's scope.

    Like TAM, this extractor consumes an existing forward, installs no hooks and returns a single tensor.
    The caller supplies the exact downstream-used embeddings and their spatial order; final language-model
    states, detached copies and slices created after the forward are not interchangeable with those inputs.

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
            logits: full vocabulary logits shaped ``(N, vocabulary)``; select the next-token query, e.g.
                ``output.logits[:, -1]`` for an unpadded prefix. No temperature or sampling filters are applied.
            embeddings: exact projected visual embeddings shaped ``(N, tokens, channels)`` used by the forward,
                or the mixed visual/text input embeddings with ``visual_tokens`` selecting image positions
            visual_tokens: optional one-dimensional boolean mask of length ``tokens``, or unique int32/int64
                indices in row-major spatial order, shared across the batch. If omitted, all tokens are visual.
            normalized: whether to min-max normalize each image to [0, 1]; constant maps become zero
            retain_graph: whether to retain the forward graph for another attribution or backward pass.
                The default releases it; reusing it then requires a new forward.

        Returns:
            detached maps shaped ``(N, height, width)`` on the embeddings' device. Entropy and channel norms
            accumulate in float32, or float64 if either input is double precision. Entropy uses natural logs.

        Raises:
            ValueError: if shapes, devices, dtypes, finite values or visual-token selection are incompatible
            RuntimeError: if gradient tracking is disabled, inputs are not differentiable, the logits are
                disconnected from the embeddings, or their forward graph has already been released

        Note:
            The batch objective is the sum of sample entropies, assuming samples do not interact in the model.
            ``autograd.grad`` leaves model parameter gradients intact. Accumulation cannot recover gradients
            already rounded to zero by a low-precision forward or backward.
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
