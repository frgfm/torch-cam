# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from contextlib import nullcontext

import torch
from torch import Tensor, nn

from .core import _CAM

__all__ = ["TAM"]


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
