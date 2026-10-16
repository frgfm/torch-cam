# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from ._utils import _model_eval

__all__ = ["TokenAblation", "TokenAblationResult", "TokenAttribution", "TokenIG"]


@dataclass(frozen=True)
class TokenAblationResult:
    """Single-token replacement effects; interactions mean contributions need not add up.

    Attributes:
        scores: original scores shaped ``(N, classes)``; no calibration is applied
        attributions: original minus replaced-token scores, shaped ``(N, classes, tokens)``; excluded tokens are zero
    """

    scores: Tensor
    attributions: Tensor


class TokenAblation:
    """Explain a classifier by replacing one input token at a time, without gradients.

    All classes share one forward per selected token position, plus the original forward. Attention, positions and
    other context stay fixed. Use a replacement ID appropriate to the model, such as its pretrained mask token.
    Models must be deterministic in evaluation mode, have independent batch samples, and not mutate their parameters.

    Args:
        model: PyTorch classifier accepting keyword arguments, including ``input_ids``
        output_transform: optional function selecting floating-point ``(N, classes)`` scores from model output

    Raises:
        ValueError: if an embedding renormalizes weights during forward
    """

    def __init__(self, model: nn.Module, output_transform: Callable[[Any], Tensor] | None = None) -> None:
        if any(isinstance(layer, nn.Embedding) and layer.max_norm is not None for layer in model.modules()):
            raise ValueError("embeddings with `max_norm` mutate weights during forward and are unsupported")
        self.model = model
        self.output_transform = output_transform

    def __call__(
        self, inputs: Mapping[str, Any], replacement_id: int, *, token_mask: Tensor | None = None
    ) -> TokenAblationResult:
        """Return signed effects of individual token replacements.

        Args:
            inputs: unchanged model keyword arguments; ``input_ids`` must be integer ``(N, tokens)``
            replacement_id: vocabulary ID substituted at each selected position; attention remains unchanged
            token_mask: boolean ``(N, tokens)`` mask on the IDs' device; omitted means every token

        Returns:
            detached original scores and signed effects. Positive values mean replacement lowers a class score.
            Effects are marginal changes with other tokens fixed, not additive contributions or token probabilities.

        Raises:
            ValueError: if IDs, replacement, mask or model scores are invalid
        """
        ids = inputs.get("input_ids")
        if (
            not isinstance(ids, Tensor)
            or ids.ndim != 2
            or not ids.numel()
            or ids.dtype not in {torch.int32, torch.int64}
        ):
            raise ValueError("`input_ids` must be non-empty integer (N, tokens)")
        if not isinstance(replacement_id, int) or isinstance(replacement_id, bool) or replacement_id < 0:
            raise ValueError("`replacement_id` must be a nonnegative vocabulary ID")
        mask = torch.ones_like(ids, dtype=torch.bool) if token_mask is None else token_mask
        if (
            not isinstance(mask, Tensor)
            or mask.shape != ids.shape
            or mask.dtype != torch.bool
            or mask.device != ids.device
        ):
            raise ValueError("`token_mask` must be boolean (N, tokens) on the IDs' device")

        def score(batch: Mapping[str, Any]) -> Tensor:
            output = self.model(**batch)
            value = output if self.output_transform is None else self.output_transform(output)
            shape = value.shape if isinstance(value, Tensor) else ()
            if (
                len(shape) != 2
                or shape[0] != ids.shape[0]
                or not shape[1]
                or not value.is_floating_point()
                or not value.isfinite().all()
            ):
                raise ValueError("model scores must be finite floating-point (N, classes)")
            return value.detach()

        with _model_eval(self.model), torch.no_grad():
            scores = score(inputs)
            maps = scores.new_zeros((*scores.shape, ids.shape[1]))
            for position in mask.any(0).nonzero().flatten().tolist():
                replaced = ids.clone()
                replaced[mask[:, position], position] = replacement_id
                changed = score({**inputs, "input_ids": replaced})
                if changed.shape != scores.shape:
                    raise ValueError("model score shape changed after token replacement")
                maps[:, :, position] = (scores - changed) * mask[:, position, None]
        return TokenAblationResult(scores, maps)


@dataclass(frozen=True)
class TokenAttribution:
    """Input-token contributions relative to an embedding baseline.

    Attributes:
        scores: original model scores, shaped ``(N, classes)``; no calibration is applied
        baseline_scores: model scores with selected embeddings replaced by the baseline
        attributions: signed contributions shaped ``(N, selected_classes, tokens)``; excluded tokens are zero
        class_indices: output columns, in the same order as the attribution maps
        completeness_error: signed attribution sum minus score change, shaped ``(N, selected_classes)``
        steps: quadrature nodes used by the final integration; earlier refinement passes are additional
    """

    scores: Tensor
    baseline_scores: Tensor
    attributions: Tensor
    class_indices: tuple[int, ...]
    completeness_error: Tensor
    steps: int


@lru_cache(maxsize=16)
def _quadrature(steps: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(steps)
    u = (nodes + 1) / 2
    # Reparameterize the same straight line to resolve gradients near the zero baseline.
    return u**3, weights * 1.5 * u**2


class TokenIG:
    """Explain text classifiers with embedding Integrated Gradients.

    Implements [Integrated Gradients](https://arxiv.org/abs/1703.01365) on a straight embedding path.
    Gauss-Legendre integration is refined until signed completeness and successive token maps agree.
    One forward per node is shared across selected classes. No tokenizer or model-specific dependency is needed.
    Temporary embedding hooks and evaluation mode are restored on success or failure; parameter gradients are preserved.

    Args:
        model: differentiable PyTorch classifier; call its underlying module rather than a no-grad prediction API
        embedding_layer: input word-embedding module, or its name in ``model``; it must run exactly once per forward
        output_transform: optional function selecting a floating-point ``(N, classes)`` score tensor from model output;
            for example ``lambda output: output.logits`` or ``lambda output: output[0]``. Prefer pre-softmax logits.

    Raises:
        ValueError: if the embedding layer is outside the model or renormalizes weights during forward
    """

    def __init__(
        self,
        model: nn.Module,
        embedding_layer: nn.Module | str,
        output_transform: Callable[[Any], Tensor] | None = None,
    ) -> None:
        self.model = model
        self.embedding_layer = (
            model.get_submodule(embedding_layer) if isinstance(embedding_layer, str) else embedding_layer
        )
        if self.embedding_layer not in model.modules():
            raise ValueError("`embedding_layer` must belong to `model`")
        if isinstance(self.embedding_layer, nn.Embedding) and self.embedding_layer.max_norm is not None:
            raise ValueError("embeddings with `max_norm` mutate weights during forward and are unsupported")
        self.output_transform = output_transform

    def __call__(
        self,
        inputs: Mapping[str, Any],
        class_idx: int | list[int] | None = None,
        *,
        token_mask: Tensor | None = None,
        baseline: Tensor | None = None,
        steps: int = 32,
        max_steps: int = 4096,
        tolerance: float = 1e-4,
        stability_tolerance: float = 0.01,
    ) -> TokenAttribution:
        """Attribute selected class scores to the input tokens.

        Args:
            inputs: keyword arguments forwarded unchanged to the model, including IDs, masks and position metadata
            class_idx: one output column, an ordered list, or ``None`` for all columns; shared across the batch
            token_mask: boolean ``(N, tokens)`` mask selecting embeddings to vary; omitted means every token
            baseline: reference embeddings matching the embedding output's shape, device and dtype; defaults to zero
            steps: initial quadrature nodes, doubled on refinement; at least two integration passes are required
            max_steps: maximum nodes per integration pass, at least twice ``steps``
            tolerance: maximum absolute signed completeness error for every sample and selected class
            stability_tolerance: maximum L1 map change divided by map L1 norm on successive passes

        Returns:
            detached scores, signed maps and convergence information. Positive values support a class relative to
            the baseline; negative values oppose it. These are score contributions, not token probabilities.

        Raises:
            ValueError: if inputs, scores, baseline, selection or integration settings are invalid
            RuntimeError: if gradients are unavailable, the embedding module is reused, or integration fails to converge

        Note:
            Batch samples must be independent. Use eager inference and non-reentrant gradient checkpointing.
            An outer ``torch.no_grad()`` is supported, but inference mode and no-grad model forwards are not.
            Numerical convergence does not establish causal influence or bias; compare baselines and controlled edits.
        """
        if torch.is_inference_mode_enabled():
            raise RuntimeError("TokenIG cannot run in inference mode; use a differentiable model forward")
        if any(not isinstance(n, int) or isinstance(n, bool) for n in (steps, max_steps)):
            raise ValueError("`steps` and `max_steps` must be integers")
        if steps < 2 or max_steps < 2 * steps:
            raise ValueError("use `steps` >= 2 and `max_steps` >= 2 * steps")
        if not 0 < tolerance < float("inf") or not 0 < stability_tolerance < float("inf"):
            raise ValueError("convergence tolerances must be finite and positive")
        with _model_eval(self.model), torch.enable_grad():
            return self._explain(
                inputs, class_idx, token_mask, baseline, steps, max_steps, tolerance, stability_tolerance
            )

    @contextmanager
    def _forward(
        self, inputs: Mapping[str, Any], replacement: Tensor | None = None
    ) -> Generator[tuple[Tensor, Tensor], None, None]:
        captured: list[Tensor] = []
        record = True

        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Tensor) -> Tensor:
            if record:
                captured.append(
                    output.detach().clone() if replacement is None and isinstance(output, Tensor) else output
                )
            return output if replacement is None else replacement.clone()

        handle = self.embedding_layer.register_forward_hook(hook)
        try:
            output = self.model(**inputs)
            if len(captured) != 1:
                raise RuntimeError("the input embedding module must run exactly once per forward")
            embedding = captured[0]
            record = False  # Keep replacement active during checkpoint backward, without retaining recomputed outputs.
            if not isinstance(embedding, Tensor) or embedding.ndim != 3 or not embedding.is_floating_point():
                raise ValueError("embedding output must be floating-point (N, tokens, channels)")
            scores = output if self.output_transform is None else self.output_transform(output)
            if (
                not isinstance(scores, Tensor)
                or scores.ndim != 2
                or scores.shape[0] != embedding.shape[0]
                or any(size == 0 for size in scores.shape)
                or not scores.is_floating_point()
            ):
                raise ValueError(
                    "model scores must be floating-point (N, classes); supply `output_transform` if needed"
                )
            yield scores, embedding
        finally:
            handle.remove()

    def _explain(
        self,
        inputs: Mapping[str, Any],
        class_idx: int | list[int] | None,
        mask: Tensor | None,
        baseline: Tensor | None,
        steps: int,
        max_steps: int,
        tolerance: float,
        stability_tolerance: float,
    ) -> TokenAttribution:
        with torch.no_grad():
            with self._forward(inputs) as (scores, embedding):
                actual = embedding.detach()
            if mask is not None and (
                mask.shape != actual.shape[:2] or mask.dtype != torch.bool or mask.device != actual.device
            ):
                raise ValueError("`token_mask` must be boolean (N, tokens) on the embeddings' device")
            if baseline is None:
                baseline = torch.zeros_like(actual)
            if baseline.shape != actual.shape or baseline.device != actual.device or baseline.dtype != actual.dtype:
                raise ValueError("`baseline` must match the embeddings' shape, device and dtype")
            reference = baseline.detach() if mask is None else torch.where(mask.unsqueeze(-1), baseline, actual)
            if not actual.isfinite().all() or not reference.isfinite().all():
                raise ValueError("input and reference embeddings must be finite")
            difference = actual - reference
            with self._forward(inputs, reference) as (base, _):
                baseline_scores = base.detach()
        indices = (
            list(range(scores.shape[1]))
            if class_idx is None
            else [class_idx]
            if isinstance(class_idx, int)
            else class_idx
        )
        if (
            not isinstance(indices, list)
            or not indices
            or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < scores.shape[1] for i in indices)
        ):
            raise ValueError("`class_idx` must select valid output columns")
        if len(set(indices)) != len(indices):
            raise ValueError("`class_idx` must not contain duplicates")
        if not scores[:, indices].isfinite().all() or not baseline_scores[:, indices].isfinite().all():
            raise ValueError("selected class scores must be finite")
        # Float64 accumulation improves cancellation without changing model precision; MPS requires float32.
        dtype = torch.float32 if actual.device.type == "mps" else torch.float64
        delta = scores[:, indices].to(dtype) - baseline_scores[:, indices].to(dtype)
        previous = None
        while steps <= max_steps:
            maps = self._integrate(inputs, reference, difference, indices, steps, dtype)
            error = maps.sum(-1) - delta
            if previous is not None:
                change = (maps - previous).abs().sum(-1) / maps.abs().sum(-1).clamp_min(tolerance)
                if error.abs().max() <= tolerance and change.max() <= stability_tolerance:
                    return TokenAttribution(
                        scores.detach(), baseline_scores.detach(), maps, tuple(indices), error, steps
                    )
            previous = maps
            steps *= 2
        raise RuntimeError(
            "TokenIG did not converge; increase `max_steps`, review the baseline or use higher precision"
        )

    def _integrate(
        self,
        inputs: Mapping[str, Any],
        reference: Tensor,
        difference: Tensor,
        indices: list[int],
        steps: int,
        dtype: torch.dtype,
    ) -> Tensor:
        maps = torch.zeros((reference.shape[0], len(indices), reference.shape[1]), device=reference.device, dtype=dtype)
        weighted_difference = difference.to(dtype)
        for alpha, weight in zip(*_quadrature(steps), strict=True):
            current = (reference + float(alpha) * difference).detach().requires_grad_(True)
            with self._forward(inputs, current) as (scores, _):
                if not scores.requires_grad:
                    raise RuntimeError("model scores must retain gradients; call the underlying differentiable module")
                for position, index in enumerate(indices):
                    grad = torch.autograd.grad(
                        scores[:, index].sum(), current, retain_graph=position + 1 < len(indices), allow_unused=True
                    )[0]
                    if grad is None:
                        raise RuntimeError("model scores are disconnected from the input embeddings")
                    maps[:, position] += (grad.detach().to(dtype) * weighted_difference).sum(-1) * float(weight)
        if not maps.isfinite().all():
            raise ValueError("token contributions must be finite")
        return maps
