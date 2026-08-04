# Copyright (C) 2022-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol, cast

import torch


class _CAMExtractor(Protocol):
    model: torch.nn.Module

    def __call__(
        self,
        class_idx: int | list[int],
        scores: torch.Tensor | None = None,
        normalized: bool = True,
        **kwargs: Any,
    ) -> list[torch.Tensor]: ...

    def fuse_cams(self, cams: list[torch.Tensor]) -> torch.Tensor: ...

    def _hooks_off(self) -> AbstractContextManager[None]: ...


@contextmanager
def _model_eval(model: torch.nn.Module) -> Iterator[None]:
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def _get_scores(
    cam_extractor: _CAMExtractor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    logits = cam_extractor.model(input_tensor)
    return cast(torch.Tensor, logits if logits_fn is None else logits_fn(logits))


def _resolve_class_idx(scores: torch.Tensor, class_idx: int | list[int] | None) -> tuple[int | list[int], torch.Tensor]:
    batch_size, num_classes = scores.shape
    if class_idx is None:
        target_indices = scores.argmax(dim=-1)
        return target_indices.detach().cpu().tolist(), target_indices

    if isinstance(class_idx, int):
        if class_idx < 0 or class_idx >= num_classes:
            raise ValueError("class_idx is out of range")
        return class_idx, torch.full((batch_size,), class_idx, device=scores.device, dtype=torch.long)

    if not isinstance(class_idx, list) or any(not isinstance(idx, int) or isinstance(idx, bool) for idx in class_idx):
        raise TypeError("class_idx must be an integer, a list of integers, or None")
    if len(class_idx) != batch_size:
        raise ValueError("per-sample class_idx must match the batch size")
    if any(idx < 0 or idx >= num_classes for idx in class_idx):
        raise ValueError("class_idx is out of range")
    return class_idx, torch.tensor(class_idx, device=scores.device)


def _get_cam(
    cam_extractor: _CAMExtractor,
    scores: torch.Tensor,
    class_idx: int | list[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    extractor_idx, target_indices = _resolve_class_idx(scores, class_idx)
    cams = cam_extractor(extractor_idx, scores)
    return cam_extractor.fuse_cams(cams), target_indices


def _resize_cam(cam: torch.Tensor, spatial_shape: tuple[int, ...]) -> torch.Tensor:
    interpolation_mode = {1: "linear", 2: "bilinear", 3: "trilinear"}.get(len(spatial_shape))
    if interpolation_mode is None:
        raise ValueError("only 1D, 2D, and 3D spatial inputs are supported")
    return torch.nn.functional.interpolate(
        cam.unsqueeze(1),
        spatial_shape,
        mode=interpolation_mode,
        align_corners=False,
    ).squeeze(1)


class ClassificationMetric:
    r"""Implements Average Drop and Increase in Confidence from ["Grad-CAM++: Improved Visual Explanations for Deep
    Convolutional Networks."](https://arxiv.org/pdf/1710.11063.pdf).

    The raw aggregated metric is computed as follows:

    $$
    \forall N, C, S_1, \ldots, S_d \in \mathbb{N},
    \forall X \in \mathbb{R}^{N \times C \times S_1 \times \cdots \times S_d},
    \forall m \in \mathcal{M}, \forall c \in \mathcal{C}, \\
    AvgDrop_{m, c}(X) = \frac{1}{N} \sum\limits_{i=1}^N f_{m, c}(X_i) \\
    IncrConf_{m, c}(X) = \frac{1}{N} \sum\limits_{i=1}^N g_{m, c}(X_i)
    $$

    where $\mathcal{C}$ is the set of class activation generators,
    $\mathcal{M}$ is the set of classification models,
    with the function $f_{m, c}$ defined as:

    $$
    \forall x \in \mathbb{R}^{3 \times H \times W},
    f_{m, c}(x) = \frac{\max(0, m(x) - m(E_{m, c}(x) * x))}{m(x)}
    $$

    where $E_{m, c}(x)$ is the class activation map of $m$ for input $x$ with method $m$,
    resized to (H, W),

    and with the function $g_{m, c}$ defined as:

    $$
    \forall x \in \mathbb{R}^{3 \times H \times W},\quad
    g_{m, c}(x) =
    \begin{cases}
        1 & \text{if } m(x) < m(E_{m, c}(x) \cdot x) \\
        0 & \text{otherwise}
    \end{cases}
    $$

    Example:
        ```python
        from functools import partial
        from torchcam.metrics import ClassificationMetric
        metric = ClassificationMetric(cam_extractor, partial(torch.softmax, dim=-1))
        metric.update(input_tensor)
        metric.summary()
        ```
    """

    def __init__(
        self,
        cam_extractor: _CAMExtractor,
        logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        # This is a typa, I don't know how to rites
        self.cam_extractor = cam_extractor
        self.logits_fn = logits_fn
        self.reset()

    def update(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | list[int] | None = None,
    ) -> None:
        """Update the state of the metric with new predictions.

        Args:
            input_tensor: preprocessed input tensor for the model
            class_idx: class index to focus on (default: index of the top predicted class for each sample)
        """
        with _model_eval(self.cam_extractor.model):
            scores = _get_scores(self.cam_extractor, self.logits_fn, input_tensor)
            cam, target_indices = _get_cam(self.cam_extractor, scores, class_idx)
            discard = torch.isnan(cam).reshape(input_tensor.shape[0], -1).any(dim=-1)
            nan_count = discard.sum().item()
            if discard.all():
                self.nan_count += nan_count
                return

            cam = _resize_cam(cam[~discard], tuple(input_tensor.shape[2:]))
            scores = scores[~discard].gather(1, target_indices[~discard].unsqueeze(1)).squeeze(1)
            target_indices = target_indices[~discard]
            valid_input = input_tensor[~discard]

            with self.cam_extractor._hooks_off(), torch.inference_mode():  # noqa: SLF001
                masked_scores = _get_scores(self.cam_extractor, self.logits_fn, cam.unsqueeze(1) * valid_input)
            masked_scores = masked_scores.gather(1, target_indices.unsqueeze(1)).squeeze(1)
            drop = torch.relu(scores - masked_scores).div(scores + 1e-7)
            increase = scores < masked_scores

        self.drop += drop.sum().item()
        self.increase += increase.sum().item()
        self.total += cam.shape[0]
        self.nan_count += nan_count

    def summary(self) -> dict[str, float]:
        """Computes the aggregated metrics.

        Returns:
            a dictionary with the average drop and the increase in confidence

        Raises:
            AssertionError: if the metric has not been updated
        """
        if self.total == 0:
            raise AssertionError("you need to update the metric before getting the summary")

        return {
            "avg_drop": self.drop / self.total,
            "conf_increase": self.increase / self.total,
        }

    def reset(self) -> None:
        """Reset the state of the metric."""
        self.drop = 0.0
        self.increase = 0.0
        self.total = 0
        self.nan_count = 0


class DeletionInsertionMetric:
    r"""Implements deletion and insertion faithfulness metrics from ["RISE: Randomized Input Sampling for
    Explanation of Black-box Models."](https://arxiv.org/abs/1806.07421).

    Spatial positions are ranked from the highest to the lowest CAM value. Deletion progressively replaces the
    highest-ranked positions with a baseline, while insertion progressively restores them from the original input.
    The mask is shared across channels. Both scores are areas under the selected-class score curves, integrated
    against the actual perturbed fraction with :func:`torch.trapezoid`.

    Example:
        ```python
        from functools import partial
        from torchcam.metrics import DeletionInsertionMetric
        metric = DeletionInsertionMetric(cam_extractor, partial(torch.softmax, dim=-1))
        metric.update(input_tensor)
        metric.summary()
        ```

    Args:
        cam_extractor: CAM extractor used to rank spatial positions
        logits_fn: optional function applied to the model output before selecting class scores
        steps: maximum number of perturbation intervals
        baseline: baseline tensor, callable producing one, or ``None`` to use zeros
        batch_size: maximum number of perturbed inputs scored per model forward

    Raises:
        TypeError: if an argument has an invalid type
        ValueError: if ``steps`` or ``batch_size`` is not positive
    """

    def __init__(
        self,
        cam_extractor: _CAMExtractor,
        logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        *,
        steps: int = 20,
        baseline: torch.Tensor | Callable[[torch.Tensor], torch.Tensor] | None = None,
        batch_size: int = 32,
    ) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool):
            raise TypeError("steps must be an integer")
        if steps <= 0:
            raise ValueError("steps must be positive")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if baseline is not None and not isinstance(baseline, torch.Tensor) and not callable(baseline):
            raise TypeError("baseline must be a tensor, a callable, or None")

        self.cam_extractor = cam_extractor
        self.logits_fn = logits_fn
        self.steps = steps
        self.baseline = baseline
        self.batch_size = batch_size
        self.reset()

    def _get_baseline(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.baseline is None:
            return torch.zeros_like(input_tensor)
        if isinstance(self.baseline, torch.Tensor):
            baseline = self.baseline
        else:
            baseline = self.baseline(input_tensor)
            if not isinstance(baseline, torch.Tensor):
                raise TypeError("baseline callable must return a tensor")
        try:
            return torch.broadcast_to(baseline.to(input_tensor), input_tensor.shape)
        except RuntimeError as exc:
            raise ValueError("baseline must be broadcastable to the input shape") from exc

    def _score_perturbations(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        ranks: torch.Tensor,
        target_indices: torch.Tensor,
        jobs: list[tuple[int, int, int, int]],
    ) -> torch.Tensor:
        sample_indices = torch.tensor([job[0] for job in jobs], device=input_tensor.device)
        curve_indices = torch.tensor([job[1] for job in jobs], device=input_tensor.device)
        perturb_counts = torch.tensor([job[3] for job in jobs], device=input_tensor.device)
        relevant = ranks[sample_indices] < perturb_counts.unsqueeze(1)
        use_original = torch.where(curve_indices.unsqueeze(1) == 1, relevant, ~relevant)
        perturbed = torch.where(
            use_original.unsqueeze(1),
            input_tensor.flatten(2)[sample_indices],
            baseline.flatten(2)[sample_indices],
        ).reshape((-1, *input_tensor.shape[1:]))
        scores = _get_scores(self.cam_extractor, self.logits_fn, perturbed)
        return scores.gather(1, target_indices[sample_indices].unsqueeze(1)).squeeze(1)

    def _compute_curves(
        self,
        input_tensor: torch.Tensor,
        baseline: torch.Tensor,
        ranks: torch.Tensor,
        target_indices: torch.Tensor,
        original_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_samples = input_tensor.shape[0]
        num_positions = ranks.shape[1]
        step_size = (num_positions + self.steps - 1) // self.steps
        counts = [*range(0, num_positions, step_size), num_positions]
        fractions = torch.tensor(counts, device=original_scores.device, dtype=torch.float32).div_(num_positions)
        deletion = original_scores.new_empty((num_samples, len(counts)))
        insertion = original_scores.new_empty((num_samples, len(counts)))
        deletion[:, 0] = original_scores
        insertion[:, -1] = original_scores

        interior_jobs = [
            (curve_idx, point_idx, count) for point_idx, count in enumerate(counts[1:-1], 1) for curve_idx in range(2)
        ]
        jobs = [
            (sample_idx, curve_idx, point_idx, count)
            for sample_idx in range(num_samples)
            for curve_idx, point_idx, count in [(2, len(counts) - 1, num_positions), *interior_jobs]
        ]

        with self.cam_extractor._hooks_off(), torch.inference_mode():  # noqa: SLF001
            for start in range(0, len(jobs), self.batch_size):
                chunk = jobs[start : start + self.batch_size]
                selected_scores = self._score_perturbations(input_tensor, baseline, ranks, target_indices, chunk)

                for score, job in zip(selected_scores, chunk, strict=True):
                    if job[1] == 0:
                        deletion[job[0], job[2]] = score
                    elif job[1] == 1:
                        insertion[job[0], job[2]] = score
                    else:
                        deletion[job[0], job[2]] = score
                        insertion[job[0], 0] = score

        return fractions, deletion, insertion

    def update(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | list[int] | None = None,
    ) -> None:
        """Update the metric with a batch of inputs.

        Args:
            input_tensor: preprocessed model input
            class_idx: shared class index, one class index per sample, or ``None`` to use original top predictions
        """
        with _model_eval(self.cam_extractor.model):
            scores = _get_scores(self.cam_extractor, self.logits_fn, input_tensor)
            cam, target_indices = _get_cam(self.cam_extractor, scores, class_idx)
            discard = torch.isnan(cam).reshape(input_tensor.shape[0], -1).any(dim=-1)
            nan_count = discard.sum().item()
            if discard.all():
                self.nan_count += nan_count
                return

            with torch.no_grad():
                baseline = self._get_baseline(input_tensor)[~discard]
            valid_input = input_tensor[~discard]
            cam = _resize_cam(cam[~discard], tuple(input_tensor.shape[2:]))
            target_indices = target_indices[~discard]
            original_scores = scores[~discard].gather(1, target_indices.unsqueeze(1)).squeeze(1).detach()
            order = torch.argsort(cam.flatten(1), dim=1, descending=True, stable=True)
            ranks = torch.argsort(order, dim=1)
            fractions, deletion, insertion = self._compute_curves(
                valid_input,
                baseline,
                ranks,
                target_indices,
                original_scores,
            )
            deletion_auc = torch.trapezoid(deletion, fractions, dim=1)
            insertion_auc = torch.trapezoid(insertion, fractions, dim=1)

        self.deletion += deletion_auc.sum().item()
        self.insertion += insertion_auc.sum().item()
        self.total += valid_input.shape[0]
        self.nan_count += nan_count

    def summary(self) -> dict[str, float]:
        """Compute the mean deletion and insertion AUCs.

        Returns:
            deletion and insertion AUCs averaged over non-NaN samples

        Raises:
            AssertionError: if the metric has not been updated with a valid CAM
        """
        if self.total == 0:
            raise AssertionError("you need to update the metric before getting the summary")
        return {
            "deletion_auc": self.deletion / self.total,
            "insertion_auc": self.insertion / self.total,
        }

    def reset(self) -> None:
        """Reset the accumulated metric state."""
        self.deletion = 0.0
        self.insertion = 0.0
        self.total = 0
        self.nan_count = 0
