# Evaluation metrics

Apart from qualitative visual comparison, it is important to have a refined evaluation metric for class activation maps. This submodule is dedicated to the evaluation of CAM methods.

## Classification confidence

## ClassificationMetric

```python
ClassificationMetric(cam_extractor: _CAMExtractor, logits_fn: Callable[[Tensor], Tensor] | None = None)
```

Implements Average Drop and Increase in Confidence from ["Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks."](https://arxiv.org/pdf/1710.11063.pdf).

The raw aggregated metric is computed as follows:

[ \\forall N, C, S_1, \\ldots, S_d \\in \\mathbb{N}, \\forall X \\in \\mathbb{R}^{N \\times C \\times S_1 \\times \\cdots \\times S_d}, \\forall m \\in \\mathcal{M}, \\forall c \\in \\mathcal{C}, \\ AvgDrop\_{m, c}(X) = \\frac{1}{N} \\sum\\limits\_{i=1}^N f\_{m, c}(X_i) \\ IncrConf\_{m, c}(X) = \\frac{1}{N} \\sum\\limits\_{i=1}^N g\_{m, c}(X_i) ]

where (\\mathcal{C}) is the set of class activation generators, (\\mathcal{M}) is the set of classification models, with the function (f\_{m, c}) defined as:

[ \\forall x \\in \\mathbb{R}^{3 \\times H \\times W}, f\_{m, c}(x) = \\frac{\\max(0, m(x) - m(E\_{m, c}(x) * x))}{m(x)} ]

where (E\_{m, c}(x)) is the class activation map of (m) for input (x) with method (m), resized to (H, W),

and with the function (g\_{m, c}) defined as:

[ \\forall x \\in \\mathbb{R}^{3 \\times H \\times W},\\quad g\_{m, c}(x) = \\begin{cases} 1 & \\text{if } m(x) < m(E\_{m, c}(x) \\cdot x) \\ 0 & \\text{otherwise} \\end{cases} ]

Example

```python
from functools import partial
from torchcam.metrics import ClassificationMetric
metric = ClassificationMetric(cam_extractor, partial(torch.softmax, dim=-1))
metric.update(input_tensor)
metric.summary()
```

Source code in `torchcam/metrics.py`

```python
def __init__(
    self,
    cam_extractor: _CAMExtractor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> None:
    self.cam_extractor = cam_extractor
    self.logits_fn = logits_fn
    self.reset()
```

### torchcam.metrics.ClassificationMetric.reset

```python
reset() -> None
```

Reset the state of the metric.

Source code in `torchcam/metrics.py`

```python
def reset(self) -> None:
    """Reset the state of the metric."""
    self.drop = 0.0
    self.increase = 0.0
    self.total = 0
    self.nan_count = 0
```

### torchcam.metrics.ClassificationMetric.update

```python
update(input_tensor: Tensor, class_idx: int | list[int] | None = None) -> None
```

Update the state of the metric with new predictions.

| PARAMETER      | DESCRIPTION                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------- |
| `input_tensor` | preprocessed input tensor for the model **TYPE:** `Tensor`                                          |
| `class_idx`    | class index to focus on (default: index of the top predicted class for each sample) **TYPE:** \`int |

Source code in `torchcam/metrics.py`

```python
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
```

### torchcam.metrics.ClassificationMetric.summary

```python
summary() -> dict[str, float]
```

Computes the aggregated metrics.

| RETURNS            | DESCRIPTION                                                       |
| ------------------ | ----------------------------------------------------------------- |
| `dict[str, float]` | a dictionary with the average drop and the increase in confidence |

| RAISES           | DESCRIPTION                        |
| ---------------- | ---------------------------------- |
| `AssertionError` | if the metric has not been updated |

Source code in `torchcam/metrics.py`

```python
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
```

## Deletion and insertion faithfulness

[Deletion and insertion](https://arxiv.org/abs/1806.07421) measure how the model's selected-class score changes as spatial positions are perturbed in descending CAM order. For an input (X), baseline (B), and the set (R_t) containing the top-ranked positions restored or removed by step (t):

\[ D_t[p] = \\begin{cases} B[p] & p \\in R_t \\ X[p] & p \\notin R_t \\end{cases} \]

\[ I_t[p] = \\begin{cases} X[p] & p \\in R_t \\ B[p] & p \\notin R_t. \\end{cases} \]

The same spatial mask is applied to every input channel. If (x_t = |R_t| / P) is the actual perturbed fraction for (P) spatial positions and (s_c) is the selected-class score, TorchCAM computes:

[ \\operatorname{DeletionAUC} = \\operatorname{trapz}(s_c(D_t), x_t), ]

[ \\operatorname{InsertionAUC} = \\operatorname{trapz}(s_c(I_t), x_t). ]

Lower deletion AUC and higher insertion AUC indicate a more faithful ranking. Both the unperturbed and fully perturbed endpoints are included. `steps` is the maximum number of intervals: each interval changes (\\lceil P / \\text{steps} \\rceil) positions, except for the shorter final interval, and integration uses the resulting fractions rather than an assumed uniform grid.

The default baseline is `zeros_like(input_tensor)`. This represents the dataset mean only when inputs were normalized so that the mean maps to zero. Baseline choice can introduce out-of-distribution evidence and materially change both scores. The original RISE evaluation used constant deletion values and a blurred insertion substrate, while this metric deliberately uses one baseline for both curves. To reproduce those two substrates, run the metric separately with each baseline and compare only the corresponding AUC.

`batch_size` limits how many perturbed inputs are scored in one forward pass. It bounds temporary memory but does not reduce the number of perturbed samples. With (S) effective intervals, each valid input requires (2S - 1) additional scoring samples, plus the original CAM-producing forward and any backward pass required by the extractor.

By default, the metric integrates raw model outputs. Pass a function such as softmax for probability curves comparable to the paper; raw-logit AUCs may fall outside ([0, 1]) and should not be compared with probability AUCs.

```python
from functools import partial

import torch

from torchcam.methods import GradCAM
from torchcam.metrics import DeletionInsertionMetric

model.eval()
with GradCAM(model, "layer4") as cam_extractor:
    metric = DeletionInsertionMetric(
        cam_extractor,
        partial(torch.softmax, dim=-1),
        steps=20,
        batch_size=32,
    )
    metric.update(input_tensor)
    scores = metric.summary()
```

Warning

Deletion and insertion test perturbation faithfulness to the model's score. They do not establish localization quality, human interpretability, or causal correctness outside the chosen perturbation and baseline protocol.

## DeletionInsertionMetric

```python
DeletionInsertionMetric(cam_extractor: _CAMExtractor, logits_fn: Callable[[Tensor], Tensor] | None = None, *, steps: int = 20, baseline: Tensor | Callable[[Tensor], Tensor] | None = None, batch_size: int = 32)
```

Implements deletion and insertion faithfulness metrics from ["RISE: Randomized Input Sampling for Explanation of Black-box Models."](https://arxiv.org/abs/1806.07421).

Spatial positions are ranked from the highest to the lowest CAM value. Deletion progressively replaces the highest-ranked positions with a baseline, while insertion progressively restores them from the original input. The mask is shared across channels. Both scores are areas under the selected-class score curves, integrated against the actual perturbed fraction with :func:`torch.trapezoid`.

Example

```python
from functools import partial
from torchcam.metrics import DeletionInsertionMetric
metric = DeletionInsertionMetric(cam_extractor, partial(torch.softmax, dim=-1))
metric.update(input_tensor)
metric.summary()
```

| PARAMETER       | DESCRIPTION                                                                                                          |
| --------------- | -------------------------------------------------------------------------------------------------------------------- |
| `cam_extractor` | CAM extractor used to rank spatial positions **TYPE:** `_CAMExtractor`                                               |
| `logits_fn`     | optional function applied to the model output before selecting class scores **TYPE:** \`Callable\[[Tensor], Tensor\] |
| `steps`         | maximum number of perturbation intervals **TYPE:** `int` **DEFAULT:** `20`                                           |
| `baseline`      | baseline tensor, callable producing one, or None to use zeros **TYPE:** \`Tensor                                     |
| `batch_size`    | maximum number of perturbed inputs scored per model forward **TYPE:** `int` **DEFAULT:** `32`                        |

| RAISES       | DESCRIPTION                            |
| ------------ | -------------------------------------- |
| `TypeError`  | if an argument has an invalid type     |
| `ValueError` | if steps or batch_size is not positive |

Source code in `torchcam/metrics.py`

```python
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
```

### torchcam.metrics.DeletionInsertionMetric.reset

```python
reset() -> None
```

Reset the accumulated metric state.

Source code in `torchcam/metrics.py`

```python
def reset(self) -> None:
    """Reset the accumulated metric state."""
    self.deletion = 0.0
    self.insertion = 0.0
    self.total = 0
    self.nan_count = 0
```

### torchcam.metrics.DeletionInsertionMetric.update

```python
update(input_tensor: Tensor, class_idx: int | list[int] | None = None) -> None
```

Update the metric with a batch of inputs.

| PARAMETER      | DESCRIPTION                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| `input_tensor` | preprocessed model input **TYPE:** `Tensor`                                                             |
| `class_idx`    | shared class index, one class index per sample, or None to use original top predictions **TYPE:** \`int |

Source code in `torchcam/metrics.py`

```python
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
```

### torchcam.metrics.DeletionInsertionMetric.summary

```python
summary() -> dict[str, float]
```

Compute the mean deletion and insertion AUCs.

| RETURNS            | DESCRIPTION                                               |
| ------------------ | --------------------------------------------------------- |
| `dict[str, float]` | deletion and insertion AUCs averaged over non-NaN samples |

| RAISES           | DESCRIPTION                                         |
| ---------------- | --------------------------------------------------- |
| `AssertionError` | if the metric has not been updated with a valid CAM |

Source code in `torchcam/metrics.py`

```python
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
```
