# Evaluation metrics

Apart from qualitative visual comparison, it is important to have a refined evaluation metric for class activation maps. This submodule is dedicated to the evaluation of CAM methods.

Run the [performance benchmark notebook](https://github.com/frgfm/notebooks/blob/main/torch-cam/performance_benchmark.ipynb) ([Colab](https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/performance_benchmark.ipynb)) to evaluate confidence scores and deletion/insertion AUCs, compare explicit baselines, and record runtime metadata.

Both metrics' `update()` methods compute CAMs from the model's raw output. A `logits_fn` such as softmax changes only the scores used to measure the explanation. It does not change the score differentiated by a gradient-based CAM. Class indices and callable `targets` select the same outputs before and after this score transform, so `logits_fn` must preserve that output layout.

The score transform receives separate tensor copies with no autograd graph. This also applies to tensor leaves in dictionaries, lists, and tuples, including named tuples. Thus, an in-place score transform does not change the raw outputs used to build the CAM. Use `output_fn` to convert custom output objects into these supported containers before score transformation.

Use `output_fn` when the model output needs an adapter. The adapter runs before both CAM extraction and metric scoring, including extra model calls inside methods such as ScoreCAM and SmoothGradCAMpp. It does not change the model output outside the metric update. For example, a model that returns a dictionary of batched logits can use:

```python
from functools import partial
from operator import itemgetter

import torch

from torchcam.metrics import ClassificationMetric

metric = ClassificationMetric(
    cam_extractor,
    logits_fn=partial(torch.softmax, dim=-1),
    output_fn=itemgetter("logits"),
)
metric.update(input_tensor)
```

If you previously used `logits_fn` to extract or reshape a model output, move that adapter to `output_fn`. Keep the probability transform in `logits_fn`. Native per-sample list outputs still work with callable `targets` and do not require an adapter. The target must accept the same per-sample structure from both the raw and transformed outputs.

Preserve the class identity and order expected by the CAM extractor when adapting outputs. For example, CAM selects classifier weight rows directly. Extracting `output["logits"]` preserves those class indices; an arbitrary class permutation is not supported across all CAM methods.

## Classification confidence

![Average Drop and Increase in Confidence compare the selected-class score on the original and CAM-masked inputs.](../img/classification-metrics.svg)

Average Drop uses the exact relative loss for each positive original confidence. It does not add an epsilon to the denominator, which would suppress drops at small confidence values. When the original confidence is zero, its drop is defined as zero. A larger masked confidence still counts as an increase.

Selected original and masked scores must be finite and nonnegative. Positive raw scores remain supported, but use probabilities to compare with published confidence metrics. Negative scores raise `ValueError`; Average Drop has no confidence-loss interpretation for a negative denominator. Use deletion/insertion for signed scalar targets.

Both metrics reject nonfinite selected scores with `ValueError` and leave their accumulated results unchanged. CAMs that contain NaNs are still skipped and counted by `nan_count`.

::: torchcam.metrics.ClassificationMetric
    options:
        members:
            - reset
            - update
            - summary

## Deletion and insertion faithfulness

[Deletion and insertion](https://arxiv.org/abs/1806.07421) measure how the model's selected-class score changes as spatial positions are perturbed in descending CAM order. For an input $X$, baseline $B$, and the set $R_t$ containing the top-ranked positions restored or removed by step $t$:

$$
D_t[p] =
\begin{cases}
B[p] & p \in R_t \\
X[p] & p \notin R_t
\end{cases}
$$

$$
I_t[p] =
\begin{cases}
X[p] & p \in R_t \\
B[p] & p \notin R_t.
\end{cases}
$$

The same spatial mask is applied to every input channel. If $x_t = |R_t| / P$ is the actual perturbed fraction for $P$ spatial positions and $s_c$ is the selected-class score, TorchCAM computes:

$$
\operatorname{DeletionAUC} = \operatorname{trapz}(s_c(D_t), x_t),
$$

$$
\operatorname{InsertionAUC} = \operatorname{trapz}(s_c(I_t), x_t).
$$

Lower deletion AUC and higher insertion AUC indicate a more faithful ranking. Both the unperturbed and fully perturbed endpoints are included. `steps` is the maximum number of intervals: each interval changes $\lceil P / \text{steps} \rceil$ positions, except for the shorter final interval, and integration uses the resulting fractions rather than an assumed uniform grid.

The default baseline is `zeros_like(input_tensor)`. This represents the dataset mean only when inputs were normalized so that the mean maps to zero. Baseline choice can introduce out-of-distribution evidence and materially change both scores. The original RISE evaluation used constant deletion values and a blurred insertion substrate, while this metric deliberately uses one baseline for both curves. To reproduce those two substrates, run the metric separately with each baseline and compare only the corresponding AUC.

`batch_size` limits how many perturbed inputs are scored in one forward pass. It bounds temporary memory but does not reduce the number of perturbed samples. With $S$ effective intervals, each valid input requires $2S - 1$ additional scoring samples, plus the original CAM-producing forward and any backward pass required by the extractor.

By default, the metric integrates raw model outputs. Pass a function such as softmax for probability curves comparable to the paper; raw-logit AUCs may fall outside $[0, 1]$ and should not be compared with probability AUCs.

Finite signed scalar scores are supported. Half-precision and bfloat16 scores are promoted to float32 before integration, so adjacent finite values do not overflow during half-precision addition.

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

!!! warning

    Deletion and insertion test perturbation faithfulness to the model's score. They do not establish localization quality, human interpretability, or causal correctness outside the chosen perturbation and baseline protocol.

### Evaluate an existing CAM

Use `evaluate()` to score an already-computed map without calling the extractor or fusing maps again. Keep the extractor open, as with `update()`; calling either method after its context closes raises `RuntimeError`. Evaluation reuses the metric's score transforms, baseline, steps, and scoring batch size. Supply the same class index or callable target used to compute the map. If you omit both, evaluation selects each input's original top prediction once and keeps that target throughout both curves. TorchCAM cannot infer the target from the map.

```python
# cam is the exact map to evaluate, shaped (N, H, W) for 2D inputs.
with GradCAM(model, "layer4") as cam_extractor:
    metric = DeletionInsertionMetric(cam_extractor, partial(torch.softmax, dim=-1))
    result = metric.evaluate(input_tensor, cam, class_idx=class_indices)
fractions = result["fractions"]       # (K,), includes 0 and 1
deletion = result["deletion"]         # (N, K), selected scores
insertion = result["insertion"]       # (N, K), selected scores
deletion_auc = result["deletion_auc"] # (N,)
insertion_auc = result["insertion_auc"] # (N,)
valid = result["valid"]               # (N,), boolean
```

The map must be a floating-point tensor shaped `(N, *map_spatial_dims)`, without a channel dimension. Its batch size and spatial rank must match the nonempty input shaped `(N, C, *spatial_dims)`. One, two, and three spatial dimensions are supported. Maps are detached, moved to the input device, and resized with linear, bilinear, or trilinear interpolation as needed. Half-precision and bfloat16 maps are promoted to float32 before resizing to avoid low-precision overflow. Nonfinite resized values raise an error. Values do not need to lie in `[0, 1]`. Ties retain flattened spatial order.

**Signed maps retain their signs.** Ranking uses the supplied values from highest to lowest after resizing. No normalization, absolute value, or clamping is applied. Positive values precede zero and negative values; the most negative values are perturbed last. This evaluates a descending signed ranking, not the strength of negative evidence. For a signed flip-difference map, neither negation nor conversion to a magnitude map happens implicitly. If you choose such a conversion, apply it explicitly before evaluation and record it as a different evaluation protocol.

Each row in the result keeps its original input index. A map containing any NaN is skipped: `valid` is false and its curves and AUCs are filled with NaNs. An all-skipped batch returns the same shapes and fraction grid. `evaluate()` rejects infinite CAM values with an error naming the affected sample indices, even if those maps also contain NaNs. Invalid map shapes or types also raise an error. `update()` keeps its existing NaN-only filtering for extracted maps. Selected original and perturbed scores must remain finite, including original scores for skipped rows.

Keep score transforms and callable target scores on the input device. All returned tensors are detached and stay on that device. Evaluation does not retain results or change the accumulated `summary()`, `total`, or `nan_count`; use `update()` to accumulate extracted-CAM metrics as before. A baseline callable receives the full original batch once before valid samples are selected, and is not called for an all-skipped batch. Evaluation temporarily uses model evaluation mode and disables extractor hooks for every forward, including the original score. Module training modes, hook state, and the output adapter are restored on success and on errors. This path requires one original scoring forward plus the perturbation scoring samples described above, with no CAM extraction or backward pass.

::: torchcam.metrics.DeletionInsertionMetric
    options:
        members:
            - reset
            - evaluate
            - update
            - summary
