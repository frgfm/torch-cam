# Text decision models

`TokenIG` attributes a classifier's decisions to its input tokens using
[Integrated Gradients](https://arxiv.org/abs/1703.01365) on word embeddings. It works with differentiable PyTorch
classifiers, including transformer decision models whose heads have no built-in explainability.
The library needs no tokenizer, Transformers, or Laya dependency. Available in the development version;
[install from source](installation.md) until the next release.

## Explain each decision

--8<-- "README.md:quickstart-text"

The extractor forwards `inputs` unchanged as keyword arguments. Supply the word-embedding module, before positional
embeddings and contextual mixing, either directly or by module name. It must run exactly once per forward and return
`(batch, tokens, channels)`. The model must return `(batch, classes)` floating-point scores; use `output_transform`
to select logits from a structured output. For a tuple output, use `lambda output: output[0]`.

By default every output column gets a map. `class_idx=2` selects one column; `[2, 0]` preserves that order.
Keep class labels and tokenizer offsets aligned with the model's actual output and input order.

## Scores, contributions and a simple heatmap

Maps target pre-softmax class logits, following the class-score objective used by image CAMs. This method is IG,
rather than Grad-CAM. Display the model's decision probabilities separately; use its own temperature, sigmoid,
or calibration when applicable. Explaining softmax probabilities instead couples all competing classes and can
produce mirrored maps. Different classes can still legitimately share influential tokens.

The [`TokenAttribution`](../reference/methods.md#torchcam.methods.TokenAttribution) result contains detached tensors
on the embedding device:

| Field | Shape / meaning |
| --- | --- |
| `scores` | `(N, classes)`: original uncalibrated model scores |
| `baseline_scores` | `(N, classes)`: scores at the embedding reference |
| `attributions` | `(N, K, tokens)`: signed contributions for selected classes |
| `class_indices` | Tuple of the `K` output columns in map order |
| `completeness_error` | `(N, K)`: signed contribution sum minus selected score change |
| `steps` | Nodes used by the final integration pass; earlier refinement passes cost additional work |

For a CAM-style view, show the original input once and one row per decision, with its probability and positive-token
highlights. Normalize each row for display only:

```python
support = result.attributions.clamp_min(0)
heatmaps = support / support.amax(-1, keepdim=True).clamp_min(1e-12)
```

Retain the signed raw values for comparisons and completeness checks. Color intensities normalized separately per
class cannot compare contribution magnitudes across classes. Contributions are changes in score relative to the
baseline, not token probabilities or a proof of bias. Controlled name substitutions or word deletions can test
suspected proxies; keep the task and options fixed.

## Select the input and reference

`token_mask` is a boolean `(N, tokens)` tensor. Only these embeddings vary along the integration path; all others
stay at their original values and receive zero attribution. Attention masks and position metadata never change.
Use it to hold questions, decision options and special tokens fixed, and exclude padding. A padding mask alone,
as in the quick start, also attributes any attended special tokens.

The default reference is zero at selected word embeddings. To use padding embeddings instead:

```python
import torch

embedding = model.get_input_embeddings()
with torch.no_grad():
    baseline = embedding(torch.full_like(inputs["input_ids"], tokenizer.pad_token_id))
result = extractor(inputs, token_mask=inputs["attention_mask"].bool(), baseline=baseline)
```

The reference must match embedding shape, device and dtype. It is an embedding intervention; selected tokens remain
attended. Tokenization, offsets, truncation checks and model-specific input construction belong to the caller.

## Laya integration

Laya's underlying decision module exposes differentiable per-option logits even though its high-level prediction
API disables gradients. Use eager mode and `agent.model`, select the first tensor from its output, and mask only the
state/ticket tokens:

```python
# agent: an already loaded laya.Agent(..., backend="eager")
# batch: the exact model inputs built by Laya for the ticket and routing question
# state_mask: boolean (N, tokens), true only at ticket tokens in that encoded sequence
extractor = TokenIG(
    agent.model,
    agent.model.encoder.get_input_embeddings(),
    output_transform=lambda output: output[0],
)
result = extractor(batch, token_mask=state_mask)
```

Forward the complete native batch, including `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype`,
and, when present, `position_ids` and `option_ids`. Preserve Laya's attention-mask dictionaries for parallel layouts.
Laya's sequence builder defines the state positions; do not assume the entire sequence is ticket text. Reject or
explicitly show truncated state and truncated options. For padded option columns, select only valid `class_idx` values.
Apply the same option-order permutation to labels, scores and maps. Obtain displayed probabilities from Laya's
normal decoder, including its temperature and language calibration, rather than using an uncalibrated softmax.

Compatibility was checked on Laya [0.4.6 source](https://github.com/NandhaKishorM/laya/tree/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d)
with native ModernBERT and the decision head, random weights and a synthetic tokenizer: choice, score and noul
questions, sequential and parallel option layouts. These checks establish architecture compatibility, not learned
token relevance on a trained checkpoint. Architecture knowledge alone cannot produce explanations: actual maps
need executable model weights and gradient access. A labels/probabilities-only remote API needs a perturbation
method instead.

## Cost and numerical checks

One forward per quadrature node is shared by all selected classes; each class adds an embedding-gradient pass.
Token contributions accumulate directly into `(N, K, tokens)` maps, without storing every node's graph or a
class-by-channel attribution tensor. Selecting fewer classes reduces backward work. Independent input samples
can be batched; models that mix samples are unsupported.

Gauss-Legendre quadrature integrates the straight embedding path with `alpha = u**3` and its Jacobian, concentrating
nodes near the baseline. Starting at `steps=32`, the node count doubles until every selected map's signed completeness
error is at most `tolerance=1e-4` and successive maps change by at most `stability_tolerance=0.01` in relative L1 norm
(with the denominator floored at `tolerance`). At least two passes run. The default limit is `max_steps=4096` nodes
per pass. If checks fail, the extractor raises rather than returning an unverified map. Increase the limit or review
the baseline and precision; difficult paths can cost substantially more than a prediction.

Use eager model execution, preferably float32 or float64, and disable reentrant gradient checkpointing. An outer
`torch.no_grad()` is supported; `torch.inference_mode()` and no-grad model forwards are not. Accumulation uses
float64 (float32 on MPS), but cannot recover gradients lost inside a low-precision model. Temporary hooks and
all submodule training modes are restored even on errors; parameter values and `.grad` buffers are preserved.
