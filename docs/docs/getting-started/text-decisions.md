# Text decision models

`TokenAblation` measures single-token replacement effects without gradients. `TokenIG` explains differentiable
classifiers with [Integrated Gradients](https://arxiv.org/abs/1703.01365) on word embeddings. Neither adds a tokenizer,
Transformers or Laya dependency. Until the next release,
[install from source](installation.md).

--8<-- "README.md:quickstart-text"

## Inputs and results

Both extractors forward `inputs` as keyword arguments and accept `output_transform` for structured model scores.
`TokenAblation` returns original `(N, classes)` scores and signed `(N, classes, tokens)` effects. Each effect is the
original score minus the score after replacing that token's ID; attention and other context stay fixed. Choose an
appropriate `replacement_id`, such as the model's mask token. All classes share one forward per selected position.
Interactions mean these marginal effects need not sum to a total score change; there is no completeness diagnostic.

For `TokenIG`, supply the word-embedding module before positional/contextual
mixing, directly or by name. It must run once per forward and return `(N, tokens, channels)`. Use `output_transform`
to select `(N, classes)` scores from structured outputs, e.g. `lambda output: output.logits` or `output[0]`.
`class_idx=2` selects one output column; `[2, 0]` preserves that order; omitting it explains every column.

[`TokenAttribution`](../reference/methods.md#torchcam.methods.TokenAttribution) contains detached original/baseline
scores, signed `(N, selected_classes, tokens)` contributions, class indices, completeness error and integration steps.
Maps explain pre-softmax logits; display probabilities separately using the model's calibration. Explaining softmax
probabilities couples competing classes and can yield mirrored maps. Different classes can legitimately share tokens.

For CAM-style highlights, keep positive values and normalize each row for display:

```python
support = result.attributions.clamp_min(0)
heatmaps = support / support.amax(-1, keepdim=True).clamp_min(1e-12)
```

Retain signed values for numerical checks and comparisons. Independently normalized colors do not compare magnitudes
across classes. These are baseline-relative score contributions, not token probabilities or proof of bias; use
controlled word deletions/name substitutions to investigate suspected proxies.

`token_mask` selects the `(N, tokens)` positions to vary; excluded tokens stay fixed and receive zero attribution.
Exclude padding, special tokens, questions and decision options as appropriate. A padding mask alone also attributes
attended special tokens. For `TokenIG`, `baseline` defaults to zero and may match embedding shape, device and dtype,
including padding embeddings. Selected tokens remain attended; attention/position metadata never change. The caller
owns tokenization, offsets, truncation checks and model-specific input construction.

## Laya integration

Use an already loaded `laya.Agent(..., backend="eager")`, its exact native model batch, and a boolean mask selecting
only the state/ticket tokens. High-level prediction APIs disable gradients; call the underlying module:

```python
extractor = TokenIG(
    agent.model,
    agent.model.encoder.get_input_embeddings(),
    output_transform=lambda output: output[0],
)
result = extractor(batch, token_mask=state_mask)
```

For interactive explanations without backward passes, use
`TokenAblation(agent.model, lambda output: output[0])(batch, agent.tok.mask_token_id, token_mask=state_mask)`.
This also avoids the steep zero-embedding integration paths that can make pretrained decision models slow to explain.

Forward `input_ids`, `attention_mask`, `marker_pos`, `marker_mask`, `qtype` and any `position_ids`/`option_ids` unchanged,
including attention-mask dictionaries for parallel layouts. Use Laya's sequence builder to identify state positions,
check truncated state/options, and select valid output columns when options are padded. Apply the same option-order
permutation to labels, scores and maps. Display probabilities from Laya's native decoder, including its temperature
and language calibration.

Laya [0.4.6 architecture](https://github.com/NandhaKishorM/laya/tree/c73908d96ff73c0062c340a11bf7c54ec7d8bc4d) was checked
with native ModernBERT, random weights and a synthetic tokenizer: choice, score and noul questions, sequential and
parallel layouts. This establishes architecture compatibility, not learned relevance. Local adapters need an executable
model; `TokenIG` additionally needs gradients. `TokenAblation` can use a remote scoring wrapper if it supports the same
controlled token replacements.

## Numerical checks and cost

Gauss-Legendre quadrature follows the straight embedding path with `alpha = u**3` and its Jacobian, placing more nodes
near the baseline. Node counts double until signed completeness and successive-map stability meet the tolerances;
at least two passes run. See [`TokenIG`](../reference/methods.md#torchcam.methods.TokenIG) for budgets and defaults.
A failed check raises. Passing checks cannot guarantee per-token accuracy: narrow activations can be missed and
opposing integration errors can cancel. Increase the initial node count and compare baselines/controlled edits.

Each node shares one forward across selected classes, with one gradient pass per class. Selecting fewer classes
reduces work; independent samples can be batched. Maps accumulate without storing every node's graph or a class-by-
channel tensor. Difficult paths can cost substantially more than prediction.

Use eager execution, preferably float32/float64, and non-reentrant checkpointing. Outer `torch.no_grad()` is supported;
inference mode and no-grad/weight-mutating forwards (`max_norm`) are unsupported. Accumulation uses float64 (float32 on
MPS), without recovering model precision. Hooks, submodule training modes, parameter values and `.grad` buffers are preserved.
