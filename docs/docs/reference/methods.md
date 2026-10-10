# Interpretability methods

## Class activation map

The class activation map gives you the importance of each region of a feature map on a model's output.
More specifically, a class activation map is relative to:

* the layer at which it is computed (e.g. the N-th layer of your model)
* the model's classification output (e.g. the raw logits of the model)
* the class index to focus on

With TorchCAM, the target layer is selected when you create your CAM extractor. You will need to pass the model logits to the extractor and a class index for it to do its magic!

## Activation-based methods

Methods related to activation-based class activation maps.

::: torchcam.methods
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - CAM
            - ScoreCAM
            - SSCAM
            - ISCAM


## Gradient-based methods

Methods related to gradient-based class activation maps.

::: torchcam.methods
    options:
        heading_level: 3
        show_root_heading: false
        show_root_toc_entry: false
        members:
            - FinerCAM
            - GradCAM
            - GradCAMpp
            - SmoothGradCAMpp
            - XGradCAM
            - LayerCAM
            - LeGrad
            - RefineCAM

## Entropy-gradient maps for VLMs

[`EntropyGradient`](#torchcam.methods.EntropyGradient) implements the [paper's](https://arxiv.org/abs/2604.08456)
map primitive: next-token Shannon entropy, its gradient with respect to projected visual input embeddings,
then the channel L2 norm. Bright patches indicate sensitivity of uncertainty, not support for a generated word.
Region selection and iterative cropping are excluded; the full pipeline's answer-accuracy gains are not claimed.

```python
from torchcam.methods import EntropyGradient

# embeddings: (batch, tokens, channels), requiring gradients BEFORE this forward
output = language_model(inputs_embeds=embeddings, use_cache=False)
maps = EntropyGradient((height, width))(
    output.logits[:, -1], embeddings, visual_tokens=image_positions,
)  # (batch, height, width), detached and normalized per image
```

Supply the exact embeddings entering the language model, not final hidden states or post-forward copies/slices.
`visual_tokens` is an optional boolean mask or ordered integer indices; omit it for an all-visual input.
Samples must be independent and share a row-major grid and selection. Both forward and attribution need autograd.
Checkpointed forwards require `use_reentrant=False`.
`retain_graph=True` permits graph reuse; `normalized=False` retains raw norms. Parameter gradients are preserved.
Entropy and norms accumulate in float32/float64; they cannot recover gradients lost inside a low-precision model.
The caller owns model extraction and spatial layout; no automatic VLM adapter or Transformers dependency is added.

::: torchcam.methods.EntropyGradient
    options:
        heading_level: 3
        show_root_heading: true
        members: [__call__]

## Token activation maps

Visual explanations for selected vocabulary tokens in multimodal language models. TAM consumes final language-model
states and a linear vocabulary head directly; it does not use the class CAM hook API.

::: torchcam.methods.TAM
    options:
        heading_level: 3
        show_root_heading: true
        members: [__call__]

::: torchcam.methods.DEXAR
    options:
        heading_level: 3
        show_root_heading: true
        members: [__call__, aggregate]
