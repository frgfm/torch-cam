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

[`EntropyGradient`](#torchcam.methods.EntropyGradient) implements **only the map primitive** from
[Entropy-Gradient Grounding](https://arxiv.org/abs/2604.08456), section 3.2, equations (1)–(4).
It asks: *which image patches could most change how unsure the model is about its next token?*
It does not attribute a particular generated word. A large score measures local uncertainty sensitivity;
the L2 norm does not tell you whether a change would increase or decrease uncertainty.

For the full vocabulary distribution, it computes Shannon entropy in nats using `log_softmax`, differentiates
that scalar with respect to the **projected visual embeddings entering the language model**, and takes the
gradient's L2 norm over embedding channels. It reshapes those scores into the caller's explicit visual grid.
There is no head selection, attention extraction, smoothing, region selection, crop-and-refine loop or answer
accuracy evaluation. The paper's answer-accuracy gains concern its full pipeline and are not claimed here.

### Using an existing differentiable forward

Like TAM, this extractor consumes tensors and returns a single `(batch, height, width)` map. It needs no model
hooks or mandatory Transformers dependency. Unlike TAM, its input is the **language-model input embeddings**,
not final language-model states. Prepare the model-specific spatial layout and choose the next-token logits
before calling the extractor. For example, given `embeddings` used by a language-model adapter:

```python
import torch
from torchcam.methods import EntropyGradient

# embeddings: (batch, sequence, channels), after visual projection and before the language model
# image_positions: boolean (sequence,) mask, shared by this batch, selecting a 2 x 3 image grid
# These exact embeddings must require gradients BEFORE the forward.
with torch.enable_grad():
    output = language_model(inputs_embeds=embeddings, use_cache=False)
    maps = EntropyGradient((2, 3))(
        output.logits[:, -1],  # first answer distribution for an unpadded prompt
        embeddings,
        visual_tokens=image_positions,
        normalized=True,
        retain_graph=False,
    )
# maps[0] can be passed to overlay_mask as a floating-point PIL image.
```

If the model is frozen, a caller can make the prepared input embeddings a differentiable leaf with
`embeddings = embeddings.detach().requires_grad_(True)` **before** the forward. The extractor does not freeze
parameters, clear their gradients, or run another forward. `retain_graph=True` allows another attribution or
training backward on the same forward; the default releases the graph. Both forward and attribution must run
with autograd enabled, outside inference mode.

### Token layout, precision and limitations

Omit `visual_tokens` when every embedding is an image token. Otherwise supply a boolean mask in spatial order,
or unique int32/int64 indices ordered by grid row then column. Ordered indices can undo a model's token
permutation. Selection happens **after** differentiating the original tensor: slicing or copying embeddings
after the forward creates a disconnected input and raises an error. Rectangular and singleton grids are
supported, but token counts must match `height * width`. Batched samples must be independent and share the
selection and grid shape. Variable image layouts require separate calls or caller-side adaptation.

Entropy and channel norms accumulate in float32, preserving float64 inputs. Norm rescaling avoids intermediate
squared-gradient underflow/overflow. This cannot recover values already lost during low-precision model
computation. Maps are detached. `normalized=False` retains raw gradient norms; default normalization applies
min-max scaling independently per image and maps constant scores to zero. Neither form is a calibrated
probability, a causal explanation, or a guarantee of correct visual grounding.

The caller must expose the correct projected embeddings and their differentiable connection to full next-token
logits. A cached forward that omits that connection, a detached generation result, or final hidden states do
not satisfy this contract. There is no automatic Qwen2.5-VL, other VLM, multi-image, video, quantization or
attention-backend adapter in this primitive. Qwen adapter development remains with the separate DEX-AR work.
For an example, record the actual input, question, generated answer and chosen decoding prefix; time model
loading, generation, the differentiable attribution forward and this extractor separately (synchronize CUDA
at timing boundaries). A demonstration is not an answer-accuracy benchmark.

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
