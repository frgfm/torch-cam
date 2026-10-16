# Debug one prediction with an AI agent

Use `torchcam.explain.explain` when a classifier's top prediction differs from the class you expected. It runs the model, extracts normalized class activation maps for the predicted and optional expected class, and saves a machine-readable evidence bundle.

TorchCAM deliberately does not load models, checkpoints, labels, or preprocessing. Reuse those trusted pieces from the owner's repository so the explanation describes the same inference path.

Run the [prediction debugging notebook](https://github.com/frgfm/notebooks/blob/main/torch-cam/debug_prediction.ipynb) ([Colab](https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/debug_prediction.ipynb)) to compare classes, visualize overlays, and inspect a saved evidence bundle.

Model-view saving and `context` require a [source revision](installation.md) after 0.5.0. The notebook pins its own revision.

## CNN example

This complete example uses automatic CNN target-layer resolution and writes predicted-versus-expected evidence:

```python
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms.functional import center_crop, resize

from torchcam.explain import explain

image = Image.open("surprising.jpg").convert("RGB")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).eval()
class_names = weights.meta["categories"]
preprocess = weights.transforms()
input_tensor = preprocess(image).unsqueeze(0)
model_image = center_crop(
    resize(image, preprocess.resize_size, interpolation=preprocess.interpolation, antialias=preprocess.antialias),
    preprocess.crop_size,
)

result = explain(
    model,
    input_tensor,
    expected_class_idx=207,  # repository class index, for example "golden retriever"
    class_names=class_names,
)
context = {"checkpoint_id": weights.url, "sample_id": "surprising.jpg"}
result.save("torchcam-explanation", model_image, context=context)
```

The default method is [`GradCAM`][torchcam.methods.GradCAM]. If automatic target resolution is unsuitable for a custom CNN, pass the repository's feature layer, such as `target_layer="backbone.layer4"`.

## Vision Transformer example

For torchvision `vit_b_16`, prepare the model, input tensor, and display image with `ViT_B_16_Weights.DEFAULT` using the steps above. Pass [`LeGrad`][torchcam.methods.LeGrad] and explicit target blocks:

```python
from torchcam.methods import LeGrad

result = explain(
    model,
    input_tensor,
    expected_class_idx=207,
    class_names=weights.meta["categories"],
    method=LeGrad,
    target_layer=list(model.encoder.layers)[-4:],
)
result.save("torchcam-vit-explanation", model_image)
```

For another supported ViT, pass its `score_projection`, `prefix_tokens`, or `grid_shape` through `method_kwargs`. Do not rely on automatic architecture detection; see [advanced usage](advanced-usage.md#legrad-for-torchvision-vision-transformers) for compatibility details.

## Ask a coding agent

Agents that support portable [Agent Skills](https://agentskills.io/specification) can progressively load the repository's [`torchcam-debug-prediction` skill](https://github.com/frgfm/torch-cam/tree/main/.agents/skills/torchcam-debug-prediction). See [OpenAI's Skills documentation](https://developers.openai.com/codex/skills) for one supported client. Install it in the coding agents detected on your machine with `npx skills add frgfm/torch-cam`. [`llms.txt`](https://llmstxt.org/) helps agents discover the guide and API; it is not the execution contract.

Copy this prompt into an agent running inside the repository that owns the model:

> Debug the surprising prediction for `path/to/image.jpg` with TorchCAM. Reuse this repository's real model loader, trusted checkpoint, evaluation preprocessing, and class mapping. Compare the predicted class with expected class `<name or index>`, save a complete schema-v1 explanation bundle to a new directory, validate every manifest artifact, and report the evidence without causal or correctness claims. Do not replace the repository's inference path.

## Result contract

`explain` returns a frozen [`PredictionExplanation`][torchcam.explain.PredictionExplanation] with:

- detached CPU `float32` logits;
- predicted and optional expected class indices;
- normalized two-dimensional CPU `float32` CAMs in `result.cams[class_idx]`;
- method, resolved target layers, model identity, input shape, class names, and runtime versions.

A distinct expected class gets a fresh model forward. If expected and predicted indices match, TorchCAM reuses the predicted map.

`result.save(directory, image, alpha=0.5, context=None)` creates a new directory. Retain `image` after inference's resize/crop and before normalization. Its width and height must match the input tensor; matching dimensions alone do not prove matching pixels.

Supported modes are `RGB`, `1`, `L`, `I`, `F`, `I;16`, and `I;16B`. `RGB`/`L` use PNG; the rest use TIFF. Other modes fail before writing, including `I;16L`, which older Pillow versions cannot save as TIFF.

The bundle contains:

- `input.png` or `input.tiff`: lossless model-view pixels, referenced by `input_image`;
- `class-<class_idx>-layer-<layer_idx>.npy`: the finite `float32` CAM;
- `class-<class_idx>-layer-<layer_idx>-heatmap.png`: an 8-bit grayscale heatmap;
- `class-<class_idx>-layer-<layer_idx>-overlay.png`: an overlay in model-view coordinates.

Overlays scale non-8-bit backgrounds to 0–255; constant backgrounds map to black. Use the saved input and NumPy maps for numeric inspection.

`manifest.json` is written last and marks completion. Schema 1 records `prediction`, optional `expected`, `classes` (scores and artifacts), `method`, `target_layers`, `model`, `input_shape`, `versions`, `image_size` (`[width, height]`), `input_image`, and `alpha`. Older bundles may omit `input_image` and `context`; their overlay alignment is unverified.

`context` stores caller-supplied string IDs for checkpoints, preprocessing, samples, splits, or groups. Use immutable checkpoint IDs or checksums. TorchCAM does not verify these values; empty mappings are omitted.

## Boundaries and troubleshooting

This workflow supports one 2D image and a classifier that returns finite logits shaped `(1, num_classes)`. The model must already be in evaluation mode and the input tensor must use a floating-point dtype. Batched inputs, 3D/video explanations, tuple/dictionary outputs, inference mode, non-finite logits, and non-finite maps are rejected. Wrap tuple/dictionary models so their forward returns the logits tensor.

CAMs are diagnostic activation evidence. They are not causal explanations, localization scores, proofs of correctness, or compliance evidence.

For layer selection, reshape transforms, and custom ViTs, read [advanced usage](advanced-usage.md). For hook, gradient, blank-map, and NaN failures, read [troubleshooting](troubleshooting.md).
