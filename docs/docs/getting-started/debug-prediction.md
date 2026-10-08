# Debug one prediction with an AI agent

Use `torchcam.explain.explain` when a classifier's top prediction differs from the class you expected. It runs the model, extracts normalized class activation maps for the predicted and optional expected class, and saves a machine-readable evidence bundle.

TorchCAM deliberately does not load models, checkpoints, labels, or preprocessing. Reuse those trusted pieces from the owner's repository so the explanation describes the same inference path.

Run the [prediction debugging notebook](https://github.com/frgfm/notebooks/blob/main/torch-cam/debug_prediction.ipynb) ([Colab](https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/debug_prediction.ipynb)) to compare classes, visualize overlays, and inspect a saved evidence bundle.

The model-view size check, saved input image, and optional `context` below require a development revision after 0.5.0. Install the [latest source](installation.md) or use a tested pinned revision containing these changes. The existing prediction notebook pins its own implementation.

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
# Keep the exact spatial transforms without normalizing or quantizing the display image.
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
result.save(
    "torchcam-explanation",
    model_image,
    alpha=0.5,
    context={
        "checkpoint_id": weights.url,
        "preprocessing_id": str(preprocess),
        "sample_id": "surprising.jpg",
    },
)
```

The default method is [`GradCAM`][torchcam.methods.GradCAM]. If automatic target resolution is unsuitable for a custom CNN, pass the repository's feature layer, such as `target_layer="backbone.layer4"`.

## Vision Transformer example

ViTs require an explicit method and target blocks. Torchvision `VisionTransformer` is supported directly by [`LeGrad`][torchcam.methods.LeGrad]:

```python
from PIL import Image
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.transforms.functional import center_crop, resize

from torchcam.explain import explain
from torchcam.methods import LeGrad

image = Image.open("surprising.jpg").convert("RGB")
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights).eval()
preprocess = weights.transforms()
input_tensor = preprocess(image).unsqueeze(0)
model_image = center_crop(
    resize(image, preprocess.resize_size, interpolation=preprocess.interpolation, antialias=preprocess.antialias),
    preprocess.crop_size,
)

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

`result.save(directory, image, alpha=0.5, context=None)` creates a new directory and never overwrites one. Pass the **model-view image** after the same spatial transforms used for inference, before normalization. Its width and height must match `input_tensor`. Resizing a cropped CAM onto the untouched original image can place cues in the wrong region. TorchCAM checks dimensions; the caller must ensure pixel correspondence even when two images have the same size.

For custom preprocessing, retain a display copy after resize/crop and before normalization. Do not guess an inverse transform. The examples above apply the same native spatial transforms specified by the torchvision weights. They avoid floating-point normalization round trips when retaining the display pixels.

It saves the model-view image as `input.png` for 8-bit RGB/grayscale images and as `input.tiff` for other supported modes, preserving floating-point and signed integer pixels. Read the filename from `input_image` in the manifest. It then writes the following files for each class and returned map:

- `class-<class_idx>-layer-<layer_idx>.npy`: the finite `float32` CAM;
- `class-<class_idx>-layer-<layer_idx>-heatmap.png`: an 8-bit grayscale heatmap;
- `class-<class_idx>-layer-<layer_idx>-overlay.png`: an overlay matching the model-view image.

It writes `manifest.json` last. Its presence marks a complete bundle. Schema version 1 contains `prediction`, optional `expected`, per-class logits/probabilities and relative artifact paths, the contributing `target_layers` for each map, `method`, all resolved target layers, `model`, `input_shape`, `versions`, `[width, height]` `image_size`, `input_image`, and `alpha`. A directory with artifacts but no manifest is incomplete. Earlier schema-v1 bundles may omit `input_image` and `context`.

Optional `context` is a mapping of string keys to string values, copied into the manifest without changing schema version 1. Use it to record the owner's checkpoint identity, preprocessing recipe, sample identity, split, or group. For local checkpoints, prefer an immutable revision or checksum over a path that can be overwritten. These identifiers are caller-supplied records; TorchCAM does not load or verify their contents. Omit `context` to retain the existing manifest fields.

When upgrading from 0.5.0, replace original-image overlays with model-view overlays. Old bundles remain readable, but an overlay alone cannot establish whether its source image matched the model's crop.

## Boundaries and troubleshooting

This workflow supports one 2D image and a classifier that returns finite logits shaped `(1, num_classes)`. The model must already be in evaluation mode and the input tensor must use a floating-point dtype. Batched inputs, 3D/video explanations, tuple/dictionary outputs, inference mode, non-finite logits, and non-finite maps are rejected. Wrap tuple/dictionary models so their forward returns the logits tensor.

CAMs are diagnostic activation evidence. They are not causal explanations, localization scores, proofs of correctness, or compliance evidence.

For layer selection, reshape transforms, and custom ViTs, read [advanced usage](advanced-usage.md). For hook, gradient, blank-map, and NaN failures, read [troubleshooting](troubleshooting.md).
