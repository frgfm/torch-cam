---
name: torchcam-debug-prediction
description: Debug one surprising 2D image-classification prediction in an existing PyTorch repository with TorchCAM. Use when an owner asks for predicted-versus-expected CAM evidence, a saved explanation bundle, GradCAM on a CNN, or LeGrad on a supported Vision Transformer. Reuse the repository's trusted model loader, checkpoint, class mapping, and preprocessing instead of inventing replacements.
compatibility: Requires local Python execution, PyTorch, Pillow, and torchcam>=0.5.0.
metadata:
  author: frgfm
  version: "1.0"
---

# Debug one prediction with TorchCAM

Produce reproducible visual evidence for one surprising classifier prediction. CAMs show class-associated activation, not why the model decided, causal influence, correctness, or localization quality.

## 1. Discover the repository's inference path

Find and reuse the code that already defines:

- the model architecture and trusted checkpoint loader;
- evaluation preprocessing, including resize, crop, normalization, and color conversion;
- the ordered class-name mapping;
- the original image before normalization.

Search the repository before writing code. Prefer its test or inference entry point over reconstructing the model. Load only checkpoints already trusted by the owner; TorchCAM does not load checkpoints or preprocessing.

Confirm the model returns one logits tensor shaped `(1, num_classes)`. If it returns a tuple or dictionary, wrap it in a small `nn.Module` that selects the logits tensor without changing inference.

## 2. Choose the extractor

- CNN: start with the default `GradCAM`. Let TorchCAM resolve the last spatial layer. If resolution fails or the repository has a known semantic feature layer, pass that exact module or name as `target_layer`.
- Torchvision Vision Transformer: use `LeGrad` and explicit transformer blocks, usually the last four: `list(model.encoder.layers)[-4:]`.
- Other ViTs: use `LeGrad` only when the blocks expose supported batch-first `nn.MultiheadAttention`. Pass the repository-specific `score_projection`, `prefix_tokens`, or `grid_shape` through `method_kwargs` when required.
- Existing reshape-based CAM setup: preserve its explicit target layer and pass `reshape_transform` through `method_kwargs`.

Do not add an architecture registry or guess a ViT configuration.

## 3. Run one explanation

Call `model.eval()`. Keep the call outside `torch.inference_mode()` because CAM extraction needs gradients. The input must be a batch of one shaped `(1, C, H, W)`.

```python
from torchcam.explain import explain

result = explain(
    model,
    input_tensor,
    expected_class_idx=expected_idx,
    class_names=class_names,
    target_layer=target_layer,  # omit for automatic CNN resolution
)
bundle = result.save("torchcam-explanation", image=original_image, alpha=0.5)
```

For a ViT, also pass `method=LeGrad` and the explicit blocks. Use a new output directory; `save` refuses to overwrite an existing one.

## 4. Validate the evidence bundle

Treat `manifest.json` as the completion marker. Before reporting success:

1. Parse it and require `schema_version == 1`.
2. Read artifacts from `manifest["classes"][str(class_idx)]["artifacts"]`. Resolve every relative `map`, `heatmap`, and `overlay` path under the bundle directory and require each file to exist. Prediction and expected entries contain class references; logits and probabilities live in the corresponding class entry.
3. Load each `.npy` map with `allow_pickle=False`; require a finite two-dimensional `float32` array.
4. Open every overlay and require its dimensions to match `manifest.json`'s `image_size`.
5. Confirm the prediction and optional expected class indices match the repository's class ordering.

No manifest means the bundle is incomplete, even if some images exist.

## 5. Report to the owner

Return:

- predicted class and the owner's expected class, with indices and model scores/probabilities from the manifest;
- the resolved method and target layers;
- the bundle path and the most useful overlay paths;
- any compatibility boundary or uncertainty encountered.

Use language such as “the GradCAM map highlights…” or “activation differs between…”. Do not claim the map proves causation, model correctness, object localization, bias, safety, or compliance.

Check each map's range before describing it. If it is constant or all zero, call it a blank map and do not claim it highlights a region or shows positive class-associated activation; a blank CAM is not proof that a feature is absent.

If extraction fails, use the TorchCAM prediction-debugging guide and troubleshooting page before changing repository model code.
