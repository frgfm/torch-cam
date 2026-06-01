# Troubleshooting

Answers to recurring support questions. For API details, see the [package reference](../reference/methods.md).

## Stable release vs. `main`

| Source | Install | What you get |
|--------|---------|--------------|
| **PyPI (stable)** | `pip install torchcam` | Latest tagged release (see [changelog](changelog.md)). |
| **Git (`main`)** | `pip install torchcam @ git+https://github.com/frgfm/torch-cam.git` | Unreleased changes; may differ from the [online docs](https://frgfm.github.io/torch-cam), which track `main`. |

Check your installed version:

```python
import importlib.metadata
print(importlib.metadata.version("torchcam"))
```

**Examples in this site** use current APIs (`get_model` / `get_model_weights`, context managers, `enable_hooks`). Older snippets you may find online often use `pretrained=True` or private `_hooks_enabled` — those apply to releases before [v0.4.1](https://github.com/frgfm/torch-cam/releases/tag/v0.4.1).

TorchCAM **0.4.x** requires **PyTorch ≥ 2.0**. See the [installation guide](../getting-started/installation.md) for environment setup.

## `torch.no_grad()` and gradient-based CAMs

CAM **computation** always runs under `torch.no_grad()` internally. That is separate from whether the **model forward** and **backward** need autograd.

| Method family | Forward | Before `cam_extractor(...)` |
|---------------|---------|------------------------------|
| Activation-based (CAM, Score-CAM, …) | `torch.inference_mode()` or `no_grad()` is fine | No backward required |
| Gradient-based (Grad-CAM, Layer-CAM, …) | Do **not** wrap forward in `no_grad()` if gradients are required | Model must retain graph; backward runs inside the extractor |

Recommended pattern for gradient methods:

```python
from torchcam.methods import LayerCAM

model.eval()
with LayerCAM(model) as cam_extractor:
    out = model(input_tensor)  # autograd enabled
    cams = cam_extractor(out.squeeze(0).argmax().item(), out)
```

If you only need inference until CAM time, disable hook recording during bulk forward passes (see [Hook lifecycle](#hook-lifecycle)), then run a single forward with hooks enabled before calling the extractor.

!!! warning "Empty or flat gradients"
    Wrapping `model(input_tensor)` in `torch.no_grad()` or `torch.inference_mode()` prevents gradient-based methods from backpropagating. You will get incorrect or zero maps.

## Hook lifecycle

Extractors register forward hooks on `target_layer` (and backward hooks for gradient methods). Manage them explicitly or with a context manager.

### Context manager (recommended)

```python
with LayerCAM(model) as cam_extractor:
    out = model(input_tensor)
    cams = cam_extractor(class_idx, out)
# Hooks removed; internal state reset
```

Leaving hooks attached after use can interfere with training, tracing, or a second CAM extractor on the same model.

### Enable / disable without removing hooks

Since v0.4.1, use the public API instead of `_hooks_enabled`:

```python
cam_extractor = LayerCAM(model, enable_hooks=False)
with torch.inference_mode():
    out = model(input_tensor)  # hooks inactive: less RAM / latency

cam_extractor.enable_hooks()
out = model(input_tensor)
cams = cam_extractor(class_idx, out)
cam_extractor.disable_hooks()
```

### Reset state between samples

- `reset_hooks()` — clears cached activations and gradients (hooks stay registered).
- `remove_hooks()` — detaches all handles; call `enable_hooks()` only affects recording, not registration.

### Common error: forward before CAM

```text
AssertionError: Inputs need to be forwarded in the model for the conv features to be hooked
```

Run at least one forward through the hooked model **after** creating the extractor and **before** `cam_extractor(class_idx, ...)`. The forward must hit the `target_layer` (same path as your CAM use case).

## Choosing `target_layer`

If `target_layer` is omitted, TorchCAM picks the last convolutional layer whose output still has spatial extent, using a dummy forward with `input_shape`:

```python
LayerCAM(model, input_shape=(3, 224, 224))  # default for ImageNet-style 2D
```

Override when auto-selection is wrong (multi-branch nets, custom heads, video/volume models):

```python
# By submodule name (from model.named_modules())
LayerCAM(model, target_layer="layer4")

# By module reference
LayerCAM(model, target_layer=model.layer4)

# Multiple layers (where the method supports it)
LayerCAM(model, target_layer=["layer3", "layer4"])
```

Tips:

- Pick a **conv** (or equivalent) layer **before** global pooling / flattening.
- Pass the correct `input_shape` (channels + spatial dims, no batch) so auto-resolution matches your tensor layout.
- **CAM** (activation-only) needs a compatible fully-connected / classifier wiring; see method docs if you use a non-classic head.
- A warning log means auto-resolution ran: `no value was provided for target_layer, thus set to '...'`.

## 3D and higher-dimensional inputs

Volumetric (or spatio-temporal) models are supported when activations keep spatial dimensions. Set `input_shape` to your tensor layout **without** batch, e.g. `(1, 32, 32, 32)` for a single-channel 3D volume.

```python
input_shape = (1, 32, 32, 32)  # (C, D, H, W)
with LayerCAM(model, input_shape=input_shape) as cam_extractor:
    out = model(volume_tensor)
    cams = cam_extractor(class_idx, out)
```

Upsampling and fusion use **trilinear** interpolation when CAM tensors are 4D `(N, D, H, W)`; 2D maps use bilinear. Overlay utilities expect image-style inputs — resize or slice volumes yourself for visualization.

Added in [v0.2.0](https://github.com/frgfm/torch-cam/releases/tag/v0.2.0); see the [release note](https://github.com/frgfm/torch-cam/releases/tag/v0.2.0) for examples.

## Still stuck?

- Compare your code with the [quick start](../index.md#quick-start) and [notebooks](../getting-started/notebooks.md).
- Search [existing issues](https://github.com/frgfm/torch-cam/issues); include TorchCAM version, PyTorch version, method name, and `target_layer` choice when opening a new one.
