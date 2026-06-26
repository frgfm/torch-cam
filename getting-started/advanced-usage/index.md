# Advanced usage

The [quick start](../../) uses a torchvision classifier, but TorchCAM works with any PyTorch model. This guide covers the questions that come up most often once you move past the basic example. Hitting an error rather than a usage question? Jump to [Troubleshooting](../troubleshooting/).

## Use your own model

TorchCAM works with **any** `nn.Module` whose forward returns class scores (logits) of shape `(N, num_classes)` — it is not limited to torchvision. You only need to tell the extractor which layer to read the activations from.

List the candidate layers by name:

```python
for name, module in model.named_modules():
    print(name, "->", type(module).__name__)
```

Then pass the name **or** the module itself as `target_layer`:

```python
from torchcam.methods import SmoothGradCAMpp

cam_extractor = SmoothGradCAMpp(model, target_layer="features.7")        # by name
# equivalently
cam_extractor = SmoothGradCAMpp(model, target_layer=model.features[7])   # by module
```

If you omit `target_layer`, TorchCAM runs a dummy forward of shape `(1, *input_shape)` (default `(3, 224, 224)`), picks the last layer whose output still has spatial dimensions, and logs the choice. If your model expects a different input, set `input_shape` accordingly — otherwise the dummy forward will fail or pick the wrong layer:

```python
cam_extractor = LayerCAM(model, input_shape=(3, 384, 384))
```

## Choosing the target layer

A CAM is computed on the activation map of a **convolutional (spatial)** layer. The default — the last convolutional layer before global pooling — is the most class-discriminative but also the coarsest. Earlier layers give finer, less semantic maps. Rules of thumb for common torchvision backbones:

| Architecture     | Typical `target_layer` | `fc_layer` for `CAM`                |
| ---------------- | ---------------------- | ----------------------------------- |
| ResNet / ResNeXt | `"layer4"`             | `"fc"`                              |
| DenseNet         | `"features"`           | `"classifier"`                      |
| MobileNet v2     | `"features"`           | `"classifier.1"`                    |
| EfficientNet     | `"features"`           | `"classifier.1"`                    |
| MobileNet v3     | `"features"`           | *two `Linear` layers — `CAM` n/a*   |
| VGG              | `"features"`           | *three `Linear` layers — `CAM` n/a* |
| SqueezeNet       | `"features"`           | *no `Linear` head — `CAM` n/a*      |

When does the base `CAM` work?

`CAM` needs **exactly one** `nn.Linear` classification head fed by global pooling, and resolves it automatically. It therefore works for ResNet, DenseNet, MobileNet v2, EfficientNet, etc., but **not** for models with several linear layers (VGG, MobileNet v3) or none (SqueezeNet) — there, use a gradient- or score-based method, or pass a compatible `fc_layer` explicitly. All the other methods have no such requirement.

You can also pass a **list** of layers and fuse them — `LayerCAM` benefits a lot from this:

```python
from torchcam.methods import LayerCAM

with LayerCAM(model, ["layer2", "layer3", "layer4"]) as cam_extractor:
    out = model(input_tensor)
    class_idx = out.squeeze(0).argmax().item()
    cams = cam_extractor(class_idx, out)        # one map per layer
    fused = cam_extractor.fuse_cams(cams)       # single fused map
```

## Understanding `class_idx` and the call signature

```python
cam_extractor(class_idx, scores=None, normalized=True)
```

- **`class_idx`** (`int` or `list[int]`) — the index, in the output logits, of the class you want to explain. To explain the top prediction use the argmax (`out.squeeze(0).argmax().item()`), but you can pass **any** valid index to see where the model looks for that class. For a batch, pass one index per sample (see below).
- **`scores`** — the raw model output of shape `(N, num_classes)`. Required by the gradient-based methods (used for backprop) and by the Score-CAM family; ignored by `SmoothGradCAMpp` and `CAM`.
- **`normalized`** — when `True` (default) each map is min-max normalized to `[0, 1]`, which is what you want for visualization/overlay. Pass `normalized=False` to get the raw weighted maps, e.g. when comparing magnitudes across layers before fusing them yourself.
- **Returns** a `list` of activation maps, **one tensor per hooked layer**, each of shape `(N, H, W)`. With a single layer and a single image, the map you want is `cams[0].squeeze(0)`.

Gradient-based extractors also accept `retain_graph=True` (forwarded to `loss.backward`), needed when you call the extractor several times after a single forward — see [Troubleshooting](../troubleshooting/#runtimeerror-trying-to-backward-through-the-graph-a-second-time).

## Batched inputs

Batches are supported: pass a list of class indices whose length matches the batch size.

```python
import torch
from torchcam.methods import GradCAM

input_batch = torch.stack([img1, img2, img3])   # (3, C, H, W)
with GradCAM(model) as cam_extractor:
    out = model(input_batch)                     # (3, num_classes)
    class_ids = out.argmax(dim=1).tolist()       # one class per sample
    cams = cam_extractor(class_ids, out)         # cams[0] has shape (3, H, W)
```

## Models with multiple inputs or non-tensor outputs

The extractor expects the **model output to be the class logits**, and the hooked layer to output a single tensor. If your model returns a tuple/dict (e.g. `(logits, aux)`) or takes several inputs (e.g. a siamese network), wrap it so the forward used for the CAM returns a single logits tensor:

```python
import torch.nn as nn
from torchcam.methods import GradCAM

class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]          # keep the logits, drop the rest

wrapped = LogitsOnly(model)
cam_extractor = GradCAM(wrapped, target_layer=wrapped.model.backbone.layer4)  # pass the module directly
```

Passing the **module object** (rather than its name) sidesteps the naming gotcha: wrapping shifts every layer name under a `"model."` prefix, so a hard-coded string like `"backbone.layer4"` would raise a `ValueError`. If you prefer names, discover the correct one *after* wrapping:

```python
print([n for n, _ in wrapped.named_modules() if n.endswith("layer4")])
# -> ['model.backbone.layer4']
```

## Vision Transformers and other non-CNN models

TorchCAM's methods operate on **convolutional feature maps** of shape `(N, C, H, W)` (or `(N, C, D, H, W)` in 3D). Transformer blocks emit token sequences of shape `(N, num_tokens, dim)`, which have no spatial grid, so the CAM methods do not apply directly and automatic `target_layer` resolution will not find a suitable layer.

To use a ViT you have to reshape a block's token output back to a spatial grid (dropping the class token) and expose *that* as the `target_layer`. There is no built-in helper yet, but the wrapper below is a starting point — it turns a block's `(N, num_tokens, dim)` output into an `(N, dim, H, W)` map:

```python
import torch.nn as nn

class ViTToGrid(nn.Module):
    """Experimental: reshape a ViT block's token output to a spatial grid for CAM."""
    def __init__(self, vit_block, h, w):   # h, w = patch grid, e.g. 14x14 for 224px / patch 16
        super().__init__()
        self.block = vit_block
        self.h, self.w = h, w

    def forward(self, x):
        out = self.block(x)                                  # (N, num_tokens, dim)
        return out[:, 1:].transpose(1, 2).reshape(x.size(0), -1, self.h, self.w)  # (N, dim, h, w)
```

Insert it into the model in place of the block you want to read, then point the extractor at it.

Experimental

This is a sketch, not a drop-in. The exact reshape is architecture-specific (patch grid size, and whether there is a class and/or distillation token to drop), and for gradient-based methods you must read a block whose tokens the class score actually depends on. If you get this working for a model, please share it in the [discussions](https://github.com/frgfm/torch-cam/discussions).

## 3D and video models

Volumetric inputs work out of the box: set `input_shape` to your 3D input shape as `(C, D, H, W)` (i.e. excluding the batch dimension) and the resulting map has shape `(N, D, H, W)`. Visualize it slice by slice:

```python
import matplotlib.pyplot as plt
from torchcam.methods import GradCAM

cam_extractor = GradCAM(model, target_layer="...", input_shape=(1, 64, 128, 128))
out = model(volume)                                  # volume: (N, C, D, H, W)
cam = cam_extractor(out.squeeze(0).argmax().item(), out)[0]   # (N, D, H, W)
plt.imshow(cam[0, 32].cpu().numpy())                 # one depth slice
```

Video models that output `(N, C, T, H, W)` features are handled the same way (the temporal axis behaves like an extra spatial dimension). Note that `overlay_mask` works on 2D PIL images, so overlay each slice/frame separately.

## Choosing a CAM method

| Method                         | Needs gradients | Relative cost          | Notes                                                                                                |
| ------------------------------ | --------------- | ---------------------- | ---------------------------------------------------------------------------------------------------- |
| `CAM`                          | no              | cheapest               | needs global pooling + a **single** `nn.Linear` head (e.g. ResNet); fails on multi-FC heads like VGG |
| `GradCAM`                      | yes             | one backward pass      | robust default for most CNNs                                                                         |
| `LayerCAM`                     | yes             | one backward pass      | best localization in our benchmark; ideal when fusing layers                                         |
| `GradCAMpp` / `XGradCAM`       | yes             | one backward pass      | alternative weighting schemes                                                                        |
| `SmoothGradCAMpp`              | yes             | `num_samples` forwards | sharper maps via noise averaging                                                                     |
| `ScoreCAM` / `SSCAM` / `ISCAM` | no              | many forwards (slow)   | gradient-free; tune `batch_size`; useful when gradients are unavailable                              |

See the latency and faithfulness benchmarks in the [README](https://github.com/frgfm/torch-cam#performance-benchmarks) for concrete numbers, and the [methods reference](../../reference/methods/) for the full API.

## Using CAM during or after training

CAM methods are **post-hoc**: run them on a trained model in `eval()` mode to interpret its predictions — they are not a training objective. To quantify how faithful a method is on your own data, use the [`ClassificationMetric`](../../reference/metrics/).
