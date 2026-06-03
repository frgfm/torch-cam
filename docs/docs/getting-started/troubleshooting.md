# Troubleshooting

The errors below are the ones most frequently reported on the
[issue tracker](https://github.com/frgfm/torch-cam/issues) and in
[discussions](https://github.com/frgfm/torch-cam/discussions). Each entry explains the cause and the fix.

## `RuntimeError: cannot register a hook on a tensor that doesn't require gradient`

*Also reported as `element 0 of tensors does not require grad and does not have a grad_fn`.*

Gradient-based methods (`GradCAM`, `GradCAMpp`, `SmoothGradCAMpp`, `XGradCAM`, `LayerCAM`) backpropagate
through the model, so the forward pass **must build an autograd graph**. The error is raised when autograd is
disabled during the forward — almost always because it ran inside `torch.no_grad()` or `torch.inference_mode()`
(or because *every* model parameter has `requires_grad=False`).

!!! failure "Disables autograd — the gradient hook cannot attach"
    ```python
    from torchcam.methods import GradCAM

    with GradCAM(model) as cam_extractor:
        with torch.inference_mode():            # ❌ no graph is built
            out = model(input_tensor)
        cams = cam_extractor(out.squeeze(0).argmax().item(), out)
    ```

!!! success "Let autograd run during the forward"
    ```python
    from torchcam.methods import GradCAM

    with GradCAM(model) as cam_extractor:
        out = model(input_tensor)               # ✅ no no_grad / inference_mode
        cams = cam_extractor(out.squeeze(0).argmax().item(), out)
    ```

!!! tip "Activation-based methods don't need gradients"
    `CAM`, `ScoreCAM`, `SSCAM` and `ISCAM` do not backpropagate, so you *can* keep their forward pass inside
    `torch.inference_mode()`. Only the gradient-based methods require autograd.

## `AssertionError: Inputs need to be forwarded in the model ...`

The extractor reads activations through hooks that are only active while it is "open". Run the forward pass
**after** creating the extractor (or inside its `with` block), not before:

```python
with GradCAM(model) as cam_extractor:   # hooks registered here
    out = model(input_tensor)           # forward happens while hooks are live
    cams = cam_extractor(class_idx, out)
```

## The activation map is all `NaN`

TorchCAM guards the normalization denominator with an epsilon, so a *flat* map yields zeros rather than `NaN`.
A `NaN` map therefore almost always means the hooked activations or gradients already contained `NaN`/`Inf`
coming from the model itself.

- Check the output: `assert torch.isfinite(out).all()`.
- Run the CAM forward in `float32` (avoid mixed-precision/`autocast` for this step).
- Try a different `target_layer`, or a more numerically stable method (`GradCAM`, `LayerCAM`) — the higher-order
  terms in `GradCAMpp`/`SmoothGradCAMpp` are more prone to floating-point underflow.

## The heatmap is blank / all zeros

Gradient methods apply a `ReLU` before normalization, so a layer with no positive contribution to the chosen
class produces an empty map. Common causes:

- `class_idx` is not the class the layer responds to — pass the prediction: `out.squeeze(0).argmax().item()`.
- The target layer is too shallow/deep — try `LayerCAM` (keeps pixel-wise positive contributions) or another layer.
- The model is untrained or the input is out of distribution.

## The CAM changes on every run

The model is in training mode, so dropout / batch-norm add randomness. Switch to eval mode before extracting:

```python
model.eval()
```

## `RuntimeError: Trying to backward through the graph a second time`

This happens when you call the extractor more than once after a single forward pass (e.g. to compare several
classes): the first backward frees the graph. Pass `retain_graph=True` on every call but the last, or re-run the
forward before each call:

```python
with GradCAM(model) as cam_extractor:
    out = model(input_tensor)
    cam_a = cam_extractor(class_a, out, retain_graph=True)
    cam_b = cam_extractor(class_b, out)
```

## `AttributeError: '...' object has no attribute 'enable_hooks'` / errors when clearing hooks

This usually means a stale or mismatched install. The current API exposes `enable_hooks()`, `disable_hooks()`,
`remove_hooks()` and `reset_hooks()`, and supports the context-manager form (`with GradCAM(model) as ...:`),
which removes the hooks automatically on exit.

- Upgrade first: `pip install -U torchcam`.
- Prefer the `with` form so hooks are always cleaned up.
- Use **one extractor per model at a time** — instantiating several extractors on the same model stacks hooks.
  Scope them with `with`, or call `remove_hooks()` when done.

## `ImportError: cannot import name '...' from 'torchcam'`

The symbol exists in a newer release than the one installed. Upgrade with `pip install -U torchcam`, or install
the latest unreleased version from source (see the [installation guide](installation.md)).

## Still stuck?

Open a [discussion](https://github.com/frgfm/torch-cam/discussions) or an
[issue](https://github.com/frgfm/torch-cam/issues) with a minimal snippet **and** the output of:

```python
import torch, torchvision, torchcam
print(torchcam.__version__, torch.__version__, torchvision.__version__)
```
