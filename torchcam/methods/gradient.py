# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import partial
from math import isfinite, isqrt
from types import TracebackType
from typing import Any, Self, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import _CAM, OutputTarget, _target_scores

__all__ = ["FinerCAM", "GradCAM", "GradCAMpp", "LayerCAM", "LeGrad", "RefineCAM", "SmoothGradCAMpp", "XGradCAM"]


class _GradCAM(_CAM):
    """Implements a gradient-based class activation map extractor.

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str | list[nn.Module | str] | None = None,
        input_shape: tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, target_layer, input_shape, **kwargs)
        # Ensure ReLU is applied before normalization
        self._relu = True
        # Model output is used by the extractor
        self._score_used = True
        self._targets_supported = True
        for idx, name in enumerate(self.target_names):
            # Trick to avoid issues with inplace operations cf. https://github.com/pytorch/pytorch/issues/61519
            self.hook_handles.append(self.submodule_dict[name].register_forward_hook(partial(self._hook_g, idx=idx)))

    def reset_hooks(self) -> None:
        super().reset_hooks()
        self._hook_outputs: list[Tensor | None] = [None] * len(self.target_names)

    def _hook_g(self, _: nn.Module, _input: tuple[Tensor, ...], output: Tensor, idx: int = 0) -> None:
        """Store the target-layer output used to compute gradients."""
        if self._hooks_enabled:
            self._hook_outputs[idx] = output

    def _backprop(
        self,
        scores: Any,
        class_idx: int | list[int] | None,
        retain_graph: bool = False,
        targets: OutputTarget | list[OutputTarget] | None = None,
    ) -> None:
        """Backpropagate the loss for a specific output class.

        Raises:
            RuntimeError: if the target score is disconnected from a target layer
        """
        if targets is not None:
            loss = _target_scores(scores, targets).sum()
        elif isinstance(class_idx, int):
            loss = scores[:, class_idx].sum()
        else:
            loss = scores.gather(1, torch.tensor(class_idx, device=scores.device).view(-1, 1)).sum()
        outputs = cast(list[Tensor], self._hook_outputs)
        try:
            gradients = torch.autograd.grad(loss, outputs, retain_graph=retain_graph)
        except RuntimeError as exc:
            raise RuntimeError("target score is not connected to every target layer") from exc
        for idx, gradient in enumerate(gradients):
            transformed = self._reshape_transform(gradient) if self._reshape_transform is not None else gradient
            self.hook_g[idx] = transformed.detach()


class GradCAM(_GradCAM):
    r"""Implements a class activation map extractor as described in ["Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization"](https://arxiv.org/pdf/1610.02391.pdf).

    The localization map is computed as follows:

    $$
    L^{(c)}_{Grad-CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)
    $$

    with the coefficient $w_k^{(c)}$ being defined as:

    $$
    w_k^{(c)} = \frac{1}{H \cdot W} \sum\limits_{i=1}^H \sum\limits_{j=1}^W
    \frac{\partial Y^{(c)}}{\partial A_k(i, j)}
    $$

    where $A_k(x, y)$ is the activation of node $k$ in the target layer of the model at
    position $(x, y)$,
    and $Y^{(c)}$ is the model output score for class $c$ before softmax.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import GradCAM
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with GradCAM(model, 'layer4') as cam_extractor:
            scores = model(input_tensor)
            cam = cam_extractor(class_idx=100, scores=scores)
        ```

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(self, class_idx: int | list[int], scores: Tensor, **kwargs: Any) -> list[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""  # noqa: DOC201
        # Backpropagate
        self._backprop(scores, class_idx, **kwargs)

        self.hook_g: list[Tensor]  # type: ignore[assignment]
        # Global average pool the gradients over spatial dimensions
        return [grad.flatten(2).mean(-1) for grad in self.hook_g]


class GradCAMpp(_GradCAM):
    r"""Implements a class activation map extractor as described in ["Grad-CAM++: Improved Visual Explanations for
    Deep Convolutional Networks"](https://arxiv.org/pdf/1710.11063.pdf).

    The localization map is computed as follows:

    $$
    L^{(c)}_{Grad-CAM++}(x, y) = \sum\limits_k w_k^{(c)} A_k(x, y)
    $$

    with the coefficient $w_k^{(c)}$ being defined as:

    $$
    w_k^{(c)} = \sum\limits_{i=1}^H \sum\limits_{j=1}^W \alpha_k^{(c)}(i, j) \cdot
    ReLU\Big(\frac{\partial Y^{(c)}}{\partial A_k(i, j)}\Big)
    $$

    where $A_k(x, y)$ is the activation of node $k$ in the target layer of the model at
    position $(x, y)$,
    $Y^{(c)}$ is the model output score for class $c$ before softmax,
    and $\alpha_k^{(c)}(i, j)$ being defined as:

    $$
    \alpha_k^{(c)}(i, j) = \frac{1}{\sum\limits_{i, j} \frac{\partial Y^{(c)}}{\partial A_k(i, j)}}
    = \frac{\frac{\partial^2 Y^{(c)}}{(\partial A_k(i,j))^2}}{2 \cdot
    \frac{\partial^2 Y^{(c)}}{(\partial A_k(i,j))^2} + \sum\limits_{a,b} A_k (a,b) \cdot
    \frac{\partial^3 Y^{(c)}}{(\partial A_k(i,j))^3}}
    $$

    if $\frac{\partial Y^{(c)}}{\partial A_k(i, j)} = 1$ else $0$.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import GradCAMpp
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with GradCAMpp(model, 'layer4') as cam_extractor:
            scores = model(input_tensor)
            cam = cam_extractor(class_idx=100, scores=scores)
        ```

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(
        self,
        class_idx: int | list[int],
        scores: Tensor,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""  # noqa: DOC201
        # Backpropagate
        self._backprop(scores, class_idx, **kwargs)
        self.hook_a: list[Tensor]  # type: ignore[assignment]
        self.hook_g: list[Tensor]  # type: ignore[assignment]
        # Alpha coefficient for each pixel
        grad_2 = [grad.pow(2) for grad in self.hook_g]
        grad_3 = [g2 * grad for g2, grad in zip(grad_2, self.hook_g, strict=True)]
        # Watch out for NaNs produced by underflow
        spatial_dims = self.hook_a[0].ndim - 2
        denom = [
            2 * g2 + (g3 * act).flatten(2).sum(-1)[(...,) + (None,) * spatial_dims]
            for g2, g3, act in zip(grad_2, grad_3, self.hook_a, strict=True)
        ]
        nan_mask = [g2 > 0 for g2 in grad_2]
        alpha = grad_2
        for idx, d, mask in zip(range(len(grad_2)), denom, nan_mask, strict=True):
            alpha[idx][mask].div_(d[mask] + eps)

        # Apply pixel coefficient in each weight
        return [a.mul_(torch.relu(grad)).flatten(2).sum(-1) for a, grad in zip(alpha, self.hook_g, strict=True)]


class SmoothGradCAMpp(_GradCAM):
    r"""Implements a class activation map extractor as described in ["Smooth Grad-CAM++: An Enhanced Inference Level
    Visualization Technique for Deep Convolutional Neural Network Models"](https://arxiv.org/pdf/1908.01224.pdf)
    with a personal correction to the paper (alpha coefficient numerator).

    The localization map is computed as follows:

    $$
    L^{(c)}_{Smooth Grad-CAM++}(x, y) = \sum\limits_k w_k^{(c)} A_k(x, y)
    $$

    with the coefficient $w_k^{(c)}$ being defined as:

    $$
    w_k^{(c)} = \sum\limits_{i=1}^H \sum\limits_{j=1}^W \alpha_k^{(c)}(i, j) \cdot
    ReLU\Big(\frac{\partial Y^{(c)}}{\partial A_k(i, j)}\Big)
    $$

    where $A_k(x, y)$ is the activation of node $k$ in the target layer of the model at
    position $(x, y)$,
    $Y^{(c)}$ is the model output score for class $c$ before softmax,
    and $\alpha_k^{(c)}(i, j)$ being defined as:

    $$
    \alpha_k^{(c)}(i, j)
    = \frac{\frac{\partial^2 Y^{(c)}}{(\partial A_k(i,j))^2}}{2 \cdot
    \frac{\partial^2 Y^{(c)}}{(\partial A_k(i,j))^2} + \sum\limits_{a,b} A_k (a,b) \cdot
    \frac{\partial^3 Y^{(c)}}{(\partial A_k(i,j))^3}}
    = \frac{\frac{1}{n} \sum\limits_{m=1}^n D^{(c, 2)}_k(i, j)}{
    \frac{2}{n} \sum\limits_{m=1}^n D^{(c, 2)}_k(i, j) + \sum\limits_{a,b} A_k (a,b) \cdot
    \frac{1}{n} \sum\limits_{m=1}^n D^{(c, 3)}_k(i, j)}
    $$

    if $\frac{\partial Y^{(c)}}{\partial A_k(i, j)} = 1$ else $0$. Here $D^{(c, p)}_k(i, j)$
    refers to the p-th partial derivative of the class score of class $c$ relatively to the activation in layer
    $k$ at position $(i, j)$, and $n$ is the number of samples used to get the gradient estimate.

    Please note the difference in the numerator of $\alpha_k^{(c)}(i, j)$,
    which is actually $\frac{1}{n} \sum\limits_{k=1}^n D^{(c, 1)}_k(i,j)$ in the paper.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import SmoothGradCAMpp
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with SmoothGradCAMpp(model, 'layer4') as cam_extractor:
            scores = model(input_tensor)
            cam = cam_extractor(class_idx=100)
        ```

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        num_samples: number of samples to use for smoothing
        std: standard deviation of the noise
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str | list[nn.Module | str] | None = None,
        num_samples: int = 4,
        std: float = 0.3,
        input_shape: tuple[int, ...] = (3, 224, 224),
        **kwargs: Any,
    ) -> None:
        super().__init__(model, target_layer, input_shape, **kwargs)
        # Model scores is not used by the extractor
        self._score_used = False

        # Input hook
        self.hook_handles.append(model.register_forward_pre_hook(self._store_input))
        # Noise distribution
        self.num_samples = num_samples
        self.std = std
        self._distrib = torch.distributions.normal.Normal(0, self.std)
        # Specific input hook updater
        self._ihook_enabled = True

    def _store_input(self, _: nn.Module, input_: tuple[Any, ...]) -> None:
        """Store model input tensor."""
        if self._ihook_enabled:
            self._input = input_[0].detach().clone()

    def _get_weights(
        self,
        class_idx: int | list[int],
        _: Tensor | None = None,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""  # noqa: DOC201
        previous_ihook_enabled = self._ihook_enabled
        self._ihook_enabled = False
        try:
            return self._compute_smoothgrad_weights(class_idx, eps, **kwargs)
        finally:
            self._ihook_enabled = previous_ihook_enabled

    def _compute_smoothgrad_weights(
        self,
        class_idx: int | list[int],
        eps: float,
        **kwargs: Any,
    ) -> list[Tensor]:
        # Keep initial activation
        self.hook_a: list[Tensor]  # type: ignore[assignment]
        self.hook_g: list[Tensor]  # type: ignore[assignment]
        init_fmap = [act.clone() for act in self.hook_a]
        # Initialize our gradient estimates
        grad_2 = [torch.zeros_like(act) for act in self.hook_a]
        grad_3 = [torch.zeros_like(act) for act in self.hook_a]
        # Perform the operations N times
        for _idx in range(self.num_samples):
            # Add noise
            noisy_input = self._input + self._distrib.sample(self._input.size()).to(device=self._input.device)
            noisy_input.requires_grad_(True)
            # Forward & Backward
            out = self.model(noisy_input)
            self._backprop(out, class_idx, **kwargs)

            # Sum partial derivatives
            grad_2 = [g2.add_(grad.pow(2)) for g2, grad in zip(grad_2, self.hook_g, strict=True)]
            grad_3 = [g3.add_(grad.pow(3)) for g3, grad in zip(grad_3, self.hook_g, strict=True)]

        # Average the gradient estimates
        grad_2 = [g2.div_(self.num_samples) for g2 in grad_2]
        grad_3 = [g3.div_(self.num_samples) for g3 in grad_3]

        # Alpha coefficient for each pixel
        spatial_dims = self.hook_a[0].ndim - 2
        alpha = [
            g2 / (2 * g2 + (g3 * act).flatten(2).sum(-1)[(...,) + (None,) * spatial_dims] + eps)
            for g2, g3, act in zip(grad_2, grad_3, init_fmap, strict=True)
        ]

        # Apply pixel coefficient in each weight
        return [a.mul_(torch.relu(grad)).flatten(2).sum(-1) for a, grad in zip(alpha, self.hook_g, strict=True)]

    def _extra_repr(self) -> str:
        return f"target_layer={self.target_names}, num_samples={self.num_samples}, std={self.std}"


class XGradCAM(_GradCAM):
    r"""Implements a class activation map extractor as described in ["Axiom-based Grad-CAM: Towards Accurate
    Visualization and Explanation of CNNs"](https://arxiv.org/pdf/2008.02312.pdf).

    The localization map is computed as follows:

    $$
    L^{(c)}_{XGrad-CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)
    $$

    with the coefficient $w_k^{(c)}$ being defined as:

    $$
    w_k^{(c)} = \sum\limits_{i=1}^H \sum\limits_{j=1}^W
    \Big( \frac{\partial Y^{(c)}}{\partial A_k(i, j)} \cdot
    \frac{A_k(i, j)}{\sum\limits_{m=1}^H \sum\limits_{n=1}^W A_k(m, n)} \Big)
    $$

    where $A_k(x, y)$ is the activation of node $k$ in the target layer of the model at
    position $(x, y)$,
    and $Y^{(c)}$ is the model output score for class $c$ before softmax.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import XGradCAM
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with XGradCAM(model, 'layer4') as cam_extractor:
            scores = model(input_tensor)
            cam = cam_extractor(class_idx=100, scores=scores)
        ```

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(
        self,
        class_idx: int | list[int],
        scores: Tensor,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""  # noqa: DOC201
        # Backpropagate
        self._backprop(scores, class_idx, **kwargs)

        self.hook_a: list[Tensor]  # type: ignore[assignment]
        self.hook_g: list[Tensor]  # type: ignore[assignment]
        return [
            (grad * act).flatten(2).sum(-1) / act.flatten(2).sum(-1).add(eps)
            for act, grad in zip(self.hook_a, self.hook_g, strict=True)
        ]


class LayerCAM(_GradCAM):
    r"""Implements a class activation map extractor as described in ["LayerCAM: Exploring Hierarchical Class Activation
    Maps for Localization"](http://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf).

    The localization map is computed as follows:

    $$
    L^{(c)}_{Layer-CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)}(x, y) \cdot A_k(x, y)\Big)
    $$

    with the coefficient $w_k^{(c)}(x, y)$ being defined as:

    $$
    w_k^{(c)}(x, y) = ReLU\Big(\frac{\partial Y^{(c)}}{\partial A_k(i, j)}(x, y)\Big)
    $$

    where $A_k(x, y)$ is the activation of node $k$ in the target layer of the model at
    position $(x, y)$,
    and $Y^{(c)}$ is the model output score for class $c$ before softmax.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import LayerCAM
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with LayerCAM(model, 'layer4') as cam_extractor:
            scores = model(input_tensor)
            cams = cam_extractor(class_idx=100, scores=scores)
            fused_cam = cam_extractor.fuse_cams(cams)
        ```

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(self, class_idx: int | list[int], scores: Tensor, **kwargs: Any) -> list[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""  # noqa: DOC201
        # Backpropagate
        self._backprop(scores, class_idx, **kwargs)

        self.hook_g: list[Tensor]  # type: ignore[assignment]
        # List of (N, C, H, W)
        return [torch.relu(grad) for grad in self.hook_g]

    @staticmethod
    def _scale_cams(cams: list[Tensor], gamma: float = 2.0) -> list[Tensor]:
        # cf. Equation 9 in the paper
        return [torch.tanh(gamma * cam) for cam in cams]


class _CAMWrapper:
    """Delegate CAM extractor metadata and lifecycle without duplicating hooks."""

    def __init__(self, base_cam: _CAM) -> None:
        self.base_cam = base_cam

    @property
    def model(self) -> nn.Module:
        """The model wrapped by the base extractor."""
        return self.base_cam.model

    @property
    def target_names(self) -> list[str]:
        """The target layer names used by the base extractor."""
        return self.base_cam.target_names

    def enable_hooks(self) -> None:
        """Enable the base extractor hooks."""
        self.base_cam.enable_hooks()

    def disable_hooks(self) -> None:
        """Disable the base extractor hooks."""
        self.base_cam.disable_hooks()

    def reset_hooks(self) -> None:
        """Clear the activations and gradients stored by the base extractor."""
        self.base_cam.reset_hooks()

    def remove_hooks(self) -> None:
        """Remove the base extractor hooks from the model."""
        self.base_cam.remove_hooks()

    def _hooks_off(self) -> AbstractContextManager[None]:
        return self.base_cam._hooks_off()  # noqa: SLF001

    def __enter__(self) -> Self:
        self.base_cam.__enter__()
        return self

    def __exit__(
        self,
        exct_type: type[BaseException] | None,
        exce_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.base_cam.__exit__(exct_type, exce_value, traceback)


class LeGrad(_CAM):
    r"""Implements LeGrad as described in ["LeGrad: An Explainability Method for Vision Transformers via Feature
    Formation Sensitivity"](https://arxiv.org/abs/2404.03214).

    For every selected transformer block :math:`l`, LeGrad differentiates that block's class score
    :math:`s^l` with respect to its post-softmax attention probabilities :math:`A^l`. Positive gradients are
    averaged over heads and query tokens, prefix keys are removed, and the resulting patch maps are averaged over
    layers before normalization. Attention values are not multiplied into the gradients.

    Target layers must return tokens shaped ``(batch, tokens, embedding)`` and expose a direct
    ``self_attention`` child implemented by batch-first :class:`torch.nn.MultiheadAttention`. The built-in score
    projection supports torchvision ``VisionTransformer`` models; other matching blocks require
    ``score_projection`` to map intermediate tokens to class logits.

    Example:
        ```python
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        from torchcam.methods import LeGrad

        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval()
        with LeGrad(model, list(model.encoder.layers)[-4:]) as cam_extractor:
            scores = model(input_tensor)
            cam = cam_extractor(scores[0].argmax().item())[0]
        ```

    Args:
        model: input model
        target_layer: transformer block or blocks, specified as modules or their names
        score_projection: optional function mapping intermediate tokens to class logits shaped ``(N, C)``
        prefix_tokens: number of non-spatial key tokens to remove before reshaping
        grid_shape: patch-grid ``(height, width)``; inferred when the patch count is a perfect square
        enable_hooks: whether hooks should be enabled by default

    Raises:
        TypeError: if an argument has an invalid type
        ValueError: if the model, target blocks, attention modules, or grid are unsupported
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str | list[nn.Module | str],
        *,
        score_projection: Callable[[Tensor], Tensor] | None = None,
        prefix_tokens: int = 1,
        grid_shape: tuple[int, int] | None = None,
        enable_hooks: bool = True,
    ) -> None:
        if target_layer is None or (isinstance(target_layer, list) and not target_layer):
            raise ValueError("LeGrad requires at least one explicit target block")
        self._validate_init_args(prefix_tokens, grid_shape, score_projection)

        if score_projection is None:
            encoder = getattr(model, "encoder", None)
            norm = getattr(encoder, "ln", None)
            heads = getattr(model, "heads", None)
            if not isinstance(norm, nn.Module) or not isinstance(heads, nn.Module):
                raise ValueError("`score_projection` is required for models other than torchvision VisionTransformer")

            def project_tokens(tokens: Tensor) -> Tensor:
                return heads(norm(tokens.mean(dim=1)))

            score_projection = project_tokens

        self._score_projection = score_projection
        self.prefix_tokens = prefix_tokens
        self.grid_shape = grid_shape
        super().__init__(model, target_layer, enable_hooks=enable_hooks)

        try:
            self._register_attention_hooks()
        except Exception:
            self.remove_hooks()
            raise

    @staticmethod
    def _validate_init_args(
        prefix_tokens: int,
        grid_shape: tuple[int, int] | None,
        score_projection: Callable[[Tensor], Tensor] | None,
    ) -> None:
        if not isinstance(prefix_tokens, int):
            raise TypeError("`prefix_tokens` must be an integer")
        if prefix_tokens < 0:
            raise ValueError("`prefix_tokens` must be non-negative")
        if grid_shape is not None and (
            not isinstance(grid_shape, tuple)
            or len(grid_shape) != 2
            or any(not isinstance(dim, int) for dim in grid_shape)
        ):
            raise TypeError("`grid_shape` must be a tuple of two integers")
        if grid_shape is not None and any(dim <= 0 for dim in grid_shape):
            raise ValueError("`grid_shape` dimensions must be positive")
        if score_projection is not None and not callable(score_projection):
            raise TypeError("`score_projection` must be callable")

    def _register_attention_hooks(self) -> None:
        for idx, name in enumerate(self.target_names):
            attention = getattr(self.submodule_dict[name], "self_attention", None)
            if not isinstance(attention, nn.MultiheadAttention):
                raise TypeError(f"target block '{name}' must expose `self_attention: nn.MultiheadAttention`")
            self._validate_attention(attention, name)
            self.hook_handles.append(
                attention.register_forward_pre_hook(partial(self._prepare_attention, idx=idx), with_kwargs=True)
            )
            self.hook_handles.append(
                attention.register_forward_hook(partial(self._capture_attention, idx=idx), with_kwargs=True)
            )

    @staticmethod
    def _validate_attention(attention: nn.MultiheadAttention, name: str) -> None:
        if not attention.batch_first:
            raise ValueError(f"target block '{name}' requires batch-first attention")
        if (
            attention.kdim != attention.embed_dim
            or attention.vdim != attention.embed_dim
            or attention.in_proj_weight is None
        ):
            raise ValueError(f"target block '{name}' requires shared-dimension Q/K/V projections")
        if attention.bias_k is not None or attention.bias_v is not None or attention.add_zero_attn:
            raise ValueError(f"target block '{name}' does not support added attention tokens")

    def reset_hooks(self) -> None:
        """Clear stored layer scores and attention probabilities."""
        super().reset_hooks()
        self.hook_attn: list[Tensor | None] = [None] * len(self.target_names)
        self._token_counts: list[int | None] = [None] * len(self.target_names)
        self._weight_options: list[tuple[bool, bool] | None] = [None] * len(self.target_names)

    def _prepare_attention(
        self,
        module: nn.MultiheadAttention,
        input_: tuple[Any, ...],
        kwargs: dict[str, Any],
        idx: int = 0,
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if not self._hooks_enabled:
            return None
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise RuntimeError("LeGrad requires a forward pass with gradient tracking enabled")
        if module.training and module.dropout > 0:
            raise ValueError("LeGrad does not support active attention dropout")
        if len(input_) < 3 or any(not isinstance(arg, Tensor) for arg in input_[:3]):
            raise ValueError("LeGrad requires tensor query, key, and value inputs")

        query, key, value = input_[:3]
        if query is not key or query is not value:
            raise ValueError("LeGrad only supports self-attention")
        if query.ndim != 3:
            raise ValueError("LeGrad requires attention inputs shaped (batch, tokens, embedding)")

        args = list(input_)
        in_proj_bias = module.in_proj_bias
        if not (
            query.requires_grad
            or module.in_proj_weight.requires_grad
            or (in_proj_bias is not None and in_proj_bias.requires_grad)
        ):
            # Keep parameter flags untouched while making attention differentiable.
            query = query.detach().requires_grad_()
            args[:3] = [query, query, query]

        # These positions follow nn.MultiheadAttention.forward; tests cover positional calls.
        requested = args[4] if len(args) > 4 else kwargs.get("need_weights", True)
        averaged = args[6] if len(args) > 6 else kwargs.get("average_attn_weights", True)
        self._weight_options[idx] = (bool(requested), bool(averaged))

        kwargs = dict(kwargs)
        if len(args) > 4:
            args[4] = True
        else:
            kwargs["need_weights"] = True
        if len(args) > 6:
            args[6] = False
        else:
            kwargs["average_attn_weights"] = False
        return tuple(args), kwargs

    def _capture_attention(
        self,
        module: nn.MultiheadAttention,
        input_: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: tuple[Tensor, Tensor | None],
        idx: int = 0,
    ) -> tuple[Tensor, Tensor | None] | None:
        if not self._hooks_enabled:
            return None
        if not isinstance(output, tuple) or len(output) != 2 or not isinstance(output[1], Tensor):
            raise ValueError("LeGrad requires per-head attention probabilities from `nn.MultiheadAttention`")

        attention = output[1]
        value = input_[2]
        if not isinstance(value, Tensor) or attention.ndim != 4:
            raise ValueError("LeGrad requires attention probabilities shaped (batch, heads, queries, keys)")

        batch_size, source_length, _ = value.shape
        if (
            attention.shape[0] != batch_size
            or attention.shape[1] != module.num_heads
            or attention.shape[-1] != source_length
        ):
            raise ValueError("incompatible attention probability shape")

        # Replay aggregation so the captured post-softmax tensor is downstream-used.
        embed_dim = module.embed_dim
        value = F.linear(
            value,
            module.in_proj_weight[2 * embed_dim :],
            None if module.in_proj_bias is None else module.in_proj_bias[2 * embed_dim :],
        )
        value = value.reshape(batch_size, source_length, module.num_heads, embed_dim // module.num_heads)
        value = value.transpose(1, 2)
        attention_output = torch.matmul(attention, value).transpose(1, 2)
        attention_output = attention_output.reshape(batch_size, attention.shape[-2], embed_dim)
        attention_output = module.out_proj(attention_output)
        self.hook_attn[idx] = attention

        options = self._weight_options[idx]
        if options is None:
            raise RuntimeError("missing original attention return options")
        requested, averaged = options
        returned_attention = attention.mean(dim=1) if requested and averaged else attention if requested else None
        return attention_output, returned_attention

    def _hook_a(self, _: nn.Module, _input: tuple[Tensor, ...], output: Tensor, idx: int = 0) -> None:
        if not self._hooks_enabled:
            return
        if not isinstance(output, Tensor) or output.ndim != 3:
            raise ValueError("LeGrad target blocks must return tokens shaped (batch, tokens, embedding)")
        # Reuse inherited activation slots for layer-specific logits.
        scores = self._score_projection(output)
        if not isinstance(scores, Tensor) or scores.ndim != 2 or scores.shape[0] != output.shape[0]:
            raise ValueError("`score_projection` must return class logits shaped (batch, classes)")
        self.hook_a[idx] = scores
        self._token_counts[idx] = output.shape[1]

    def _precheck(
        self,
        class_idx: int | list[int] | None,
        scores: Any = None,
        targets: OutputTarget | list[OutputTarget] | None = None,
    ) -> None:
        super()._precheck(class_idx, scores, targets)
        if any(not isinstance(attention, Tensor) for attention in self.hook_attn):
            raise AssertionError("Inputs need to be forwarded through every LeGrad target block")

    def _get_weights(
        self,
        class_idx: int | list[int],
        _scores: Tensor | None = None,
        retain_graph: bool = False,
        **_: Any,
    ) -> list[Tensor]:
        relevances: list[Tensor] = []
        for idx, (scores, attention) in enumerate(zip(self.hook_a, self.hook_attn, strict=True)):
            if not isinstance(scores, Tensor) or not isinstance(attention, Tensor):
                raise AssertionError("Inputs need to be forwarded through every LeGrad target block")  # noqa: TRY004
            selected = (
                scores[:, class_idx]
                if isinstance(class_idx, int)
                else scores.gather(1, torch.tensor(class_idx, device=scores.device).view(-1, 1)).squeeze(1)
            )
            try:
                grad = torch.autograd.grad(
                    selected.sum(),
                    attention,
                    retain_graph=retain_graph or idx < len(self.target_names) - 1,
                )[0]
            except RuntimeError as exc:
                raise RuntimeError(
                    f"layer-specific score for '{self.target_names[idx]}' is not connected to its attention"
                ) from exc
            relevances.append(grad.relu().mean(dim=(1, 2))[:, self.prefix_tokens :])
        return relevances

    def _resolve_grid_shape(self, patch_count: int) -> tuple[int, int]:
        if patch_count <= 0:
            raise ValueError("`prefix_tokens` must leave at least one patch token")
        if self.grid_shape is not None:
            if self.grid_shape[0] * self.grid_shape[1] != patch_count:
                raise ValueError("`grid_shape` does not match the number of patch tokens")
            return self.grid_shape
        side = isqrt(patch_count)
        if side * side != patch_count:
            raise ValueError("unable to infer a square patch grid; specify `grid_shape`")
        return side, side

    def compute_cams(
        self,
        class_idx: int | list[int] | None = None,
        scores: Any = None,
        normalized: bool = True,
        retain_graph: bool = False,
        *,
        targets: OutputTarget | list[OutputTarget] | None = None,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute and average layerwise positive attention-gradient maps.

        Raises:
            ValueError: if attention and token shapes are incompatible
        """  # noqa: DOC201
        if targets is not None:
            raise ValueError("LeGrad does not support arbitrary output targets")
        relevances = self._get_weights(cast(int | list[int], class_idx), scores, retain_graph=retain_graph, **kwargs)
        maps: list[Tensor] = []
        for idx, relevance in enumerate(relevances):
            token_count = self._token_counts[idx]
            if token_count is None or token_count != relevance.shape[-1] + self.prefix_tokens:
                raise ValueError("attention keys and target-block tokens must have matching lengths")
            height, width = self._resolve_grid_shape(relevance.shape[-1])
            maps.append(relevance.reshape(relevance.shape[0], height, width))

        with torch.no_grad():
            cam = torch.stack(maps).mean(dim=0)
            return [self._normalize(cam) if normalized else cam]


class RefineCAM(_CAMWrapper):
    r"""Implements the multi-layer refinement described in ["How to Evaluate and Refine your CAM"](
    https://arxiv.org/abs/2605.14641).

    RefineCAM normalizes class activation maps from multiple layers, resizes them to a common spatial shape, and
    multiplies them element-wise. Grad-CAM++ is used by default, but any CAM extractor supporting multiple target
    layers can be passed as ``base_method``.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import LayerCAM, RefineCAM
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with RefineCAM(model, ["layer2", "layer3", "layer4"], base_method=LayerCAM) as cam_extractor:
            scores = model(input_tensor)
            cam = cam_extractor(class_idx=100, scores=scores)[0]
        ```

    Args:
        model: input model
        target_layer: target layers, specified as modules or their names
        input_shape: shape of the expected input tensor excluding the batch dimension
        base_method: CAM extractor used to produce the per-layer maps
        base_kwargs: keyword arguments forwarded to ``base_method``
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: list[nn.Module | str],
        input_shape: tuple[int, ...] = (3, 224, 224),
        *,
        base_method: type[_CAM] = GradCAMpp,
        **base_kwargs: Any,
    ) -> None:
        if not isinstance(target_layer, list) or len(target_layer) < 2:
            raise ValueError("RefineCAM requires at least two target layers")
        if not isinstance(base_method, type) or not issubclass(base_method, _CAM):
            raise TypeError("base_method must be a CAM extractor class")

        super().__init__(base_method(model, target_layer, input_shape=input_shape, **base_kwargs))

    def __call__(
        self,
        class_idx: int | list[int] | None = None,
        scores: Any = None,
        normalized: bool = True,
        target_shape: tuple[int, ...] | None = None,
        *,
        targets: OutputTarget | list[OutputTarget] | None = None,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute and refine the per-layer CAMs for an output class."""  # noqa: DOC201
        cams = self.base_cam(class_idx, scores, normalized=True, targets=targets, **kwargs)
        return [self.fuse_cams(cams, target_shape, normalized)]

    def compute_cams(
        self,
        class_idx: int | list[int] | None = None,
        scores: Any = None,
        normalized: bool = True,
        target_shape: tuple[int, ...] | None = None,
        *,
        targets: OutputTarget | list[OutputTarget] | None = None,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute and refine CAMs without the base extractor precheck."""  # noqa: DOC201
        cams = self.base_cam.compute_cams(class_idx, scores, normalized=True, targets=targets, **kwargs)
        return [self.fuse_cams(cams, target_shape, normalized)]

    @staticmethod
    @torch.no_grad()
    def fuse_cams(
        cams: list[Tensor],
        target_shape: tuple[int, ...] | None = None,
        normalized: bool = True,
    ) -> Tensor:
        """Normalize, resize, and multiply maps from multiple layers.

        Raises:
            TypeError: if ``cams`` is not a list of tensors
            ValueError: if ``cams`` is empty
        """  # noqa: DOC201
        if not isinstance(cams, list) or any(not isinstance(cam, Tensor) for cam in cams):
            raise TypeError("invalid argument type for `cams`")
        if not cams:
            raise ValueError("argument `cams` cannot be an empty list")

        shape = target_shape or tuple(map(max, zip(*[tuple(cam.shape[1:]) for cam in cams], strict=True)))
        interpolation_mode = "bilinear" if cams[0].ndim == 3 else "trilinear" if cams[0].ndim == 4 else "nearest"
        resize_kwargs = {} if interpolation_mode == "nearest" else {"align_corners": False}
        resized_cams = [
            F.interpolate(
                _CAM._normalize(cam.clone()).unsqueeze(1),  # noqa: SLF001
                shape,
                mode=interpolation_mode,
                **resize_kwargs,
            )
            for cam in cams
        ]
        refined_cam = torch.stack(resized_cams).prod(dim=0).squeeze(1)
        return _CAM._normalize(refined_cam) if normalized else refined_cam  # noqa: SLF001

    def __repr__(self) -> str:
        """Return the RefineCAM representation."""  # noqa: DOC201
        return f"RefineCAM(base_method={self.base_cam.__class__.__name__}, target_layer={self.target_names})"


class FinerCAM(_CAMWrapper):
    r"""Implements ["Finer-CAM: Spotting the Difference Reveals Finer Details for Visual Explanation"](
    https://arxiv.org/abs/2501.11309).

    Finer-CAM changes the base extractor objective from the target score $y_c$ to the contrastive objective

    $$
    y_c - \gamma \frac{1}{T} \sum\limits_{t=1}^{T} y_{d_t},
    $$

    so comparisons are aggregated before the base method's final CAM ReLU. Automatic references are the classes
    with scores closest to the target score. The target is always excluded, and the requested count is capped by the
    available classes. Only GradCAM, GradCAMpp, and LayerCAM are supported initially.

    Example:
        ```python
        from torchvision.models import get_model, get_model_weights
        from torchcam.methods import FinerCAM, LayerCAM
        model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
        with FinerCAM(model, "layer4", base_method=LayerCAM) as cam_extractor:
            scores = model(input_tensor)
            cams = cam_extractor(class_idx=100, scores=scores, comparison_idx=[101, 102, 103])
        ```

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
        base_method: gradient CAM extractor used to produce the maps
        gamma: comparison strength applied to the mean reference score
        num_references: number of automatic references to select, capped by the available non-target classes
        base_kwargs: keyword arguments forwarded to ``base_method``
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module | str | list[nn.Module | str] | None = None,
        input_shape: tuple[int, ...] = (3, 224, 224),
        *,
        base_method: type[GradCAM] | type[GradCAMpp] | type[LayerCAM] = GradCAM,
        gamma: float = 0.6,
        num_references: int = 3,
        **base_kwargs: Any,
    ) -> None:
        if base_method not in {GradCAM, GradCAMpp, LayerCAM}:
            raise TypeError("base_method must be GradCAM, GradCAMpp, or LayerCAM")
        if isinstance(gamma, bool) or not isinstance(gamma, int | float):
            raise TypeError("gamma must be a real number")
        if not isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative")
        if isinstance(num_references, bool) or not isinstance(num_references, int):
            raise TypeError("num_references must be an integer")
        if num_references < 1:
            raise ValueError("num_references must be positive")

        self.gamma = float(gamma)
        self.num_references = num_references
        super().__init__(base_method(model, target_layer, input_shape=input_shape, **base_kwargs))

    @staticmethod
    def _target_indices(class_idx: int | list[int], scores: Tensor) -> tuple[list[int], Tensor]:
        batch_size, num_classes = scores.shape
        if isinstance(class_idx, int) and not isinstance(class_idx, bool):
            targets = [class_idx] * batch_size
        elif isinstance(class_idx, list):
            if len(class_idx) != batch_size:
                raise ValueError("expected batch size and length of `class_idx` to be the same")
            if any(isinstance(idx, bool) or not isinstance(idx, int) for idx in class_idx):
                raise TypeError("class_idx must contain integers")
            targets = class_idx
        else:
            raise TypeError("class_idx must be an integer or a list of integers")
        if any(idx < 0 or idx >= num_classes for idx in targets):
            raise ValueError("class_idx contains an out-of-range index")
        return targets, torch.tensor(targets, device=scores.device).view(-1, 1)

    @staticmethod
    def _explicit_reference_rows(comparison_idx: int | list[int] | list[list[int]], batch_size: int) -> list[list[int]]:
        if isinstance(comparison_idx, int) and not isinstance(comparison_idx, bool):
            return [[comparison_idx]] * batch_size
        if not isinstance(comparison_idx, list):
            raise TypeError("comparison_idx must be an integer, a list, or a nested list")
        if not comparison_idx:
            raise ValueError("comparison_idx cannot be empty")
        if all(isinstance(row, list) for row in comparison_idx):
            references = cast(list[list[int]], comparison_idx)
            if len(references) != batch_size:
                raise ValueError("per-sample comparison_idx must match the batch size")
            if any(not row for row in references):
                raise ValueError("comparison_idx rows cannot be empty")
            if len({len(row) for row in references}) != 1:
                raise ValueError("per-sample comparison_idx rows must have equal lengths")
            return references
        if all(isinstance(idx, int) and not isinstance(idx, bool) for idx in comparison_idx):
            return [cast(list[int], comparison_idx)] * batch_size
        raise TypeError("comparison_idx must contain integers or lists of integers")

    def _reference_indices(
        self,
        comparison_idx: int | list[int] | list[list[int]] | None,
        scores: Tensor,
        targets: list[int],
        target_indices: Tensor,
    ) -> Tensor:
        batch_size, num_classes = scores.shape
        if comparison_idx is None:
            reference_count = min(self.num_references, num_classes - 1)
            distances = (scores.detach() - scores.detach().gather(1, target_indices)).abs()
            distances.scatter_(1, target_indices, float("inf"))
            return distances.topk(reference_count, dim=1, largest=False).indices

        references = self._explicit_reference_rows(comparison_idx, batch_size)
        for sample_idx, (target, row) in enumerate(zip(targets, references, strict=True)):
            if any(isinstance(idx, bool) or not isinstance(idx, int) for idx in row):
                raise TypeError("comparison_idx must contain integers")
            if any(idx < 0 or idx >= num_classes for idx in row):
                raise ValueError(f"comparison_idx contains an out-of-range index for sample {sample_idx}")
            if len(set(row)) != len(row):
                raise ValueError(f"comparison_idx contains duplicates for sample {sample_idx}")
            if target in row:
                raise ValueError(f"comparison_idx contains the target class for sample {sample_idx}")
        return torch.tensor(references, device=scores.device)

    def _contrastive_scores(
        self,
        class_idx: int | list[int],
        scores: Tensor | None,
        comparison_idx: int | list[int] | list[list[int]] | None,
    ) -> Tensor:
        if scores is None:
            raise ValueError("model output scores is required to compute FinerCAM")
        if not isinstance(scores, Tensor):
            raise TypeError("scores must be a tensor")
        if scores.ndim != 2 or scores.shape[0] < 1:
            raise ValueError("scores must have shape (batch_size, num_classes)")
        if scores.shape[1] < 2:
            raise ValueError("FinerCAM requires at least one comparison class")

        targets, target_indices = self._target_indices(class_idx, scores)
        reference_indices = self._reference_indices(comparison_idx, scores, targets, target_indices)
        target_scores = scores.gather(1, target_indices)
        reference_scores = scores.gather(1, reference_indices).mean(dim=1, keepdim=True)
        return target_scores - self.gamma * reference_scores

    def __call__(
        self,
        class_idx: int | list[int],
        scores: Tensor | None = None,
        comparison_idx: int | list[int] | list[list[int]] | None = None,
        normalized: bool = True,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute Finer-CAMs for the target and comparison classes.

        Raises:
            ValueError: if arbitrary output targets are provided
        """  # noqa: DOC201
        if kwargs.get("targets") is not None:
            raise ValueError("FinerCAM does not support arbitrary output targets")
        contrastive_scores = self._contrastive_scores(class_idx, scores, comparison_idx)
        # The contrastive objective is the sole score column, at class index 0.
        return self.base_cam([0] * contrastive_scores.shape[0], contrastive_scores, normalized, **kwargs)

    def compute_cams(
        self,
        class_idx: int | list[int],
        scores: Tensor | None = None,
        comparison_idx: int | list[int] | list[list[int]] | None = None,
        normalized: bool = True,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute Finer-CAMs without the base extractor precheck.

        Raises:
            ValueError: if arbitrary output targets are provided
        """  # noqa: DOC201
        if kwargs.get("targets") is not None:
            raise ValueError("FinerCAM does not support arbitrary output targets")
        contrastive_scores = self._contrastive_scores(class_idx, scores, comparison_idx)
        return self.base_cam.compute_cams([0] * contrastive_scores.shape[0], contrastive_scores, normalized, **kwargs)

    def fuse_cams(self, cams: list[Tensor], target_shape: tuple[int, int] | None = None) -> Tensor:
        """Fuse maps using the selected base extractor."""  # noqa: DOC201
        return self.base_cam.fuse_cams(cams, target_shape)

    def __repr__(self) -> str:
        """Return the FinerCAM representation."""  # noqa: DOC201
        return (
            f"FinerCAM(base_method={self.base_cam.__class__.__name__}, target_layer={self.target_names}, "
            f"gamma={self.gamma}, num_references={self.num_references})"
        )
