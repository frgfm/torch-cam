# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from contextlib import AbstractContextManager
from functools import partial
from types import TracebackType
from typing import Any, Self

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .core import _CAM

__all__ = ["GradCAM", "GradCAMpp", "LayerCAM", "RefineCAM", "SmoothGradCAMpp", "XGradCAM"]


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
        for idx, name in enumerate(self.target_names):
            # Trick to avoid issues with inplace operations cf. https://github.com/pytorch/pytorch/issues/61519
            self.hook_handles.append(self.submodule_dict[name].register_forward_hook(partial(self._hook_g, idx=idx)))
        self._grad_hook_handles: list[torch.utils.hooks.RemovableHandle | None] = [None] * len(self.target_names)

    def _store_grad(self, grad: Tensor, idx: int = 0) -> None:
        if self._hooks_enabled:
            if self._reshape_transform is not None:
                grad = self._reshape_transform(grad)
            self.hook_g[idx] = grad.detach()

    def _hook_g(self, _: nn.Module, _input: tuple[Tensor, ...], output: Tensor, idx: int = 0) -> None:
        """Gradient hook."""
        if self._hooks_enabled:
            handle = self._grad_hook_handles[idx]
            if handle is not None:
                handle.remove()
            self._grad_hook_handles[idx] = output.register_hook(partial(self._store_grad, idx=idx))

    def remove_hooks(self) -> None:
        for handle in self._grad_hook_handles:
            if handle is not None:
                handle.remove()
        self._grad_hook_handles = [None] * len(self.target_names)
        super().remove_hooks()

    def _backprop(
        self,
        scores: Tensor,
        class_idx: int | list[int],
        retain_graph: bool = False,
    ) -> None:
        """Backpropagate the loss for a specific output class."""
        # Backpropagate to get the gradients on the hooked layer
        if isinstance(class_idx, int):
            loss = scores[:, class_idx].sum()
        else:
            loss = scores.gather(1, torch.tensor(class_idx, device=scores.device).view(-1, 1)).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=retain_graph)


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
            self.model.zero_grad()
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


class RefineCAM:
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

        self.base_cam = base_method(model, target_layer, input_shape=input_shape, **base_kwargs)

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
        """Return the RefineCAM context manager."""  # noqa: DOC201
        return self

    def __exit__(
        self,
        exct_type: type[BaseException] | None,
        exce_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove and reset the base extractor hooks."""
        self.base_cam.__exit__(exct_type, exce_value, traceback)

    def __call__(
        self,
        class_idx: int | list[int],
        scores: Tensor | None = None,
        normalized: bool = True,
        target_shape: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute and refine the per-layer CAMs for an output class."""  # noqa: DOC201
        cams = self.base_cam(class_idx, scores, normalized=True, **kwargs)
        return [self.fuse_cams(cams, target_shape, normalized)]

    def compute_cams(
        self,
        class_idx: int | list[int],
        scores: Tensor | None = None,
        normalized: bool = True,
        target_shape: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> list[Tensor]:
        """Compute and refine CAMs without the base extractor precheck."""  # noqa: DOC201
        cams = self.base_cam.compute_cams(class_idx, scores, normalized=True, **kwargs)
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
