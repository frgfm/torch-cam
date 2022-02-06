# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from functools import partial
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .core import _CAM

__all__ = ['GradCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'XGradCAM', 'LayerCAM']


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
        target_layer: Optional[Union[Union[nn.Module, str], List[Union[nn.Module, str]]]] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
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

    def _store_grad(self, grad: Tensor, idx: int = 0) -> None:
        if self._hooks_enabled:
            self.hook_g[idx] = grad.data  # type: ignore[call-overload]

    def _hook_g(self, module: nn.Module, input: Tensor, output: Tensor, idx: int = 0) -> None:
        """Gradient hook"""
        if self._hooks_enabled:
            self.hook_handles.append(output.register_hook(partial(self._store_grad, idx=idx)))

    def _backprop(self, scores: Tensor, class_idx: Union[int, List[int]], retain_graph: bool = False) -> None:
        """Backpropagate the loss for a specific output class"""

        # Backpropagate to get the gradients on the hooked layer
        if isinstance(class_idx, int):
            loss = scores[:, class_idx].sum()
        else:
            loss = scores.gather(1, torch.tensor(class_idx, device=scores.device).view(-1, 1)).sum()
        self.model.zero_grad()
        loss.backward(retain_graph=retain_graph)

    def _get_weights(self, class_idx: Union[int, List[int]], scores: Tensor, **kwargs: Any) -> List[Tensor]:

        raise NotImplementedError


class GradCAM(_GradCAM):
    r"""Implements a class activation map extractor as described in `"Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" <https://arxiv.org/pdf/1610.02391.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Grad-CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \frac{1}{H \cdot W} \sum\limits_{i=1}^H \sum\limits_{j=1}^W
        \frac{\partial Y^{(c)}}{\partial A_k(i, j)}

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    and :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.methods import GradCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = GradCAM(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(self, class_idx: Union[int, List[int]], scores: Tensor, **kwargs: Any) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""

        # Backpropagate
        self._backprop(scores, class_idx, **kwargs)

        self.hook_g: List[Tensor]  # type: ignore[assignment]
        # Global average pool the gradients over spatial dimensions
        return [grad.flatten(2).mean(-1) for grad in self.hook_g]  # type: ignore[attr-defined]


class GradCAMpp(_GradCAM):
    r"""Implements a class activation map extractor as described in `"Grad-CAM++: Improved Visual Explanations for
    Deep Convolutional Networks" <https://arxiv.org/pdf/1710.11063.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Grad-CAM++}(x, y) = \sum\limits_k w_k^{(c)} A_k(x, y)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \sum\limits_{i=1}^H \sum\limits_{j=1}^W \alpha_k^{(c)}(i, j) \cdot
        ReLU\Big(\frac{\partial Y^{(c)}}{\partial A_k(i, j)}\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax,
    and :math:`\alpha_k^{(c)}(i, j)` being defined as:

    .. math::
        \alpha_k^{(c)}(i, j) = \frac{1}{\sum\limits_{i, j} \frac{\partial Y^{(c)}}{\partial A_k(i, j)}}
        = \frac{\frac{\partial^2 Y^{(c)}}{(\partial A_k(i,j))^2}}{2 \cdot
        \frac{\partial^2 Y^{(c)}}{(\partial A_k(i,j))^2} + \sum\limits_{a,b} A_k (a,b) \cdot
        \frac{\partial^3 Y^{(c)}}{(\partial A_k(i,j))^3}}

    if :math:`\frac{\partial Y^{(c)}}{\partial A_k(i, j)} = 1` else :math:`0`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.methods import GradCAMpp
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = GradCAMpp(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(self, class_idx: Union[int, List[int]], scores: Tensor, **kwargs: Any) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""

        # Backpropagate
        self._backprop(scores, class_idx, **kwargs)
        self.hook_a: List[Tensor]  # type: ignore[assignment]
        self.hook_g: List[Tensor]  # type: ignore[assignment]
        # Alpha coefficient for each pixel
        grad_2 = [grad.pow(2) for grad in self.hook_g]
        grad_3 = [g2 * grad for g2, grad in zip(grad_2, self.hook_g)]
        # Watch out for NaNs produced by underflow
        spatial_dims = self.hook_a[0].ndim - 2
        denom = [
            2 * g2 + (g3 * act).flatten(2).sum(-1)[(...,) + (None,) * spatial_dims]
            for g2, g3, act in zip(grad_2, grad_3, self.hook_a)
        ]
        nan_mask = [g2 > 0 for g2 in grad_2]
        alpha = grad_2
        for idx, d, mask in zip(range(len(grad_2)), denom, nan_mask):
            alpha[idx][mask].div_(d[mask])

        # Apply pixel coefficient in each weight
        return [
            a.mul_(torch.relu(grad)).flatten(2).sum(-1)
            for a, grad in zip(alpha, self.hook_g)
        ]


class SmoothGradCAMpp(_GradCAM):
    r"""Implements a class activation map extractor as described in `"Smooth Grad-CAM++: An Enhanced Inference Level
    Visualization Technique for Deep Convolutional Neural Network Models" <https://arxiv.org/pdf/1908.01224.pdf>`_
    with a personal correction to the paper (alpha coefficient numerator).

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Smooth Grad-CAM++}(x, y) = \sum\limits_k w_k^{(c)} A_k(x, y)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \sum\limits_{i=1}^H \sum\limits_{j=1}^W \alpha_k^{(c)}(i, j) \cdot
        ReLU\Big(\frac{\partial Y^{(c)}}{\partial A_k(i, j)}\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax,
    and :math:`\alpha_k^{(c)}(i, j)` being defined as:

    .. math::
        \alpha_k^{(c)}(i, j)
        = \frac{\frac{\partial^2 Y^{(c)}}{(\partial A_k(i,j))^2}}{2 \cdot
        \frac{\partial^2 Y^{(c)}}{(\partial A_k(i,j))^2} + \sum\limits_{a,b} A_k (a,b) \cdot
        \frac{\partial^3 Y^{(c)}}{(\partial A_k(i,j))^3}}
        = \frac{\frac{1}{n} \sum\limits_{m=1}^n D^{(c, 2)}_k(i, j)}{
        \frac{2}{n} \sum\limits_{m=1}^n D^{(c, 2)}_k(i, j) + \sum\limits_{a,b} A_k (a,b) \cdot
        \frac{1}{n} \sum\limits_{m=1}^n D^{(c, 3)}_k(i, j)}

    if :math:`\frac{\partial Y^{(c)}}{\partial A_k(i, j)} = 1` else :math:`0`. Here :math:`D^{(c, p)}_k(i, j)`
    refers to the p-th partial derivative of the class score of class :math:`c` relatively to the activation in layer
    :math:`k` at position :math:`(i, j)`, and :math:`n` is the number of samples used to get the gradient estimate.

    Please note the difference in the numerator of :math:`\alpha_k^{(c)}(i, j)`,
    which is actually :math:`\frac{1}{n} \sum\limits_{k=1}^n D^{(c, 1)}_k(i,j)` in the paper.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.methods import SmoothGradCAMpp
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = SmoothGradCAMpp(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100)

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
        target_layer: Optional[Union[Union[nn.Module, str], List[Union[nn.Module, str]]]] = None,
        num_samples: int = 4,
        std: float = 0.3,
        input_shape: Tuple[int, ...] = (3, 224, 224),
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

    def _store_input(self, module: nn.Module, input: Tensor) -> None:
        """Store model input tensor."""

        if self._ihook_enabled:
            self._input = input[0].data.clone()

    def _get_weights(
        self,
        class_idx: Union[int, List[int]],
        scores: Optional[Tensor] = None,
        **kwargs: Any
    ) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""

        # Disable input update
        self._ihook_enabled = False
        # Keep initial activation
        self.hook_a: List[Tensor]  # type: ignore[assignment]
        self.hook_g: List[Tensor]  # type: ignore[assignment]
        init_fmap = [act.clone() for act in self.hook_a]
        # Initialize our gradient estimates
        grad_2 = [torch.zeros_like(act) for act in self.hook_a]
        grad_3 = [torch.zeros_like(act) for act in self.hook_a]
        # Perform the operations N times
        for _idx in range(self.num_samples):
            # Add noise
            noisy_input = self._input + self._distrib.sample(self._input.size()).to(device=self._input.device)
            # Forward & Backward
            out = self.model(noisy_input)
            self.model.zero_grad()
            self._backprop(out, class_idx, **kwargs)

            # Sum partial derivatives
            grad_2 = [g2.add_(grad.pow(2)) for g2, grad in zip(grad_2, self.hook_g)]
            grad_3 = [g3.add_(grad.pow(3)) for g3, grad in zip(grad_3, self.hook_g)]

        # Reenable input update
        self._ihook_enabled = True

        # Average the gradient estimates
        grad_2 = [g2.div_(self.num_samples) for g2 in grad_2]
        grad_3 = [g3.div_(self.num_samples) for g3 in grad_3]

        # Alpha coefficient for each pixel
        spatial_dims = self.hook_a[0].ndim - 2  # type: ignore[attr-defined]
        alpha = [
            g2 / (2 * g2 + (g3 * act).flatten(2).sum(-1)[(...,) + (None,) * spatial_dims])
            for g2, g3, act in zip(grad_2, grad_3, init_fmap)
        ]

        # Apply pixel coefficient in each weight
        return [
            a.mul_(torch.relu(grad)).flatten(2).sum(-1)
            for a, grad in zip(alpha, self.hook_g)
        ]

    def extra_repr(self) -> str:
        return f"target_layer={self.target_names}, num_samples={self.num_samples}, std={self.std}"


class XGradCAM(_GradCAM):
    r"""Implements a class activation map extractor as described in `"Axiom-based Grad-CAM: Towards Accurate
    Visualization and Explanation of CNNs" <https://arxiv.org/pdf/2008.02312.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{XGrad-CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)} A_k(x, y)\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \sum\limits_{i=1}^H \sum\limits_{j=1}^W
        \Big( \frac{\partial Y^{(c)}}{\partial A_k(i, j)} \cdot
        \frac{A_k(i, j)}{\sum\limits_{m=1}^H \sum\limits_{n=1}^W A_k(m, n)} \Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    and :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.methods import XGradCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = XGradCAM(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(self, class_idx: Union[int, List[int]], scores: Tensor, **kwargs: Any) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""

        # Backpropagate
        self._backprop(scores, class_idx, **kwargs)

        self.hook_a: List[Tensor]  # type: ignore[assignment]
        self.hook_g: List[Tensor]  # type: ignore[assignment]
        return [
            (grad * act).flatten(2).sum(-1) / act.flatten(2).sum(-1)
            for act, grad in zip(self.hook_a, self.hook_g)
        ]


class LayerCAM(_GradCAM):
    r"""Implements a class activation map extractor as described in `"LayerCAM: Exploring Hierarchical Class Activation
    Maps for Localization" <http://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Layer-CAM}(x, y) = ReLU\Big(\sum\limits_k w_k^{(c)}(x, y) \cdot A_k(x, y)\Big)

    with the coefficient :math:`w_k^{(c)}(x, y)` being defined as:

    .. math::
        w_k^{(c)}(x, y) = ReLU\Big(\frac{\partial Y^{(c)}}{\partial A_k(i, j)}(x, y)\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    and :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.methods import LayerCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> extractor = LayerCAM(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cams = extractor(class_idx=100, scores=scores)
        >>> fused_cam = extractor.fuse_cams(cams)

    Args:
        model: input model
        target_layer: either the target layer itself or its name, or a list of those
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(self, class_idx: Union[int, List[int]], scores: Tensor, **kwargs: Any) -> List[Tensor]:
        """Computes the weight coefficients of the hooked activation maps."""

        # Backpropagate
        self._backprop(scores, class_idx, **kwargs)

        self.hook_g: List[Tensor]  # type: ignore[assignment]
        # List of (N, C, H, W)
        return [torch.relu(grad) for grad in self.hook_g]

    @staticmethod
    def _scale_cams(cams: List[Tensor], gamma: float = 2.) -> List[Tensor]:
        # cf. Equation 9 in the paper
        return [torch.tanh(gamma * cam) for cam in cams]
