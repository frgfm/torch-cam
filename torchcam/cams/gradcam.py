import torch
from torch import Tensor
from typing import Optional, Tuple

from .cam import _CAM

__all__ = ['GradCAM', 'GradCAMpp', 'SmoothGradCAMpp']


class _GradCAM(_CAM):
    """Implements a gradient-based class activation map extractor

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[str] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
    ) -> None:

        super().__init__(model, target_layer, input_shape)
        # Init hook
        self.hook_g: Optional[Tensor] = None
        # Ensure ReLU is applied before normalization
        self._relu = True
        # Model output is used by the extractor
        self._score_used = True
        # Backward hook
        self.hook_handles.append(self.submodule_dict[self.target_layer].register_backward_hook(self._hook_g))

    def _hook_g(self, module: torch.nn.Module, input: Tensor, output: Tensor) -> None:
        """Gradient hook"""
        if self._hooks_enabled:
            self.hook_g = output[0].data

    def _backprop(self, scores: Tensor, class_idx: int) -> None:
        """Backpropagate the loss for a specific output class"""

        if self.hook_a is None:
            raise TypeError("Inputs need to be forwarded in the model for the conv features to be hooked")

        # Backpropagate to get the gradients on the hooked layer
        loss = scores[:, class_idx].sum()
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def _get_weights(self, class_idx, scores):

        raise NotImplementedError


class GradCAM(_GradCAM):
    """Implements a class activation map extractor as described in `"Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" <https://arxiv.org/pdf/1610.02391.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Grad-CAM}(x, y) = ReLU\\Big(\\sum\\limits_k w_k^{(c)} A_k(x, y)\\Big)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\frac{1}{H \\cdot W} \\sum\\limits_{i=1}^H \\sum\\limits_{j=1}^W
        \\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    and :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import GradCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = GradCAM(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(self, class_idx: int, scores: Tensor) -> Tensor:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""

        self.hook_g: Tensor
        # Backpropagate
        self._backprop(scores, class_idx)
        # Global average pool the gradients over spatial dimensions
        return self.hook_g.squeeze(0).mean(dim=(1, 2))


class GradCAMpp(_GradCAM):
    """Implements a class activation map extractor as described in `"Grad-CAM++: Improved Visual Explanations for
    Deep Convolutional Networks" <https://arxiv.org/pdf/1710.11063.pdf>`_.

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Grad-CAM++}(x, y) = \\sum\\limits_k w_k^{(c)} A_k(x, y)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\sum\\limits_{i=1}^H \\sum\\limits_{j=1}^W \\alpha_k^{(c)}(i, j) \\cdot
        ReLU\\Big(\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}\\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax,
    and :math:`\\alpha_k^{(c)}(i, j)` being defined as:

    .. math::
        \\alpha_k^{(c)}(i, j) = \\frac{1}{\\sum\\limits_{i, j} \\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}}
        = \\frac{\\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2}}{2 \\cdot
        \\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2} + \\sum\\limits_{a,b} A_k (a,b) \\cdot
        \\frac{\\partial^3 Y^{(c)}}{(\\partial A_k(i,j))^3}}

    if :math:`\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)} = 1` else :math:`0`.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import GradCAMpp
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = GradCAMpp(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model: input model
        target_layer: name of the target layer
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def _get_weights(self, class_idx: int, scores: Tensor) -> Tensor:  # type: ignore[override]
        """Computes the weight coefficients of the hooked activation maps"""

        self.hook_g: Tensor
        # Backpropagate
        self._backprop(scores, class_idx)
        # Alpha coefficient for each pixel
        grad_2 = self.hook_g.pow(2)
        grad_3 = grad_2 * self.hook_g
        # Watch out for NaNs produced by underflow
        denom = 2 * grad_2 + (grad_3 * self.hook_a).sum(dim=(2, 3), keepdim=True)
        nan_mask = grad_2 > 0
        alpha = grad_2
        alpha[nan_mask].div_(denom[nan_mask])

        # Apply pixel coefficient in each weight
        return alpha.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(dim=(1, 2))


class SmoothGradCAMpp(_GradCAM):
    """Implements a class activation map extractor as described in `"Smooth Grad-CAM++: An Enhanced Inference Level
    Visualization Technique for Deep Convolutional Neural Network Models" <https://arxiv.org/pdf/1908.01224.pdf>`_
    with a personal correction to the paper (alpha coefficient numerator).

    The localization map is computed as follows:

    .. math::
        L^{(c)}_{Smooth Grad-CAM++}(x, y) = \\sum\\limits_k w_k^{(c)} A_k(x, y)

    with the coefficient :math:`w_k^{(c)}` being defined as:

    .. math::
        w_k^{(c)} = \\sum\\limits_{i=1}^H \\sum\\limits_{j=1}^W \\alpha_k^{(c)}(i, j) \\cdot
        ReLU\\Big(\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)}\\Big)

    where :math:`A_k(x, y)` is the activation of node :math:`k` in the target layer of the model at
    position :math:`(x, y)`,
    :math:`Y^{(c)}` is the model output score for class :math:`c` before softmax,
    and :math:`\\alpha_k^{(c)}(i, j)` being defined as:

    .. math::
        \\alpha_k^{(c)}(i, j)
        = \\frac{\\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2}}{2 \\cdot
        \\frac{\\partial^2 Y^{(c)}}{(\\partial A_k(i,j))^2} + \\sum\\limits_{a,b} A_k (a,b) \\cdot
        \\frac{\\partial^3 Y^{(c)}}{(\\partial A_k(i,j))^3}}
        = \\frac{\\frac{1}{n} \\sum\\limits_{m=1}^n D^{(c, 2)}_k(i, j)}{
        \\frac{2}{n} \\sum\\limits_{m=1}^n D^{(c, 2)}_k(i, j) + \\sum\\limits_{a,b} A_k (a,b) \\cdot
        \\frac{1}{n} \\sum\\limits_{m=1}^n D^{(c, 3)}_k(i, j)}

    if :math:`\\frac{\\partial Y^{(c)}}{\\partial A_k(i, j)} = 1` else :math:`0`. Here :math:`D^{(c, p)}_k(i, j)`
    refers to the p-th partial derivative of the class score of class :math:`c` relatively to the activation in layer
    :math:`k` at position :math:`(i, j)`, and :math:`n` is the number of samples used to get the gradient estimate.

    Please note the difference in the numerator of :math:`\\alpha_k^{(c)}(i, j)`,
    which is actually :math:`\\frac{1}{n} \\sum\\limits_{k=1}^n D^{(c, 1)}_k(i,j)` in the paper.

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import SmoothGradCAMpp
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = SmoothGradCAMpp(model, 'layer4')
        >>> scores = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model: input model
        target_layer: name of the target layer
        num_samples: number of samples to use for smoothing
        std: standard deviation of the noise
        input_shape: shape of the expected input tensor excluding the batch dimension
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[str] = None,
        num_samples: int = 4,
        std: float = 0.3,
        input_shape: Tuple[int, ...] = (3, 224, 224),
    ) -> None:

        super().__init__(model, target_layer, input_shape)
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

    def _store_input(self, module: torch.nn.Module, input: Tensor) -> None:
        """Store model input tensor"""

        if self._ihook_enabled:
            self._input = input[0].data.clone()

    def _get_weights(self, class_idx: int, scores: Optional[Tensor] = None) -> Tensor:
        """Computes the weight coefficients of the hooked activation maps"""

        self.hook_a: Tensor
        self.hook_g: Tensor
        # Disable input update
        self._ihook_enabled = False
        # Keep initial activation
        init_fmap = self.hook_a.clone()
        # Initialize our gradient estimates
        grad_2, grad_3 = torch.zeros_like(self.hook_a), torch.zeros_like(self.hook_a)
        # Perform the operations N times
        for _idx in range(self.num_samples):
            # Add noise
            noisy_input = self._input + self._distrib.sample(self._input.size()).to(device=self._input.device)
            # Forward & Backward
            out = self.model(noisy_input)
            self.model.zero_grad()
            self._backprop(out, class_idx)

            # Sum partial derivatives
            grad_2.add_(self.hook_g.pow(2))
            grad_3.add_(self.hook_g.pow(3))

        # Reenable input update
        self._ihook_enabled = True

        # Average the gradient estimates
        grad_2.div_(self.num_samples)
        grad_3.div_(self.num_samples)

        # Alpha coefficient for each pixel
        alpha = grad_2 / (2 * grad_2 + (grad_3 * init_fmap).sum(dim=(2, 3), keepdim=True))

        # Apply pixel coefficient in each weight
        return alpha.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(dim=(1, 2))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_samples={self.num_samples}, std={self.std})"
