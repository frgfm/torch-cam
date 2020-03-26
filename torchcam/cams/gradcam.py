#!usr/bin/python
# -*- coding: utf-8 -*-

"""
GradCAM
"""

import torch

from .cam import _CAM

__all__ = ['GradCAM', 'GradCAMpp', 'SmoothGradCAMpp']


class _GradCAM(_CAM):
    """Implements a gradient-based class activation map extractor

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None
    hook_handles = []

    def __init__(self, model, conv_layer):

        super().__init__(model, conv_layer)
        # Ensure ReLU is applied before normalization
        self._relu = True
        # Model output is used by the extractor
        self._score_used = True
        # Backward hook
        self.hook_handles.append(self.model._modules.get(conv_layer).register_backward_hook(self._hook_g))

    def _hook_g(self, module, input, output):
        """Gradient hook"""
        if self._hooks_enabled:
            self.hook_g = output[0].data

    def _backprop(self, scores, class_idx):
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
    """Implements a class activation map extractor as described in https://arxiv.org/pdf/1710.11063.pdf

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import GradCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = GradCAM(model, 'layer4')
        >>> with torch.no_grad(): scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None

    def __init__(self, model, conv_layer):

        super().__init__(model, conv_layer)

    def _get_weights(self, class_idx, scores):
        """Computes the weight coefficients of the hooked activation maps"""

        # Backpropagate
        self._backprop(scores, class_idx)
        # Global average pool the gradients over spatial dimensions
        return self.hook_g.squeeze(0).mean(axis=(1, 2))


class GradCAMpp(_GradCAM):
    """Implements a class activation map extractor as described in https://arxiv.org/pdf/1710.11063.pdf

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import GradCAMpp
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = GradCAMpp(model, 'layer4')
        >>> with torch.no_grad(): scores = model(input_tensor)
        >>> cam(class_idx=100, scores=scores)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None

    def __init__(self, model, conv_layer):

        super().__init__(model, conv_layer)

    def _get_weights(self, class_idx, scores):
        """Computes the weight coefficients of the hooked activation maps"""

        # Backpropagate
        self._backprop(scores, class_idx)
        # Alpha coefficient for each pixel
        grad_2 = self.hook_g.pow(2)
        grad_3 = self.hook_g.pow(3)
        alpha = grad_2 / (2 * grad_2 + (grad_3 * self.hook_a).sum(axis=(2, 3), keepdims=True))

        # Apply pixel coefficient in each weight
        return alpha.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(axis=(1, 2))


class SmoothGradCAMpp(_GradCAM):
    """Implements a class activation map extractor as described in https://arxiv.org/pdf/1908.01224.pdf
    with a personal correction to the paper (alpha coefficient numerator)

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import SmoothGradCAMpp
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = SmoothGradCAMpp(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): scores = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None
    hook_handles = []

    def __init__(self, model, conv_layer, first_layer, num_samples=4, std=0.3):

        super().__init__(model, conv_layer)
        # Model scores is not used by the extractor
        self._score_used = False

        # Input hook
        self.hook_handles.append(self.model._modules.get(first_layer).register_forward_pre_hook(self._store_input))
        # Noise distribution
        self.num_samples = num_samples
        self.std = std
        self._distrib = torch.distributions.normal.Normal(0, self.std)
        # Specific input hook updater
        self._ihook_enabled = True

    def _store_input(self, module, input):
        """Store model input tensor"""

        if self._ihook_enabled:
            self._input = input[0].data.clone()

    def _get_weights(self, class_idx, scores=None):
        """Computes the weight coefficients of the hooked activation maps"""

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
        alpha = grad_2 / (2 * grad_2 + (grad_3 * init_fmap).sum(axis=(2, 3), keepdims=True))

        # Apply pixel coefficient in each weight
        return alpha.squeeze_(0).mul_(torch.relu(self.hook_g.squeeze(0))).sum(axis=(1, 2))

    def __repr__(self):
        return f"{self.__class__.__name__}(num_samples={self.num_samples}, std={self.std})"
