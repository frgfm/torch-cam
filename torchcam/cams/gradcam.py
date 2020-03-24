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

    def __init__(self, model, conv_layer):

        super().__init__(model, conv_layer)
        # Backward hook
        self.model._modules.get(conv_layer).register_backward_hook(self._hook_g)

    def _hook_g(self, module, input, output):
        self.hook_g = output[0].data

    def _backprop(self, output, class_idx):

        if self.hook_a is None:
            raise TypeError("Inputs need to be forwarded in the model for the conv features to be hooked")

        # Backpropagate to get the gradients on the hooked layer
        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def _get_weights(self, output, class_idx):

        raise NotImplementedError

    def __call__(self, output, class_idx, normalized=True):

        # Get map weight
        weights = self._get_weights(output, class_idx)

        # Perform the weighted combination to get the CAM
        batch_cams = torch.relu((weights.view(*weights.shape, 1, 1) * self.hook_a).sum(dim=1))

        # Normalize the CAM
        if normalized:
            batch_cams = self._normalize(batch_cams)

        return batch_cams


class GradCAM(_GradCAM):
    """Implements a class activation map extractor as described in https://arxiv.org/pdf/1710.11063.pdf

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None

    def __init__(self, model, conv_layer):

        super().__init__(model, conv_layer)

    def _get_weights(self, output, class_idx):

        # Backpropagate
        self._backprop(output, class_idx)
        # Global average pool the gradients over spatial dimensions
        return self.hook_g.data.mean(axis=(2, 3))


class GradCAMpp(_GradCAM):
    """Implements a class activation map extractor as described in https://arxiv.org/pdf/1710.11063.pdf

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None

    def __init__(self, model, conv_layer):

        super().__init__(model, conv_layer)

    def _get_weights(self, output, class_idx):

        # Backpropagate
        self._backprop(output, class_idx)
        # Alpha coefficient for each pixel
        grad_2 = self.hook_g.data.pow(2)
        grad_3 = self.hook_g.data.pow(3)
        alpha = grad_2 / (2 * grad_2 + (grad_3 * self.hook_a.data).sum(axis=(2, 3), keepdims=True))

        # Apply pixel coefficient in each weight
        return alpha.mul(torch.relu(self.hook_g.data)).sum(axis=(2, 3))


class SmoothGradCAMpp(_GradCAM):
    """Implements a class activation map extractor as described in https://arxiv.org/pdf/1908.01224.pdf
    with a personal correction to the paper (alpha coefficient numerator)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None

    def __init__(self, model, conv_layer, first_layer, num_samples=4, std=0.3):

        super().__init__(model, conv_layer)

        # Input hook
        self.model._modules.get(first_layer).register_forward_pre_hook(self._store_input)
        # Noise distribution
        self.num_samples = num_samples
        self.std = std
        self._distrib = torch.distributions.normal.Normal(0, self.std)
        self._observing = True

    def _store_input(self, module, input):

        if self._observing:
            self._input = input[0].data.clone()

    def _get_weights(self, output, class_idx):

        # Keep initial activation
        init_fmap = self.hook_a.data
        # Initialize our gradient estimates
        grad_2, grad_3 = torch.zeros_like(self.hook_a.data), torch.zeros_like(self.hook_a.data)
        # Disable input update
        self._observing = False
        # Perform the operations N times
        for _idx in range(self.num_samples):
            # Add noise
            noisy_input = self._input + self._distrib.sample(self._input.size())
            # Forward & Backward
            out = self.model(noisy_input)
            self.model.zero_grad()
            self._backprop(out, class_idx)

            # Sum partial derivatives
            grad_2.add_(self.hook_g.data.pow(2))
            grad_3.add_(self.hook_g.data.pow(3))

        # Reenable input update
        self._observing = True

        # Average the gradient estimates
        grad_2.div_(self.num_samples)
        grad_3.div_(self.num_samples)

        # Alpha coefficient for each pixel
        alpha = grad_2 / (2 * grad_2 + (grad_3 * init_fmap).sum(axis=(2, 3), keepdims=True))

        # Apply pixel coefficient in each weight
        return alpha.mul(torch.relu(self.hook_g.data)).sum(axis=(2, 3))

    def __repr__(self):
        return f"{self.__class__.__name__}(num_samples={self.num_samples}, std={self.std})"
