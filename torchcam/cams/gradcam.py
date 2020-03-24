#!usr/bin/python
# -*- coding: utf-8 -*-

"""
GradCAM
"""

import torch
from .cam import _CAM


__all__ = ['GradCAM', 'GradCAMpp']


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

        # Backpropagate
        self._backprop(output, class_idx)

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

        # Alpha coefficient for each pixel
        grad_2 = self.hook_g.data.pow(2)
        grad_3 = self.hook_g.data.pow(3)
        alpha = grad_2 / (2 * grad_2 + (grad_3 * self.hook_a.data).sum(axis=(2, 3), keepdims=True))

        # Apply pixel coefficient in each weight
        return alpha.mul(torch.relu(self.hook_g.data)).sum(axis=(2, 3))
