#!usr/bin/python
# -*- coding: utf-8 -*-

"""
GradCAM
"""

import torch


__all__ = ['GradCAM', 'GradCAMpp']


class _GradCAM(object):
    """Implements a class activation map extractor as described in https://arxiv.org/pdf/1610.02391.pdf

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None

    def __init__(self, model, conv_layer):

        if not hasattr(model, conv_layer):
            raise ValueError(f"Unable to find submodule {conv_layer} in the model")
        self.model = model
        # Forward hook
        self.model._modules.get(conv_layer).register_forward_hook(self._hook_a)
        # Backward hook
        self.model._modules.get(conv_layer).register_backward_hook(self._hook_g)

    def _hook_a(self, module, input, output):
        self.hook_a = output.data

    def _hook_g(self, module, input, output):
        self.hook_g = output[0].data

    def _compute_gradcams(self, weights, normalized=True):

        # Perform the weighted combination to get the CAM
        batch_cams = torch.relu((weights.view(*weights.shape, 1, 1) * self.hook_a).sum(dim=1))

        # Normalize the CAM
        if normalized:
            batch_cams -= batch_cams.flatten(start_dim=1).min().view(-1, 1, 1)
            batch_cams /= batch_cams.flatten(start_dim=1).max().view(-1, 1, 1)

        return batch_cams

    def _backprop(self, output, class_idx):

        if self.hook_a is None:
            raise TypeError("Inputs need to be forwarded in the model for the conv features to be hooked")

        # Backpropagate to get the gradients on the hooked layer
        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

    def get_activation_maps(self, output, class_idx, normalized=True):
        """Class activation map computation"""

        raise NotImplementedError


class GradCAM(_GradCAM):
    """Implements a class activation map extractor as described in https://arxiv.org/pdf/1710.11063.pdf

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None

    def __init__(self, model, conv_layer):

        super().__init__(model, conv_layer)

    def get_activation_maps(self, output, class_idx, normalized=True):
        """Recreate class activation maps

        Args:
            output (torch.Tensor[N, K]): output of the hooked model
            class_idx (int): class index for expected activation map
            normalized (bool, optional): should the activation map be normalized

        Returns:
            torch.Tensor[N, H, W]: activation maps of the last forwarded batch at the hooked layer
        """

        # Retrieve the activation and gradients of the target layer
        self._backprop(output, class_idx)

        # Global average pool the gradients over spatial dimensions
        weights = self.hook_g.data.mean(axis=(2, 3))

        # Assemble the CAM
        return self._compute_gradcams(weights, normalized)


class GradCAMpp(_GradCAM):
    """Implements a class activation map extractor as described in https://arxiv.org/pdf/1710.11063.pdf

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a, hook_g = None, None

    def __init__(self, model, conv_layer):

        super().__init__(model, conv_layer)

    def get_activation_maps(self, output, class_idx, normalized=True):
        """Recreate class activation maps

        Args:
            output (torch.Tensor[N, K]): output of the hooked model
            class_idx (int): class index for expected activation map
            normalized (bool, optional): should the activation map be normalized

        Returns:
            torch.Tensor[N, H, W]: activation maps of the last forwarded batch at the hooked layer
        """

        # Retrieve the activation and gradients of the target layer
        self._backprop(output, class_idx)

        # Alpha coefficient for each pixel
        grad_2 = self.hook_g.data.pow(2)
        grad_3 = self.hook_g.data.pow(3)
        alpha = grad_2 / (2 * grad_2 + (grad_3 * self.hook_a.data).sum(axis=(2, 3), keepdims=True))
        # Apply pixel coefficient in each weight
        weights = alpha.mul(torch.relu(self.hook_g.data)).sum(axis=(2, 3))

        # Assemble the CAM
        return self._compute_gradcams(weights, normalized)
