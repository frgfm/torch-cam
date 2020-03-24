#!usr/bin/python
# -*- coding: utf-8 -*-

"""
GradCAM
"""

import torch


__all__ = ['CAM']


class CAM(object):
    """Implements a class activation map extractor as described in https://arxiv.org/abs/1512.04150

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a = None

    def __init__(self, model, conv_layer, fc_layer):

        if not hasattr(model, conv_layer):
            raise ValueError(f"Unable to find submodule {conv_layer} in the model")
        self.model = model
        # Forward hook
        self.model._modules.get(conv_layer).register_forward_hook(self._hook_a)
        # Softmax weight
        self._fc_weights = self.model._modules.get(fc_layer).weight.data

    def _hook_a(self, module, input, output):
        self.hook_a = output.data

    def _compute_cams(self, weights, normalized=True):

        # Perform the weighted combination to get the CAM
        batch_cams = (weights.view(-1, 1, 1) * self.hook_a).sum(dim=1)

        # Normalize the CAM
        if normalized:
            batch_cams -= batch_cams.flatten(start_dim=1).min().view(-1, 1, 1)
            batch_cams /= batch_cams.flatten(start_dim=1).max().view(-1, 1, 1)

        return batch_cams

    def get_activation_maps(self, class_idx, normalized=True):
        """Recreate class activation maps

        Args:
            class_idx (int): class index for expected activation map
            normalized (bool, optional): should the activation map be normalized

        Returns:
            torch.Tensor[N, H, W]: activation maps of the last forwarded batch at the hooked layer
        """

        if class_idx >= self._fc_weights.shape[0]:
            raise ValueError("Expected class_idx to be lower than number of output classes")

        if self.hook_a is None:
            raise TypeError("Inputs need to be forwarded in the model for the conv features to be hooked")

        # Take the FC weights of the target class
        weights = self._fc_weights[class_idx, :]

        # Assemble CAMs
        return self._compute_cams(weights, normalized)
