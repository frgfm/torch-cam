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
        self.smax_weights = self.model._modules.get(fc_layer).weight.data

    def _hook_a(self, module, input, output):
        self.hook_a = output.data

    def get_activation_maps(self, class_idx, normalized=True):
        """Recreate class activation maps

        Args:
            class_idx (int): class index for expected activation map
            normalized (bool, optional): should the activation map be normalized

        Returns:
            torch.Tensor[N, H, W]: activation maps of the last forwarded batch at the hooked layer
        """

        if class_idx >= self.smax_weights.shape[0]:
            raise ValueError("Expected class_idx to be lower than number of output classes")

        if self.hook_a is None:
            raise TypeError("Inputs need to be forwarded in the model for the conv features to be hooked")

        # Flatten spatial dimensions of feature map
        batch_cams = self.smax_weights[class_idx, :].unsqueeze(0) @ torch.flatten(self.hook_a, 2)
        # Normalize feature map
        if normalized:
            batch_cams -= batch_cams.min(dim=2, keepdim=True)[0]
            batch_cams /= batch_cams.max(dim=2, keepdim=True)[0]

        return batch_cams.view(self.hook_a.size(0), self.hook_a.size(3), self.hook_a.size(2))
