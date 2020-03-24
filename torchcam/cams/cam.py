#!usr/bin/python
# -*- coding: utf-8 -*-

"""
GradCAM
"""

import torch

__all__ = ['CAM']


class _CAM(object):
    """Implements a class activation map extractor

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer):

        if not hasattr(model, conv_layer):
            raise ValueError(f"Unable to find submodule {conv_layer} in the model")
        self.model = model
        # Forward hook
        self.hook_handles.append(self.model._modules.get(conv_layer).register_forward_hook(self._hook_a))

    def _hook_a(self, module, input, output):
        self.hook_a = output.data

    def clear_hooks(self):
        """Clear model hooks"""
        for handle in self.hook_handles:
            handle.remove()

    @staticmethod
    def _normalize(cams):
        cams -= cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1)
        cams /= cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1)

        return cams

    def _get_weights(self, class_idx):

        raise NotImplementedError

    def __call__(self, class_idx, normalized=True):

        # Get map weight
        weights = self._get_weights(class_idx)

        # Perform the weighted combination to get the CAM
        batch_cams = (weights.view(-1, 1, 1) * self.hook_a).sum(dim=1)

        # Normalize the CAM
        if normalized:
            batch_cams = self._normalize(batch_cams)

        return batch_cams

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class CAM(_CAM):
    """Implements a class activation map extractor as described in https://arxiv.org/abs/1512.04150

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, fc_layer):

        super().__init__(model, conv_layer)
        # Softmax weight
        self._fc_weights = self.model._modules.get(fc_layer).weight.data

    def _get_weights(self, class_idx):

        # Take the FC weights of the target class
        return self._fc_weights[class_idx, :]
