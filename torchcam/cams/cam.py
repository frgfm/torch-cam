#!usr/bin/python
# -*- coding: utf-8 -*-

"""
CAM
"""

import math
import torch
import torch.nn.functional as F

__all__ = ['CAM', 'ScoreCAM']


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
        # Enable hooks
        self._hooks_enabled = True

    def _hook_a(self, module, input, output):
        if self._hooks_enabled:
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

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import CAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = CAM(model, 'layer4', 'fc')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

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


class ScoreCAM(_CAM):
    """Implements a class activation map extractor as described in https://arxiv.org/abs/1910.01279

    Example::
        >>> from torchvision.models import resnet18
        >>> from torchcam.cams import ScoreCAM
        >>> model = resnet18(pretrained=True).eval()
        >>> cam = ScoreCAM(model, 'layer4', 'conv1')
        >>> with torch.no_grad(): out = model(input_tensor)
        >>> cam(class_idx=100)

    Args:
        model (torch.nn.Module): input model
        conv_layer (str): name of the last convolutional layer
    """

    hook_a = None
    hook_handles = []

    def __init__(self, model, conv_layer, input_layer, max_batch=32):

        super().__init__(model, conv_layer)

        # Input hook
        self.hook_handles.append(self.model._modules.get(input_layer).register_forward_pre_hook(self._store_input))
        self.max_batch = max_batch

    def _store_input(self, module, input):

        if self._hooks_enabled:
            self._input = input[0].data.clone()

    def _get_weights(self, class_idx):

        # Upsample activation to input_size
        # 1 * O * M * N
        upsampled_a = F.interpolate(self.hook_a, self._input.shape[-2:], mode='bilinear', align_corners=False)

        # Normalize it
        upsampled_a = self._normalize(upsampled_a)

        # Use it as a mask
        # O * I * H * W
        masked_input = upsampled_a.squeeze(0).unsqueeze(1) * self._input

        # Initialize weights
        weights = torch.zeros(masked_input.shape[0], dtype=masked_input.dtype).to(device=masked_input.device)

        # Disable hook updates
        self._hooks_enabled = False
        # Process by chunk (GPU RAM limitation)
        for idx in range(math.ceil(weights.shape[0] / self.max_batch)):

            selection_slice = slice(idx * self.max_batch, min((idx + 1) * self.max_batch, weights.shape[0]))
            with torch.no_grad():
                # Get the softmax probabilities of the target class
                weights[selection_slice] = F.softmax(self.model(masked_input[selection_slice]), dim=1)[:, class_idx]

        # Reenable hook updates
        self._hooks_enabled = True

        return weights

    def __call__(self, class_idx, normalized=True):

        # Get map weight
        weights = self._get_weights(class_idx)

        # Perform the weighted combination to get the CAM
        batch_cams = torch.relu((weights.view(-1, 1, 1) * self.hook_a).sum(dim=1))

        # Normalize the CAM
        if normalized:
            batch_cams = self._normalize(batch_cams)

        return batch_cams
