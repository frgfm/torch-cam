# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import logging
from functools import partial
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union

from .utils import locate_candidate_layer

__all__ = ['_CAM']


class _CAM:
    """Implements a class activation map extractor

    Args:
        model: input model
        target_layer: either the target layer itself or its name
        input_shape: shape of the expected input tensor excluding the batch dimension
        enable_hooks: should hooks be enabled by default
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[Union[nn.Module, str]] = None,
        input_shape: Tuple[int, ...] = (3, 224, 224),
        enable_hooks: bool = True,
    ) -> None:

        # Obtain a mapping from module name to module instance for each layer in the model
        self.submodule_dict = dict(model.named_modules())

        if isinstance(target_layer, str):
            target_names = [target_layer]
        elif isinstance(target_layer, nn.Module):
            # Find the location of the module
            target_names = [self._resolve_layer_name(target_layer)]
        elif isinstance(target_layer, list):
            if any(not isinstance(elt, (str, nn.Module)) for elt in target_layer):
                raise TypeError("invalid argument type for `target_layer`")
            target_names = [
                self._resolve_layer_name(layer) if isinstance(layer, nn.Module) else layer
                for layer in target_layer
            ]
        elif target_layer is None:
            # If the layer is not specified, try automatic resolution
            target_name = locate_candidate_layer(model, input_shape)  # type: ignore[assignment]
            # Warn the user of the choice
            if isinstance(target_name, str):
                logging.warning(f"no value was provided for `target_layer`, thus set to '{target_name}'.")
                target_names = [target_name]
            else:
                raise ValueError("unable to resolve `target_layer` automatically, please specify its value.")
        else:
            raise TypeError("invalid argument type for `target_layer`")

        if any(name not in self.submodule_dict.keys() for name in target_names):
            raise ValueError(f"Unable to find all submodules {target_names} in the model")
        self.target_names = target_names
        self.model = model
        # Init hooks
        self.hook_a: List[Tensor] = [None] * len(target_names)  # type: ignore[list-item]
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        # Forward hook
        for idx, name in enumerate(self.target_names):
            self.hook_handles.append(self.submodule_dict[name].register_forward_hook(partial(self._hook_a, idx=idx)))
        # Enable hooks
        self._hooks_enabled = enable_hooks
        # Should ReLU be used before normalization
        self._relu = False
        # Model output is used by the extractor
        self._score_used = False

    def _resolve_layer_name(self, target_layer: nn.Module) -> str:
        """Resolves the name of a given layer inside the hooked model"""
        _found = False
        for k, v in self.submodule_dict.items():
            if id(v) == id(target_layer):
                target_name = k
                _found = True
                break
        if not _found:
            raise ValueError("unable to locate module inside the specified model.")

        return target_name

    def _hook_a(self, module: nn.Module, input: Tensor, output: Tensor, idx: int = 0) -> None:
        """Activation hook"""
        if self._hooks_enabled:
            self.hook_a[idx] = output.data

    def clear_hooks(self) -> None:
        """Clear model hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    @staticmethod
    def _normalize(cams: Tensor, spatial_dims: Optional[int] = None) -> Tensor:
        """CAM normalization"""
        spatial_dims = cams.ndim if spatial_dims is None else spatial_dims
        cams.sub_(cams.flatten(start_dim=-spatial_dims).min(-1).values[(...,) + (None,) * spatial_dims])
        cams.div_(cams.flatten(start_dim=-spatial_dims).max(-1).values[(...,) + (None,) * spatial_dims])

        return cams

    def _get_weights(self, class_idx: int, scores: Optional[Tensor] = None) -> List[Tensor]:

        raise NotImplementedError

    def _precheck(self, class_idx: int, scores: Optional[Tensor] = None) -> None:
        """Check for invalid computation cases"""

        for fmap in self.hook_a:
            # Check that forward has already occurred
            if not isinstance(fmap, Tensor):
                raise AssertionError("Inputs need to be forwarded in the model for the conv features to be hooked")
            # Check batch size
            if fmap.shape[0] != 1:
                raise ValueError(f"expected a 1-sized batch to be hooked. Received: {fmap.shape[0]}")

        # Check class_idx value
        if not isinstance(class_idx, int) or class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

        # Check scores arg
        if self._score_used and not isinstance(scores, torch.Tensor):
            raise ValueError("model output scores is required to be passed to compute CAMs")

    def __call__(self, class_idx: int, scores: Optional[Tensor] = None, normalized: bool = True) -> List[Tensor]:

        # Integrity check
        self._precheck(class_idx, scores)

        # Compute CAM
        return self.compute_cams(class_idx, scores, normalized)

    def compute_cams(self, class_idx: int, scores: Optional[Tensor] = None, normalized: bool = True) -> List[Tensor]:
        """Compute the CAM for a specific output class

        Args:
            class_idx: output class index of the target class whose CAM will be computed
            scores: forward output scores of the hooked model
            normalized: whether the CAM should be normalized

        Returns:
            List[Tensor]: list of class activation maps, one for each hooked layer
        """

        # Get map weight & unsqueeze it
        weights = self._get_weights(class_idx, scores)

        cams: List[Tensor] = []

        for weight, activation in zip(weights, self.hook_a):
            missing_dims = activation.ndim - weight.ndim - 1  # type: ignore[union-attr]
            weight = weight[(...,) + (None,) * missing_dims]

            # Perform the weighted combination to get the CAM
            cam = torch.nansum(weight * activation.squeeze(0), dim=0)  # type: ignore[union-attr]

            if self._relu:
                cam = F.relu(cam, inplace=True)

            # Normalize the CAM
            if normalized:
                cam = self._normalize(cam)

            cams.append(cam)

        return cams

    def extra_repr(self) -> str:
        return f"target_layer={self.target_names}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"
