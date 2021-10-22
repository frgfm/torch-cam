# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import logging
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
            target_name = target_layer
        elif isinstance(target_layer, nn.Module):
            # Find the location of the module
            _found = False
            for k, v in self.submodule_dict.items():
                if id(v) == id(target_layer):
                    target_name = k
                    _found = True
                    break
            if not _found:
                raise ValueError("unable to locate `target_layer` module inside the specified model.")
        elif target_layer is None:
            # If the layer is not specified, try automatic resolution
            target_name = locate_candidate_layer(model, input_shape)  # type: ignore[assignment]
            # Warn the user of the choice
            if isinstance(target_name, str):
                logging.warning(f"no value was provided for `target_layer`, thus set to '{target_name}'.")
            else:
                raise ValueError("unable to resolve `target_layer` automatically, please specify its value.")

        if target_name not in self.submodule_dict.keys():
            raise ValueError(f"Unable to find submodule {target_name} in the model")
        self.target_name = target_name
        self.model = model
        # Init hooks
        self.hook_a: Optional[Tensor] = None
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        # Forward hook
        self.hook_handles.append(self.submodule_dict[target_name].register_forward_hook(self._hook_a))
        # Enable hooks
        self._hooks_enabled = enable_hooks
        # Should ReLU be used before normalization
        self._relu = False
        # Model output is used by the extractor
        self._score_used = False

    def _hook_a(self, module: nn.Module, input: Tensor, output: Tensor) -> None:
        """Activation hook"""
        if self._hooks_enabled:
            self.hook_a = output.data

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

    def _get_weights(self, class_idx: int, scores: Optional[Tensor] = None) -> Tensor:

        raise NotImplementedError

    def _precheck(self, class_idx: int, scores: Optional[Tensor] = None) -> None:
        """Check for invalid computation cases"""

        # Check that forward has already occurred
        if not isinstance(self.hook_a, Tensor):
            raise AssertionError("Inputs need to be forwarded in the model for the conv features to be hooked")
        # Check batch size
        if self.hook_a.shape[0] != 1:
            raise ValueError(f"expected a 1-sized batch to be hooked. Received: {self.hook_a.shape[0]}")

        # Check class_idx value
        if not isinstance(class_idx, int) or class_idx < 0:
            raise ValueError("Incorrect `class_idx` argument value")

        # Check scores arg
        if self._score_used and not isinstance(scores, torch.Tensor):
            raise ValueError("model output scores is required to be passed to compute CAMs")

    def __call__(self, class_idx: int, scores: Optional[Tensor] = None, normalized: bool = True) -> Tensor:

        # Integrity check
        self._precheck(class_idx, scores)

        # Compute CAM
        return self.compute_cams(class_idx, scores, normalized)

    def compute_cams(self, class_idx: int, scores: Optional[Tensor] = None, normalized: bool = True) -> Tensor:
        """Compute the CAM for a specific output class

        Args:
            class_idx (int): output class index of the target class whose CAM will be computed
            scores (torch.Tensor[1, K], optional): forward output scores of the hooked model
            normalized (bool, optional): whether the CAM should be normalized

        Returns:
            torch.Tensor[M, N]: class activation map of hooked conv layer
        """

        # Get map weight & unsqueeze it
        weights = self._get_weights(class_idx, scores)
        missing_dims = self.hook_a.ndim - weights.ndim - 1  # type: ignore[union-attr]
        weights = weights[(...,) + (None,) * missing_dims]

        # Perform the weighted combination to get the CAM
        batch_cams = torch.nansum(weights * self.hook_a.squeeze(0), dim=0)  # type: ignore[union-attr]

        if self._relu:
            batch_cams = F.relu(batch_cams, inplace=True)

        # Normalize the CAM
        if normalized:
            batch_cams = self._normalize(batch_cams)

        return batch_cams

    def extra_repr(self) -> str:
        return f"target_layer='{self.target_name}'"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"
