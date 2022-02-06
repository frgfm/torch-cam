# Copyright (C) 2020-2022, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from functools import partial
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

__all__ = ['locate_candidate_layer', 'locate_linear_layer']


def locate_candidate_layer(mod: nn.Module, input_shape: Tuple[int, ...] = (3, 224, 224)) -> Optional[str]:
    """Attempts to find a candidate layer to use for CAM extraction

    Args:
        mod: the module to inspect
        input_shape: the expected shape of input tensor excluding the batch dimension

    Returns:
        str: the candidate layer for CAM
    """

    # Set module in eval mode
    module_mode = mod.training
    mod.eval()

    output_shapes: List[Tuple[Optional[str], Tuple[int, ...]]] = []

    def _record_output_shape(module: nn.Module, input: Tensor, output: Tensor, name: Optional[str] = None) -> None:
        """Activation hook."""
        output_shapes.append((name, output.shape))

    hook_handles: List[torch.utils.hooks.RemovableHandle] = []
    # forward hook on all layers
    for n, m in mod.named_modules():
        hook_handles.append(m.register_forward_hook(partial(_record_output_shape, name=n)))

    # forward empty
    with torch.no_grad():
        _ = mod(torch.zeros((1, *input_shape), device=next(mod.parameters()).data.device))

    # Remove all temporary hooks
    for handle in hook_handles:
        handle.remove()

    # Put back the model in the corresponding mode
    mod.training = module_mode

    # Check output shapes
    candidate_layer = None
    for layer_name, output_shape in output_shapes:
        # Stop before flattening or global pooling
        if len(output_shape) != (len(input_shape) + 1) or all(v == 1 for v in output_shape[2:]):
            break
        else:
            candidate_layer = layer_name

    return candidate_layer


def locate_linear_layer(mod: nn.Module) -> Optional[str]:
    """Attempts to find a fully connecter layer to use for CAM extraction

    Args:
        mod: the module to inspect

    Returns:
        str: the candidate layer
    """

    candidate_layer = None
    for layer_name, m in mod.named_modules():
        if isinstance(m, nn.Linear):
            candidate_layer = layer_name
            break

    return candidate_layer
