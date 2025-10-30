# Copyright (C) 2020-2025, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from functools import partial

import torch
from torch import Tensor, nn

__all__ = ["locate_candidate_layer"]


def locate_candidate_layer(mod: nn.Module, input_shape: tuple[int, ...] = (3, 224, 224)) -> str | None:
    """Attempts to find a candidate layer to use for CAM extraction.

    Args:
        mod: the module to inspect
        input_shape: the expected shape of input tensor excluding the batch dimension

    Returns:
        the candidate layer for CAM
    """
    # Set module in eval mode
    module_mode = mod.training
    mod.eval()

    output_shapes: list[tuple[str | None, tuple[int, ...]]] = []

    def _record_output_shape(_: nn.Module, _input: Tensor, output: Tensor, name: str | None = None) -> None:
        """Activation hook."""
        output_shapes.append((name, output.shape))

    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
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
    mod.training = module_mode  # ty: ignore[unresolved-attribute]

    # Check output shapes
    candidate_layer = None
    for layer_name, output_shape in reversed(output_shapes):
        # Stop before flattening or global pooling
        if len(output_shape) == (len(input_shape) + 1) and any(v != 1 for v in output_shape[2:]):
            candidate_layer = layer_name
            break

    return candidate_layer
