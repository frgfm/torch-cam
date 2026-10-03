# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

import torch
from torch import Tensor, nn

__all__ = ["locate_candidate_layer"]


@contextmanager
def _model_eval(model: nn.Module) -> Iterator[None]:
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def locate_candidate_layer(mod: nn.Module, input_shape: tuple[int, ...] = (3, 224, 224)) -> str | None:
    """Attempts to find a candidate layer to use for CAM extraction.

    Args:
        mod: the module to inspect
        input_shape: the expected shape of input tensor excluding the batch dimension

    Returns:
        the candidate layer for CAM
    """
    output_shapes: list[tuple[str | None, tuple[int, ...]]] = []

    def _record_output_shape(_: nn.Module, _input: Tensor, output: object, name: str | None = None) -> None:
        """Activation hook."""
        if isinstance(output, Tensor):
            output_shapes.append((name, output.shape))

    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    with _model_eval(mod):
        try:
            # forward hook on all layers
            for n, m in mod.named_modules():
                hook_handles.append(m.register_forward_hook(partial(_record_output_shape, name=n)))

            # forward empty
            with torch.no_grad():
                _ = mod(torch.zeros((1, *input_shape), device=next(mod.parameters()).device))
        finally:
            for handle in hook_handles:
                handle.remove()

    # Check output shapes
    candidate_layer = None
    for layer_name, output_shape in reversed(output_shapes):
        # Stop before flattening or global pooling
        if len(output_shape) == (len(input_shape) + 1) and any(v != 1 for v in output_shape[2:]):
            candidate_layer = layer_name
            break

    return candidate_layer
