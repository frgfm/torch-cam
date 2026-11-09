# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from collections import UserDict
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

__all__ = ["locate_candidate_layer"]

_CallCallbacks = tuple[Callable[[], None], Callable[[], None], Callable[[], None]]


class _ModelCallHooks(UserDict[int, _CallCallbacks]):
    """Share an exception boundary around a model's complete PyTorch call."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.attribute = "_compiled_call_impl" if model._compiled_call_impl is not None else "_call_impl"  # noqa: SLF001
        self.had_override = self.attribute in model.__dict__
        self.previous = getattr(model, self.attribute)
        self.wrapper = self.__call__
        setattr(model, self.attribute, self.wrapper)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        entered = []
        try:
            for callbacks in tuple(self.values()):
                entered.append(callbacks)
                callbacks[0]()
            return self.previous(*args, **kwargs)
        except BaseException:
            for _, abort, _ in entered:
                abort()
            raise
        finally:
            for _, _, finish in reversed(entered):
                finish()

    def __delitem__(self, key: int) -> None:
        super().__delitem__(key)
        if not self:
            if self.model.__dict__.get(self.attribute) is self.wrapper:
                if self.had_override:
                    setattr(self.model, self.attribute, self.previous)
                else:
                    delattr(self.model, self.attribute)
            del self.model.__dict__["_torchcam_call_hooks"]


def _register_model_call_hook(model: nn.Module, callbacks: _CallCallbacks) -> RemovableHandle:
    """Observe calls before prehooks and clean up even when caller hooks raise.

    Returns:
        a handle that restores the original dispatch when the last observer closes
    """
    hooks = model.__dict__.get("_torchcam_call_hooks")
    if hooks is None:
        hooks = _ModelCallHooks(model)
        model.__dict__["_torchcam_call_hooks"] = hooks
    handle = RemovableHandle(hooks)
    hooks[handle.id] = callbacks
    return handle


@contextmanager
def _model_eval(model: nn.Module) -> Generator[None, None, None]:
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
    feature_dicts: list[str | None] = []

    def _record_output_shape(_: nn.Module, _input: Tensor, output: object, name: str | None = None) -> None:
        """Activation hook."""
        # Layers after a feature dict (e.g. detection or segmentation backbones) only feed some levels or heads
        if isinstance(output, Mapping):
            feature_dicts.append(name)
        elif isinstance(output, Tensor) and not feature_dicts:
            output_shapes.append((name, output.shape))

    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    with _model_eval(mod):
        try:
            # forward hook on all layers
            for n, m in mod.named_modules():
                hook_handles.append(m.register_forward_hook(partial(_record_output_shape, name=n)))

            # forward empty
            with torch.no_grad():
                _ = mod(next(mod.parameters()).new_zeros((1, *input_shape)))
        finally:
            for handle in hook_handles:
                handle.remove()

    # Check output shapes
    candidate_layer = None
    names = [name for name, _ in output_shapes]
    for layer_name, output_shape in reversed(output_shapes):
        # Stop before flattening or global pooling; skip per-region outputs and modules shared across feature levels
        if (
            len(output_shape) == (len(input_shape) + 1)
            and output_shape[0] == 1
            and names.count(layer_name) == 1
            and any(v != 1 for v in output_shape[2:])
        ):
            candidate_layer = layer_name
            break

    return candidate_layer
