# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""High-level prediction debugging for one image."""

from __future__ import annotations

import json
import platform
import shutil
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from PIL.Image import Image, fromarray
from torch import Tensor, nn

from torchcam.methods import GradCAM
from torchcam.methods.core import _CAM
from torchcam.utils import overlay_mask

__all__ = ["PredictionExplanation", "explain"]


@dataclass(frozen=True)
class PredictionExplanation:
    """Prediction, class activation maps, and reproducibility metadata for one image."""

    logits: Tensor
    predicted_class_idx: int
    expected_class_idx: int | None
    cams: Mapping[int, tuple[Tensor, ...]]
    method: str
    target_layers: tuple[str, ...]
    model: str
    input_shape: tuple[int, ...]
    versions: Mapping[str, str]
    class_names: tuple[str, ...] | None = None

    def save(self, directory: str | Path, image: Image, alpha: float = 0.5) -> Path:
        """Save NumPy maps, heatmaps, overlays, and a completion manifest to a new directory.

        Args:
            directory: new output directory
            image: source image used for full-size overlays
            alpha: source-image opacity in the overlay

        Returns:
            output directory

        Raises:
            TypeError: if the image is not a PIL image
            FileExistsError: if the output directory already exists
            ValueError: if the image, alpha, or CAM metadata is unsupported
        """
        if not isinstance(image, Image):
            raise TypeError("`image` must be a PIL image")

        output_dir = Path(directory)
        if output_dir.exists():
            raise FileExistsError(f"output directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        with ExitStack() as cleanup:
            cleanup.callback(shutil.rmtree, output_dir)
            probabilities = self.logits.softmax(dim=1)[0]
            classes: dict[str, dict[str, Any]] = {}

            for class_idx, maps in sorted(self.cams.items()):
                artifacts = []
                if len(maps) == len(self.target_layers):
                    artifact_layers = tuple((name,) for name in self.target_layers)
                elif len(maps) == 1:
                    artifact_layers = (self.target_layers,)
                else:
                    raise ValueError("CAM count does not match the resolved target layers")

                for layer_idx, (target_layers, cam) in enumerate(zip(artifact_layers, maps, strict=True)):
                    stem = f"class-{class_idx}-layer-{layer_idx}"
                    array = cam.numpy()
                    npy_path = output_dir / f"{stem}.npy"
                    heatmap_path = output_dir / f"{stem}-heatmap.png"
                    overlay_path = output_dir / f"{stem}-overlay.png"

                    np.save(npy_path, array, allow_pickle=False)
                    heatmap = fromarray((255 * np.clip(array, 0, 1)).round().astype(np.uint8))
                    heatmap.save(heatmap_path)
                    overlay_mask(image, heatmap, alpha=alpha).save(overlay_path)
                    artifacts.append({
                        "target_layers": list(target_layers),
                        "map": npy_path.name,
                        "heatmap": heatmap_path.name,
                        "overlay": overlay_path.name,
                    })

                classes[str(class_idx)] = {
                    "class_idx": class_idx,
                    "class_name": self._class_name(class_idx),
                    "logit": self.logits[0, class_idx].item(),
                    "probability": probabilities[class_idx].item(),
                    "artifacts": artifacts,
                }

            manifest = {
                "schema_version": 1,
                "prediction": self._class_reference(self.predicted_class_idx),
                "expected": self._class_reference(self.expected_class_idx),
                "classes": classes,
                "method": self.method,
                "target_layers": list(self.target_layers),
                "model": self.model,
                "input_shape": list(self.input_shape),
                "versions": dict(self.versions),
                "image_size": list(image.size),
                "alpha": alpha,
            }
            (output_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
            )
            cleanup.pop_all()
        return output_dir

    def _class_name(self, class_idx: int) -> str | None:
        return None if self.class_names is None else self.class_names[class_idx]

    def _class_reference(self, class_idx: int | None) -> dict[str, int | str | None] | None:
        if class_idx is None:
            return None
        return {"class_idx": class_idx, "class_name": self._class_name(class_idx)}


def _validate_logits(output: Any) -> Tensor:
    if not isinstance(output, Tensor):
        raise TypeError("model output must be a tensor; wrap models that return tuples or dictionaries")
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] < 1:
        raise ValueError("model output must have shape (1, num_classes)")
    if not torch.isfinite(output).all():
        raise ValueError("model output contains non-finite logits")
    return output


def _prepare_maps(maps: list[Tensor]) -> tuple[Tensor, ...]:
    if not maps:
        raise ValueError("extractor returned no CAMs")
    prepared = []
    for cam in maps:
        if not isinstance(cam, Tensor) or cam.ndim != 3 or cam.shape[0] != 1:
            raise ValueError("CAMs must have shape (1, height, width)")
        if not torch.isfinite(cam).all():
            raise ValueError("CAM contains non-finite values")
        prepared.append(cam[0].detach().to(device="cpu", dtype=torch.float32).contiguous())
    return tuple(prepared)


def _validate_request(
    model: nn.Module,
    input_tensor: Tensor,
    expected_class_idx: int | None,
    method: type[_CAM],
    method_kwargs: Mapping[str, Any] | None,
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a torch.nn.Module")
    if any(module.training for module in model.modules()):
        raise ValueError("`model` must be in evaluation mode; call model.eval() first")
    if not isinstance(input_tensor, Tensor):
        raise TypeError("`input_tensor` must be a torch.Tensor")
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("`input_tensor` must have shape (1, channels, height, width)")
    if not input_tensor.is_floating_point():
        raise ValueError("`input_tensor` must have a floating-point dtype")
    if expected_class_idx is not None and (
        not isinstance(expected_class_idx, int) or isinstance(expected_class_idx, bool)
    ):
        raise TypeError("`expected_class_idx` must be an integer or None")
    if not isinstance(method, type) or not issubclass(method, _CAM):
        raise TypeError("`method` must be a TorchCAM extractor class")
    if method_kwargs is not None and not isinstance(method_kwargs, Mapping):
        raise TypeError("`method_kwargs` must be a mapping or None")


def _validate_classes(
    expected_class_idx: int | None, class_names: Sequence[str] | None, num_classes: int
) -> tuple[str, ...] | None:
    if expected_class_idx is not None and not 0 <= expected_class_idx < num_classes:
        raise ValueError("`expected_class_idx` is outside the model output range")
    if class_names is None:
        return None
    if isinstance(class_names, str) or len(class_names) != num_classes:
        raise ValueError("`class_names` length must match the number of model classes")
    if any(not isinstance(name, str) for name in class_names):
        raise TypeError("every class name must be a string")
    return tuple(class_names)


def explain(
    model: nn.Module,
    input_tensor: Tensor,
    *,
    expected_class_idx: int | None = None,
    class_names: Sequence[str] | None = None,
    method: type[_CAM] = GradCAM,
    target_layer: nn.Module | str | list[nn.Module | str] | None = None,
    method_kwargs: Mapping[str, Any] | None = None,
) -> PredictionExplanation:
    """Explain the predicted and optional expected class for one 2D image.

    Returns:
        detached prediction evidence and CAMs

    Raises:
        RuntimeError: if called from inference mode
        ValueError: if an argument, model output, or CAM has an unsupported value
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("`explain` cannot run inside torch.inference_mode() because CAM extraction needs gradients")
    _validate_request(model, input_tensor, expected_class_idx, method, method_kwargs)

    kwargs = dict(method_kwargs or {})
    if "target_layer" in kwargs:
        raise ValueError("pass `target_layer` directly, not through `method_kwargs`")
    if target_layer is not None:
        kwargs["target_layer"] = target_layer
    elif method is GradCAM:
        kwargs.setdefault("input_shape", tuple(input_tensor.shape[1:]))

    working_input = input_tensor.detach().requires_grad_(True)
    parameters = tuple(model.parameters())
    gradients = tuple(parameter.grad for parameter in parameters)

    try:
        with torch.enable_grad(), method(model, **kwargs) as extractor:
            logits = _validate_logits(model(working_input))
            predicted_class_idx = int(logits[0].argmax().item())
            validated_class_names = _validate_classes(expected_class_idx, class_names, logits.shape[1])

            cams = {predicted_class_idx: _prepare_maps(extractor(predicted_class_idx, logits))}
            if expected_class_idx is not None and expected_class_idx != predicted_class_idx:
                expected_logits = _validate_logits(model(working_input))
                if expected_logits.shape != logits.shape:
                    raise ValueError("model output shape changed between explanation forwards")
                cams[expected_class_idx] = _prepare_maps(extractor(expected_class_idx, expected_logits))

            result = PredictionExplanation(
                logits=logits.detach().cpu(),
                predicted_class_idx=predicted_class_idx,
                expected_class_idx=expected_class_idx,
                cams=MappingProxyType(cams),
                method=method.__name__,
                target_layers=tuple(extractor.target_names),
                model=f"{model.__class__.__module__}.{model.__class__.__qualname__}",
                input_shape=tuple(input_tensor.shape),
                versions=MappingProxyType({
                    "python": platform.python_version(),
                    "torch": str(torch.__version__),
                    "torchcam": version("torchcam"),
                }),
                class_names=validated_class_names,
            )
    finally:
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient

    return result
