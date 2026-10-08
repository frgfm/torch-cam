# Copyright (C) 2020-2026, François-Guillaume Fernandez.
# SPDX-License-Identifier: Apache-2.0

"""Validate a saved TorchCAM schema-v1 bundle against the owner's ordered labels."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate_bundle(directory: Path, class_names: list[str]) -> dict:  # noqa: PLR0912
    """Return per-map ranges after verifying completion, paths, arrays, images and class ordering.

    Returns:
        Method, resolved layers and per-class artifact checks.

    Raises:
        ValueError: If the bundle violates the schema-v1 artifact contract.
    """
    root = directory.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Expected schema_version 1")
    classes = manifest["classes"]
    if not classes or not isinstance(manifest["prediction"], dict):
        raise ValueError("Expected class artifacts and a prediction reference")
    for reference in (manifest["prediction"], manifest.get("expected")):
        if reference is not None:
            index = reference["class_idx"]
            if type(index) is not int or str(index) not in classes:
                raise ValueError("Prediction/expected class missing from classes")
            if reference["class_name"] != classes[str(index)]["class_name"]:
                raise ValueError("Prediction/expected label differs from class entry")
    rows = []
    for key, entry in classes.items():
        index = entry["class_idx"]
        if (
            type(index) is not int
            or str(index) != key
            or not 0 <= index < len(class_names)
            or entry["class_name"] != class_names[index]
        ):
            raise ValueError("Class entry does not match the owner's ordered labels")
        if not entry["artifacts"]:
            raise ValueError("Class has no map artifacts")
        for artifact in entry["artifacts"]:
            paths = {}
            for kind in ("map", "heatmap", "overlay"):
                relative = Path(artifact[kind])
                path = (root / relative).resolve()
                if relative.is_absolute() or not path.is_relative_to(root) or not path.is_file():
                    raise ValueError("Artifact must be an existing relative file inside the bundle")
                paths[kind] = path
            cam = np.load(paths["map"], allow_pickle=False)
            if cam.dtype != np.float32 or cam.ndim != 2 or cam.size == 0 or not np.isfinite(cam).all():
                raise ValueError("Expected a nonempty, finite 2D float32 CAM")
            for kind in ("heatmap", "overlay"):
                with Image.open(paths[kind]) as image:
                    image.load()
                    if kind == "overlay" and image.size != tuple(manifest["image_size"]):
                        raise ValueError("Overlay dimensions differ from image_size [width, height]")
            rows.append({
                "class_idx": index,
                "map": artifact["map"],
                "min": float(cam.min()),
                "max": float(cam.max()),
                "blank": bool(cam.max() == cam.min()),
            })
    return {"schema_version": 1, "method": manifest["method"], "target_layers": manifest["target_layers"], "maps": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--class-names", nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(validate_bundle(args.directory, args.class_names), indent=2, allow_nan=False))  # noqa: T201
