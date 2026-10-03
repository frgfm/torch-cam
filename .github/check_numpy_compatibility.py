# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
import numpy as np
import torch
from PIL import Image
from torch import nn

from torchcam.explain import explain


def main():
    # Exercise both directions of the bridge; importing alone does not detect ABI failures.
    array = np.arange(192, dtype=np.float32).reshape(1, 3, 8, 8) / 255
    tensor = torch.from_numpy(array)
    np.testing.assert_array_equal(tensor.numpy(), array)

    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 3),
    ).eval()
    result = explain(model, tensor, target_layer="1")
    image = Image.new("RGB", (8, 8), color="white")

    with TemporaryDirectory() as directory:
        bundle = result.save(Path(directory) / "explanation", image)
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        artifact = manifest["classes"][str(result.predicted_class_idx)]["artifacts"][0]
        np.testing.assert_array_equal(
            np.load(bundle / artifact["map"], allow_pickle=False), result.cams[result.predicted_class_idx][0].numpy()
        )
        for key in ("heatmap", "overlay"):
            with Image.open(bundle / artifact[key]) as exported:
                exported.load()
                assert exported.size == image.size

    print(
        f"NumPy interoperability and explanation exports passed: "
        f"torch={torch.__version__}, numpy={np.__version__}, matplotlib={mpl.__version__}"
    )


if __name__ == "__main__":
    main()
