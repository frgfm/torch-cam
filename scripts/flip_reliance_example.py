# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Controlled example: unchanged probabilities, changed reliance after a horizontal flip."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor

if __package__:
    from .validation_cam import save_evidence, validate
else:
    from validation_cam import save_evidence, validate


class LeftCueClassifier(nn.Module):
    """Hand-built control: read red or green evidence at the left-hand location."""

    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(3, 2, 1, bias=False)
        with torch.no_grad():
            self.features.weight.copy_(torch.tensor([[1.0, 0, 0], [0, 1.0, 0]]).reshape(2, 3, 1, 1))

    def forward(self, inputs):
        features = self.features(inputs)
        evidence = features[:, :, inputs.shape[-2] // 2, inputs.shape[-1] // 4].amax(1) * 4
        return torch.stack((evidence, torch.ones_like(evidence)), dim=1)


def main(output):
    image = Image.new("RGB", (224, 224))
    draw = ImageDraw.Draw(image)
    draw.rectangle((28, 70, 83, 153), fill="red")
    draw.rectangle((140, 70, 195, 153), fill="lime")
    model = LeftCueClassifier().eval().requires_grad_(False)
    inputs = to_tensor(image).unsqueeze(0)
    no_nuisance = inputs.clone()
    no_nuisance[:, 1] = 0
    with torch.no_grad():
        scores = model(torch.cat((inputs, inputs.flip(-1), no_nuisance, no_nuisance.flip(-1))))
    probabilities = scores.softmax(1)
    torch.testing.assert_close(probabilities[0], probabilities[1], rtol=0, atol=0)
    torch.testing.assert_close(probabilities[0], probabilities[2], rtol=0, atol=0)
    assert scores[0].argmax() != scores[3].argmax()
    check = {
        "label_rule": "red object present; green is an irrelevant cue",
        "conditions": ["original", "flipped", "original_without_green", "flipped_without_green"],
        "probabilities": probabilities.tolist(),
        "predictions": scores.argmax(1).tolist(),
        "margins": (scores[:, 0] - scores[:, 1]).tolist(),
        "limit": "hand-built neural control, not a trained model or a bias-detection benchmark",
    }
    with TemporaryDirectory() as temporary:
        folder = Path(temporary) / "0"
        folder.mkdir()
        image.save(folder / "red-object-green-cue.png")
        dataset = ImageFolder(temporary, transform=to_tensor)
        cases, validation = validate(model, DataLoader(dataset), per_bucket=1)
        save_evidence(
            model,
            dataset,
            cases,
            output,
            class_names=["red present", "red absent"],
            model_view=lambda source: source,
            target_layer="features",
            batch_size=1,
            context={"checkpoint_id": "hand-built-left-cue-control"},
            validation=validation,
            compare_flip=True,
        )
    (Path(output) / "controlled-check.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(check, indent=2))
    return check


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="new evidence directory")
    main(parser.parse_args().output)
