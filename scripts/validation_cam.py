# Copyright (C) 2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Select validation examples and explain them with existing batched GradCAM extraction."""

import argparse
import json
import platform
import random
import shutil
from contextlib import ExitStack
from operator import itemgetter
from pathlib import Path
from time import perf_counter

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, SequentialSampler
from torchvision.datasets import ImageFolder
from torchvision.models import get_model, get_model_weights
from torchvision.transforms.functional import center_crop, resize

from torchcam import __version__
from torchcam.explain import PredictionExplanation

if __package__:
    from .cam_example import build_extractor, nonnegative_int, positive_int
else:
    from cam_example import build_extractor, nonnegative_int, positive_int


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def validate(model, loader, *, per_bucket=4, seed=0):  # noqa: PLR0912
    """Run normal validation and retain bounded records, never full-dataset inputs or logits.

    The loader must use dataset order, deterministic preprocessing and independent samples.

    Returns:
        selected case records and validation/selection timings

    Raises:
        ValueError: if settings, labels, outputs or the dataset are invalid
        TypeError: if the loader does not follow dataset order
    """
    if per_bucket < 1 or any(module.training for module in model.modules()):
        raise ValueError("use a positive per_bucket and model.eval()")
    if not isinstance(loader.sampler, SequentialSampler):
        raise TypeError("use a sequential validation loader (shuffle=False)")
    device = next(model.parameters()).device
    rng = random.Random(seed)  # noqa: S311
    wrong, correct, controls = [], [], {}
    total, hits, selection_seconds = 0, 0, 0.0
    _synchronize(device)
    started = perf_counter()
    with torch.no_grad():
        for inputs, labels in loader:
            scores = model(inputs.to(device))
            if scores.ndim != 2 or scores.shape[1] < 2 or not torch.isfinite(scores).all():
                raise ValueError("expected finite classifier logits shaped (batch, classes), with at least two classes")
            if (labels < 0).any() or (labels >= scores.shape[1]).any():
                raise ValueError("labels must use model output indices")
            predictions = scores.argmax(1).cpu()
            hits += int((predictions == labels).sum())
            _synchronize(device)
            selecting = perf_counter()
            # Keep argmax's first-index tie rule, including when all logits are equal.
            alternatives = scores.clone().scatter_(1, predictions.to(device).unsqueeze(1), -torch.inf).argmax(1)
            indices = torch.stack((predictions.to(device), alternatives), dim=1)
            probabilities = scores.softmax(1).gather(1, indices)
            for offset, (label, probs, classes) in enumerate(
                zip(labels.tolist(), probabilities.cpu().tolist(), indices.cpu().tolist(), strict=True)
            ):
                case = {
                    "index": total + offset,
                    "label": label,
                    "prediction": classes[0],
                    "confidence": probs[0],
                    "runner_up": classes[1],
                    "runner_up_confidence": probs[1],
                }
                if case["prediction"] != label:
                    wrong = sorted([*wrong, case], key=lambda row: (-row["confidence"], row["index"]))[:per_bucket]
                else:
                    correct = sorted([*correct, case], key=itemgetter("confidence", "index"))[:per_bucket]
                priority = rng.random()
                if label not in controls:
                    controls[label] = (rng.random(), priority, case)
                elif priority < controls[label][1]:
                    controls[label] = (controls[label][0], priority, case)
            selection_seconds += perf_counter() - selecting
            total += len(labels)
    _synchronize(device)
    validation_seconds = perf_counter() - started
    if not total:
        raise ValueError("validation loader is empty")
    # Sample controls independently of predictions so the panel stays fixed across epochs.
    control = [entry[2] for entry in sorted(controls.values(), key=itemgetter(0))[:per_bucket]]
    selected = {}
    for bucket, cases in (("wrong_high", wrong), ("correct_low", correct), ("control", control)):
        for case in cases:
            selected.setdefault(case["index"], {**case, "buckets": []})["buckets"].append(bucket)
    return list(selected.values()), {
        "samples": total,
        "accuracy": hits / total,
        "per_bucket": per_bucket,
        "seed": seed,
        "validation_seconds": validation_seconds,
        "selection_seconds": selection_seconds,
    }


def _contact_sheet(directory, records):
    width, height = 224, 290
    sheet = Image.new("RGB", (3 * width, len(records) * height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for row, record in enumerate(records):
        bundle = directory / record["bundle"]
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        predicted = manifest["classes"][str(record["prediction"])]
        compared = manifest["classes"][str(record["comparison"])]
        cells = [
            (
                manifest["input_image"],
                f"#{record['index']}\n{', '.join(record['buckets'])}\nlabel: {record['class_name']}",
            ),
            (
                predicted["artifacts"][0]["overlay"],
                f"predicted: {predicted['class_name']}\np={predicted['probability']:.3f}",
            ),
            (
                compared["artifacts"][0]["overlay"],
                f"{record['comparison_role']}: {compared['class_name']}\np={compared['probability']:.3f}",
            ),
        ]
        for col, (filename, caption) in enumerate(cells):
            with Image.open(bundle / filename) as image:
                image.thumbnail((width, width))
                sheet.paste(image, (col * width, row * height))
            caption = "\n".join(line if len(line) <= 28 else line[:25] + "..." for line in caption.splitlines())
            draw.multiline_text((col * width + 4, row * height + width + 4), caption, fill="black", font=font)
    sheet.save(directory / "contact-sheet.png")


def save_evidence(  # noqa: PLR0915
    model,
    dataset,
    cases,
    directory,
    *,
    class_names,
    model_view,
    target_layer=None,
    batch_size=32,
    context=None,
    validation=None,
):
    """Explain selected ImageFolder cases, retaining the owner's model and preprocessing.

    model_view(image) must reproduce inference's spatial transforms before normalization.
    This example saves one target layer; use an explicit target for a custom model.

    Returns:
        the saved report, including extraction and artifact timings

    Raises:
        ValueError: if the settings, outputs, maps or repeated inference are invalid
        RuntimeError: if called inside inference mode
    """
    if torch.is_inference_mode_enabled():
        raise RuntimeError("run CAM extraction outside torch.inference_mode()")
    if any(module.training for module in model.modules()) or not cases or batch_size < 1:
        raise ValueError("use model.eval(), nonempty cases and a positive batch_size")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    device = next(model.parameters()).device
    versions = {"python": platform.python_version(), "torch": str(torch.__version__), "torchcam": __version__}
    records, cam_seconds, artifact_seconds = [], 0.0, 0.0
    with ExitStack() as cleanup:
        cleanup.callback(shutil.rmtree, directory)
        _synchronize(device)
        started = perf_counter()
        shape = tuple(dataset[cases[0]["index"]][0].shape)
        with torch.enable_grad(), build_extractor(model, "GradCAM", target_layer, shape) as extractor:
            if len(extractor.target_names) != 1:
                raise ValueError("choose one target layer for this example")
            _synchronize(device)
            cam_seconds += perf_counter() - started
            for start in range(0, len(cases), batch_size):
                chunk = cases[start : start + batch_size]
                _synchronize(device)
                started = perf_counter()
                batch = [dataset[case["index"]] for case in chunk]
                if [label for _, label in batch] != [case["label"] for case in chunk]:
                    raise ValueError("labels changed since selection")
                inputs = torch.stack([image for image, _ in batch]).to(device).requires_grad_(True)
                scores = model(inputs)
                if scores.shape != (len(chunk), len(class_names)) or not torch.isfinite(scores).all():
                    raise ValueError("model output does not match class_names or contains nonfinite logits")
                probabilities = scores.detach().softmax(1)
                prediction = [case["prediction"] for case in chunk]
                comparison = [
                    case["label"] if case["label"] != case["prediction"] else case["runner_up"] for case in chunk
                ]
                expected = probabilities.new_tensor([
                    [case["confidence"], case["runner_up_confidence"]] for case in chunk
                ])
                indices = torch.tensor([[case["prediction"], case["runner_up"]] for case in chunk], device=device)
                if scores.argmax(1).tolist() != prediction or not torch.allclose(
                    probabilities.gather(1, indices), expected, rtol=1e-4, atol=1e-5
                ):
                    raise ValueError(
                        "inference changed since selection; use deterministic preprocessing and an unchanged model"
                    )
                predicted_cams = extractor(prediction, scores, retain_graph=True)[0].detach().float().cpu()
                compared_cams = extractor(comparison, scores)[0].detach().float().cpu()
                if not torch.isfinite(predicted_cams).all() or not torch.isfinite(compared_cams).all():
                    raise ValueError("nonfinite CAM; no complete evidence bundle was written")
                logits = scores.detach().cpu()
                _synchronize(device)
                cam_seconds += perf_counter() - started
                started = perf_counter()
                for idx, case in enumerate(chunk):
                    path = dataset.samples[case["index"]][0]
                    sample_id = Path(path).relative_to(dataset.root).as_posix()
                    image = model_view(dataset.loader(path))
                    maps = {prediction[idx]: (predicted_cams[idx],), comparison[idx]: (compared_cams[idx],)}
                    result = PredictionExplanation(
                        logits=logits[idx : idx + 1],
                        predicted_class_idx=prediction[idx],
                        expected_class_idx=case["label"],
                        cams=maps,
                        method="GradCAM",
                        target_layers=tuple(extractor.target_names),
                        model=f"{type(model).__module__}.{type(model).__qualname__}",
                        input_shape=(1, *inputs.shape[1:]),
                        versions=versions,
                        class_names=tuple(class_names),
                    )
                    bundle = f"sample-{case['index']:06d}"
                    result.save(
                        directory / bundle,
                        image,
                        context={**(context or {}), "sample_id": sample_id, "buckets": ",".join(case["buckets"])},
                    )
                    records.append({
                        **case,
                        "sample_id": sample_id,
                        "class_name": class_names[case["label"]],
                        "comparison": comparison[idx],
                        "comparison_role": "label" if prediction[idx] != case["label"] else "runner-up",
                        "blank_classes": [key for key, maps_ in maps.items() if maps_[0].max() == maps_[0].min()],
                        "bundle": bundle,
                    })
                artifact_seconds += perf_counter() - started
        started = perf_counter()
        _contact_sheet(directory, records)
        artifact_seconds += perf_counter() - started
        report = {
            "schema_version": 1,
            "method": "GradCAM",
            "target_layers": extractor.target_names,
            "versions": {**versions, "torchvision": torchvision.__version__},
            "device": str(device),
            "threads": torch.get_num_threads(),
            "context": context or {},
            "cam_batch_size": batch_size,
            "cam_seconds": cam_seconds,
            "artifact_seconds": artifact_seconds,
            "contact_sheet": "contact-sheet.png",
            "validation": validation,
            "samples": records,
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        cleanup.pop_all()
    return report


def main(args):
    if Path(args.output).exists():
        raise FileExistsError(f"output directory already exists: {args.output}")
    device = torch.device(args.device)
    weights = get_model_weights(args.arch).DEFAULT
    preprocess = weights.transforms()
    class_names = weights.meta["categories"]
    dataset = ImageFolder(args.data, transform=preprocess)
    # Numeric folder names are actual output indices, even for a subset of model classes.
    labels = [int(name) for name in dataset.classes]
    if any(label < 0 or label >= len(class_names) for label in labels):
        raise ValueError("folder names must be model output indices")
    dataset.target_transform = labels.__getitem__
    model = get_model(args.arch, weights=weights).to(device).eval().requires_grad_(False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    cases, validation = validate(model, loader, per_bucket=args.per_bucket, seed=args.seed)

    def model_view(image):
        return center_crop(
            resize(
                image, preprocess.resize_size, interpolation=preprocess.interpolation, antialias=preprocess.antialias
            ),
            preprocess.crop_size,
        )

    report = save_evidence(
        model,
        dataset,
        cases,
        args.output,
        class_names=class_names,
        model_view=model_view,
        target_layer=args.target,
        batch_size=args.batch_size,
        validation=validation,
        context={
            "checkpoint_id": weights.url,
            "preprocessing": repr(preprocess),
            "dataset": str(Path(args.data).resolve()),
            "seed": str(args.seed),
        },
    )
    print(f"Accuracy {validation['accuracy']:.2%}; selected {len(cases)}/{validation['samples']} examples")
    print(
        f"Validation {validation['validation_seconds']:.3f}s (including selection {validation['selection_seconds']:.3f}s); "
        f"selected CAMs {report['cam_seconds']:.3f}s; artifacts {report['artifact_seconds']:.3f}s"
    )
    print(f"Contact sheet and per-example evidence: {args.output}")


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", help="ImageFolder root; folder names are numeric model output indices")
    parser.add_argument("output", help="new evidence directory, e.g. validation/epoch-001")
    parser.add_argument("--arch", default="resnet18", help="torchvision model with pretrained weights")
    parser.add_argument("--target", help="single target layer; defaults to automatic resolution")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--per-bucket", type=positive_int, default=4, help="maximum cases per selection bucket")
    parser.add_argument("--seed", type=nonnegative_int, default=0, help="fixed control-sampling seed")
    parser.add_argument("--device", default="cpu", help="cpu, cuda or mps")
    return parser


if __name__ == "__main__":
    main(_build_parser().parse_args())
