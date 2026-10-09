# Copyright (C) 2020-2026, François-Guillaume Fernandez.
# SPDX-License-Identifier: Apache-2.0

"""Replay the trusted notebook and prepare small, paired agent workspaces."""

import argparse
import copy
import hashlib
import json
import random
import shutil
import subprocess  # noqa: S404
from collections.abc import Callable
from pathlib import Path

import torch

NOTEBOOK_REVISION = "cb719ae40cd137e93cb9a16bdb59f1c03e4ee587"
NOTEBOOK_SHA256 = "ac97e6470f983d49120c1a97c429a8e60e42779124c1a1470084028b9317781d"
ROOT = Path(__file__).resolve().parents[2]
BASELINE_REVISION = "5a0bc4d439ad642a37ed301d94601048e8d32de8"


def digest(path: Path) -> str:
    """Hash a file without interpreting it.

    Returns:
        SHA256 hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_notebook(path: Path, *, replay: bool = False) -> dict:
    """Execute only the hash-pinned, owner-trusted notebook; never arbitrary input.

    Returns:
        Notebook namespace, including observed artifacts when replayed.

    Raises:
        ValueError: If the notebook is not the trusted revision.
    """
    if digest(path) != NOTEBOOK_SHA256:
        raise ValueError("Expected the exact notebook from notebooks PR #21")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "notebook_experiment"}
    pilots = {}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code" or (not replay and index not in {2, 4, 5, 7}):
            continue
        source = "".join(cell["source"]).replace("%matplotlib inline\n", "")
        exec(compile(source, f"{path}:cell-{index}", "exec"), namespace)  # noqa: S102
        if replay and index == 5:
            original_train = namespace["train"]

            def capture(
                model: torch.nn.Module,
                view: dict,
                epochs: int,
                seed: int,
                *args,
                _train: Callable = original_train,
                **kwargs,
            ) -> dict:
                budget = _train(model, view, epochs, seed, *args, **kwargs)
                if epochs == namespace["EPOCHS"][0]:
                    pilots[namespace["regime"], seed] = copy.deepcopy(model.state_dict())
                return budget

            namespace["train"] = capture
    namespace["pilots"] = pilots
    return namespace


TRUSTED = """from pathlib import Path
import torch
from notebook_adapter import load_notebook

CLASS_NAMES = ["vertical", "horizontal"]
ROOT = Path(__file__).resolve().parent
N = load_notebook(ROOT / "shortcut_repair.ipynb")

def preprocess(images):
    # Notebook generator supplies RGB float32 in [0, 1]; model centers internally.
    return images.clone()

def load_model():
    model = N["TinyClassifier"]()
    model.load_state_dict(torch.load(ROOT / "checkpoint.pt", weights_only=True))
    return model.eval()

def training_pipeline():
    return N["train"]  # Reuse owner code, labels and configuration.

def views():
    return torch.load(ROOT / "views.pt", weights_only=True)
"""

DIAGNOSTICS = """from trusted import preprocess

def prepare_input(images):
    return preprocess(images){extra}
"""


def cue_check(namespace: dict, model: torch.nn.Module, view: dict) -> dict:
    """Measure a benchmark-only label-preserving color swap, separately from the location gate.

    Returns:
        Accuracy, flip rate and probability sensitivity.

    Raises:
        ValueError: If an edit changes protected label pixels.
    """
    images, labels = view["images"], view["labels"]
    edited = images.clone()
    edited[:, :, :8, :8] = images[:, [2, 1, 0], :8, :8]
    if not torch.equal(edited[:, :, 8:24, 8:24], images[:, :, 8:24, 8:24]):
        raise ValueError("Cue swap changed label pixels")
    original = namespace["probabilities"](model, images)
    changed = namespace["probabilities"](model, edited)
    return {
        "operator": "swap red and blue only in top-left 8x8; central label pixels unchanged",
        "n": len(labels),
        "original_accuracy": float((original.argmax(1) == labels).float().mean()),
        "edited_accuracy": float((changed.argmax(1) == labels).float().mean()),
        "prediction_flip_rate": float((original.argmax(1) != changed.argmax(1)).float().mean()),
        "mean_absolute_probability_change": float((original - changed).abs().mean()),
        "limit": "Sensitivity check on synthetic permitted pixels; not the matched location gate",
    }


def measurements(namespace: dict, model: torch.nn.Module, views: dict, extra: float, seed: int) -> dict:
    """Compute the hidden rubric from owner functions, before any agent submission.

    Returns:
        Numeric checks of inference parity, CAM coverage, cue sensitivity and matched controls.
    """
    validation = views["validation"]
    labels = validation["labels"]
    trusted = namespace["probabilities"](model, validation["images"])
    diagnostic = namespace["probabilities"](model, validation["images"] - extra)
    shortlist, rows, _, _ = namespace["proposals"](model, views["discovery"])
    supported, checks = namespace["confirm"](model, validation, views["train"], shortlist, seed)
    first = checks[0]  # Highest-ranked CAM tile, neutralization with matched border controls.
    cue = cue_check(namespace, model, validation)
    return {
        "trusted_accuracy": float((trusted.argmax(1) == labels).float().mean()),
        "diagnostic_accuracy_before": float((diagnostic.argmax(1) == labels).float().mean()),
        "tensor_max_abs_error_before": extra,
        "cue_flip_rate": cue["prediction_flip_rate"],
        "cue_probability_delta": cue["mean_absolute_probability_change"],
        "cam_successes": sum(row["correct"] for row in rows),
        "cam_failures": sum(not row["correct"] for row in rows),
        "cam_unusable": sum(row["status"] != "ok" for row in rows),
        "matched_excess": first["excess"],
        "matched_ci_low": first["ci"][0],
        "location_supported": supported is not None,
    }


def prepare(notebook: Path, output: Path, case_seed: int = 17) -> None:  # noqa: PLR0914
    """Replay all training once; share its artifacts between both evaluation conditions.

    Raises:
        FileNotFoundError: If Git is unavailable for reading the frozen baseline skill.
    """
    output.mkdir(parents=True, exist_ok=False)
    namespace = load_notebook(notebook, replay=True)
    report = namespace["report"]
    (output / "experiment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    provenance = {
        "notebook_revision": NOTEBOOK_REVISION,
        "notebook_sha256": digest(notebook),
        "experiment_sha256": digest(output / "experiment.json"),
        "runtime": report["runtime"],
        "case_seed": case_seed,
    }
    baseline = ROOT / ".agents/skills/torchcam-debug-prediction"
    if (git := shutil.which("git")) is None:
        raise FileNotFoundError("Git is required to read the frozen baseline skill")
    original_skill = subprocess.run(  # noqa: S603
        [git, "show", f"{BASELINE_REVISION}:.agents/skills/torchcam-debug-prediction/SKILL.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    trusted_hashes, expected = {}, {}
    cases = ["preprocessing", "shortcut", "control"]
    random.Random(case_seed).shuffle(cases)  # noqa: S311 - reproducible identifiers, not cryptographic secrecy
    case_map = dict(zip("abc", cases, strict=True))
    adapter = Path(__file__).read_text(encoding="utf-8").split("\nTRUSTED =", 1)[0]
    for condition in ("baseline", "extended"):
        for identifier, case in case_map.items():
            regime = "shortcut" if case == "shortcut" else "no_shortcut"
            checkpoint = (
                namespace["pilots"][regime, case_seed]
                if case == "shortcut"
                else namespace["models"][regime, case_seed, "unchanged"].state_dict()
            )
            target = output / condition / identifier
            target.mkdir(parents=True)
            shutil.copy2(notebook, target / "shortcut_repair.ipynb")
            (target / "notebook_adapter.py").write_text(adapter)
            (target / "trusted.py").write_text(TRUSTED)
            (target / "diagnostics.py").write_text(
                DIAGNOSTICS.format(extra=" - 0.5" if case == "preprocessing" else "")
            )
            torch.save(checkpoint, target / "checkpoint.pt")
            model = namespace["TinyClassifier"]()
            model.load_state_dict(checkpoint)
            model.eval()
            views = {
                "train": namespace["make_split"](1024, 0.95 if regime == "shortcut" else 0.5, case_seed)[0],
                "discovery": namespace["make_split"](64, 0.5, case_seed + 1000)[0],
                "validation": namespace["make_split"](256, 0.5, case_seed + 2000)[0],
            }
            torch.save(views, target / "views.pt")
            if case not in expected:
                expected[case] = measurements(namespace, model, views, 0.5 if case == "preprocessing" else 0, case_seed)
            evidence = {
                "provenance": provenance,
                "owner_contract": "central 16x16 defines label; border editable; RGB float32 [0,1]; model centers",
                "views": "trusted.views(): public train/discovery/validation; no group oracle or supplied diagnostic results",
            }
            (target / "evidence.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
            shutil.copy2(output / "experiment.json", target / "experiment.json")
            skill_dir = target / ".agents/skills/torchcam-debug-prediction"
            if condition == "extended":
                shutil.copytree(baseline, skill_dir)
            else:
                skill_dir.mkdir(parents=True)
                (skill_dir / "SKILL.md").write_text(original_skill, encoding="utf-8")
            hashes = {
                name: digest(target / name)
                for name in (
                    "trusted.py",
                    "checkpoint.pt",
                    "views.pt",
                    "shortcut_repair.ipynb",
                    "notebook_adapter.py",
                    "evidence.json",
                    "experiment.json",
                )
            }
            (target / "integrity.json").write_text(json.dumps(hashes, indent=2) + "\n")
            trusted_hashes[f"{condition}/{identifier}"] = hashes
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "trusted-hashes.json").write_text(json.dumps(trusted_hashes, indent=2) + "\n")
    (output / "case-map.json").write_text(json.dumps(case_map, indent=2) + "\n")
    (output / "expected.json").write_text(json.dumps(expected, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--case-seed", type=int, choices=(7, 17, 27), default=17)
    arguments = parser.parse_args()
    prepare(arguments.notebook, arguments.output, arguments.case_seed)
