import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

EVALUATION = Path(__file__).resolve().parents[1] / "evals/shortcut-investigation"
sys.path.insert(0, str(EVALUATION))
SPEC = importlib.util.spec_from_file_location("shortcut_score", EVALUATION / "score.py")
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)
sys.path.pop(0)


@pytest.fixture
def submission(tmp_path, monkeypatch):
    artifact = tmp_path / "experiment.json"
    artifact.write_text(json.dumps({"runs": [{"regime": "shortcut", "accuracy": 1.0, "worst_group": 1.0}] * 9}))
    evidence = {
        "provenance": {"experiment_sha256": SCORER.digest(artifact), "case_seed": 17},
    }
    (tmp_path / "evidence.json").write_text(json.dumps(evidence))
    response = {
        "diagnosis": "cue_dependence",
        "location_finding": "unresolved",
        "recommend_training": "unchanged",
        "recommendation_basis": "supported",
        "inference_preserved": True,
        "repair_outcome": "no_advantage",
        "evaluation_artifact": str(artifact),
        "guided_minus_unchanged": 0,
        "worst_group_regression": False,
        "evidence": [],
        "limitations": [],
        "tool_calls_used": 4,
        "measurements": {"cue_probability_delta": 0.9202487, "location_supported": False},
    }
    (tmp_path / "response.json").write_text(json.dumps(response))
    (tmp_path / "trusted.py").write_text(
        "import torch\nclass Model(torch.nn.Module):\n"
        " def forward(self,x): return torch.stack((x.flatten(),-x.flatten()),1)\n"
        "def load_model(): return Model().eval()\ndef preprocess(x): return x.clone()\n"
    )
    (tmp_path / "diagnostics.py").write_text("def prepare_input(x): return x.clone()\n")
    labels = torch.arange(1024) % 2
    view = {"images": (1 - 2 * labels).float().reshape(-1, 1, 1, 1), "labels": labels}
    namespace = {"make_split": lambda *_: (view, {"cue": torch.arange(1024) // 2 % 2})}
    monkeypatch.setattr(SCORER, "load_notebook", lambda _: namespace)
    integrity = {name: SCORER.digest(tmp_path / name) for name in ("evidence.json", "experiment.json", "trusted.py")}
    return tmp_path, response, integrity, response["measurements"].copy()


@pytest.mark.parametrize(
    ("changes", "completed", "unsupported"),
    [
        ({}, True, False),
        ({"measurements": {"cue_probability_delta": 0, "location_supported": False}}, True, False),
        ({"recommend_training": "guided", "recommendation_basis": "none"}, True, True),
        ({"tool_calls_used": 5}, False, False),
    ],
)
def test_scorer_distinguishes_grounded_null_unsupported_claim_and_invalid_record(
    submission, changes, completed, unsupported
):
    workspace, response, integrity, expected = submission
    response.update(changes)
    (workspace / "response.json").write_text(json.dumps(response))
    result = SCORER.score(workspace, "shortcut", integrity, expected)
    assert result["completed"] is completed
    if completed:
        assert result["correct_diagnosis"]
        assert result["unsupported_intervention"] is unsupported
        assert result["decision_grounded"]
        assert not result["diagnostic_fix_verified"]
        assert result["evidence_backed_resolution"] == (not changes)


@pytest.mark.parametrize("helper", ["identity", "patch_trusted", "control_mismatch", "tamper_artifact"])
def test_diagnostic_verification_freezes_owner_path_and_records_failed_repairs(submission, helper):
    workspace, response, integrity, expected = submission
    case = {"control_mismatch": "control", "tamper_artifact": "shortcut"}.get(helper, "preprocessing")
    response.update(
        diagnosis=SCORER.DIAGNOSES[case],
        recommend_training="none",
        repair_outcome={"control": "not_attempted", "shortcut": "no_advantage"}.get(case, "diagnostic_fixed"),
    )
    (workspace / "response.json").write_text(json.dumps(response))
    sources = {
        "identity": "def prepare_input(x): return x.clone()\n",
        "patch_trusted": "import trusted\ntrusted.preprocess=lambda x:x-.01\ndef prepare_input(x): return trusted.preprocess(x)\n",
        "control_mismatch": "def prepare_input(x): return x - .5\n",
        "tamper_artifact": "import trusted\nfrom pathlib import Path\ndef prepare_input(x):\n Path(trusted.__file__).with_name('experiment.json').write_text('bad')\n return x.clone()\n",
    }
    (workspace / "diagnostics.py").write_text(sources[helper])
    previous_path = sys.path.copy()
    result = SCORER.score(workspace, case, integrity, expected)
    assert result["diagnostic_fix_verified"] == (helper == "identity")
    assert result["evidence_backed_resolution"] == (helper == "identity")
    assert sys.path == previous_path
    assert "trusted" not in sys.modules
    if helper == "patch_trusted":
        assert not result["inference_path_preserved"]
        assert not result["independent_evaluation"]["tensor_equal"]
    elif helper in {"control_mismatch", "tamper_artifact"}:
        assert not result["inference_path_preserved"]
