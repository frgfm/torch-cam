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
def submission(tmp_path):
    artifact = tmp_path / "experiment.json"
    artifact.write_text(json.dumps({"runs": [{"regime": "shortcut", "accuracy": 1.0, "worst_group": 1.0}] * 9}))
    evidence = {
        "provenance": {"experiment_sha256": SCORER.digest(artifact)},
        "pilot_location_checks": {"regime": "shortcut"},
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
    }
    (tmp_path / "response.json").write_text(json.dumps(response))
    integrity = {name: SCORER.digest(tmp_path / name) for name in ("evidence.json", "experiment.json")}
    return tmp_path, response, integrity


@pytest.mark.parametrize(
    ("changes", "completed", "unsupported"),
    [({}, True, False), ({"recommend_training": "guided"}, True, True), ({"tool_calls_used": "4"}, False, False)],
)
def test_scorer_distinguishes_grounded_null_unsupported_claim_and_invalid_record(
    submission, changes, completed, unsupported
):
    workspace, response, integrity = submission
    response.update(changes)
    (workspace / "response.json").write_text(json.dumps(response))
    result = SCORER.score(workspace, "shortcut", integrity)
    assert result["completed"] is completed
    if completed:
        assert result["correct_diagnosis"]
        assert result["unsupported_intervention"] is unsupported
        assert result["verified_outcome"]


@pytest.mark.parametrize("helper", ["identity", "patch_trusted", "tamper_file", "system_exit"])
def test_diagnostic_verification_freezes_owner_path_and_records_failed_repairs(submission, monkeypatch, helper):
    workspace, response, integrity = submission
    response.update(diagnosis="preprocessing_mismatch", repair_outcome="diagnostic_fixed", recommend_training="none")
    (workspace / "response.json").write_text(json.dumps(response))
    evidence = json.loads((workspace / "evidence.json").read_text())
    evidence["provenance"]["case_seed"] = 17
    (workspace / "evidence.json").write_text(json.dumps(evidence))
    (workspace / "trusted.py").write_text(
        "import torch\nclass Model(torch.nn.Module):\n"
        " def forward(self,x): return torch.stack((x.flatten(),-x.flatten()),1)\n"
        "def load_model(): return Model().eval()\ndef preprocess(x): return x.clone()\n"
    )
    sources = {
        "identity": "def prepare_input(x): return x.clone()\n",
        "patch_trusted": "import trusted\ntrusted.preprocess=lambda x:x-.01\ndef prepare_input(x): return trusted.preprocess(x)\n",
        "tamper_file": "import trusted\nfrom pathlib import Path\nPath(trusted.__file__).write_text('changed')\ndef prepare_input(x): return x.clone()\n",
        "system_exit": "raise SystemExit('failed repair')\n",
    }
    (workspace / "diagnostics.py").write_text(sources[helper])
    integrity.update({name: SCORER.digest(workspace / name) for name in ("trusted.py", "evidence.json")})
    labels = torch.arange(1024) % 2
    view = {"images": (1 - 2 * labels).float().reshape(-1, 1, 1, 1), "labels": labels}
    namespace = {"make_split": lambda *_: (view, {"cue": torch.arange(1024) // 2 % 2})}
    monkeypatch.setattr(SCORER, "load_notebook", lambda _: namespace)
    previous_path = sys.path.copy()
    result = SCORER.score(workspace, "preprocessing", integrity)
    assert result["verified_outcome"] == (helper == "identity")
    assert sys.path == previous_path
    assert "trusted" not in sys.modules
    if helper == "patch_trusted":
        assert not result["inference_path_preserved"]
        assert not result["independent_evaluation"]["tensor_equal"]
    elif helper == "tamper_file":
        assert not result["inference_path_preserved"]
    elif helper == "system_exit":
        assert "diagnostic evaluation failed" in result["reason"]
