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
RUN_SPEC = importlib.util.spec_from_file_location("shortcut_run", EVALUATION / "run.py")
RUNNER = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUNNER)


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


def test_scorer_accepts_grounded_null_outcome_and_absolute_artifact(submission):
    workspace, _, integrity = submission
    result = SCORER.score(workspace, "shortcut", integrity)
    assert result["verified_outcome"]
    assert result["correct_diagnosis"]
    assert not result["unsupported_intervention"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_artifact", None),
        ("tool_calls_used", "4"),
        ("inference_preserved", "false"),
        ("recommendation_basis", "unsupported"),
        ("tool_calls_used", True),
        ("guided_minus_unchanged", float("nan")),
    ],
)
def test_scorer_rejects_malformed_response_fields(submission, field, value):
    workspace, response, integrity = submission
    response[field] = value
    (workspace / "response.json").write_text(json.dumps(response))
    assert not SCORER.score(workspace, "shortcut", integrity)["completed"]


@pytest.mark.parametrize("helper", ["identity", "patch_trusted", "tamper_file", "raises", "system_exit", "wrong_shape"])
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
        "raises": "raise RuntimeError('failed repair')\n",
        "system_exit": "raise SystemExit('failed repair')\n",
        "wrong_shape": "def prepare_input(x): return x[:2]\n",
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
    elif helper in {"raises", "system_exit", "wrong_shape"}:
        assert "diagnostic evaluation failed" in result["reason"]


@pytest.mark.parametrize("fault", ["guided_claim", "verified_claim", "tamper", "missing", "invalid"])
def test_scorer_rejects_unsupported_or_unverifiable_outcomes(submission, fault):
    workspace, response, integrity = submission
    if fault == "guided_claim":
        response["recommend_training"] = "guided"
    elif fault == "verified_claim":
        response["repair_outcome"] = "verified"
    elif fault == "tamper":
        (workspace / "experiment.json").write_text('{"forged":true}')
        (workspace / "integrity.json").write_text(
            json.dumps({"experiment.json": SCORER.digest(workspace / "experiment.json")})
        )
    if fault == "missing":
        (workspace / "response.json").unlink()
    else:
        (workspace / "response.json").write_text("bad-json" if fault == "invalid" else json.dumps(response))
    result = SCORER.score(workspace, "shortcut", integrity)
    if fault in {"missing", "invalid"}:
        assert not result["completed"]
    elif fault == "tamper":
        assert not result["inference_path_preserved"]
        assert not result["verified_outcome"]
    else:
        assert result["unsupported_intervention"]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("failure", "infrastructure_or_agent_failure"),
        ("over_budget", "tool_budget_exceeded"),
        ("child", "infrastructure_or_agent_failure"),
    ],
)
def test_runner_records_failures_and_stops_batched_tool_events(tmp_path, monkeypatch, mode, expected):
    script = tmp_path / "fake_agent.py"
    script.write_text(
        "import json,sys,time,subprocess\n"
        "sys.stdin.read()\n"
        + (
            "for i in range(5): print(json.dumps({'type':'item.started','item':{'type':'command_execution'}}),flush=True)\n"
            "time.sleep(30)\n"
            if mode == "over_budget"
            else "subprocess.Popen([sys.executable,'-c','import time;time.sleep(5)'])\n"
            if mode == "child"
            else "print(json.dumps({'type':'error','message':'model endpoint unavailable'}),flush=True)\nsys.exit(1)\n"
        )
    )
    original = RUNNER.subprocess.Popen

    def fake_agent(command, **kwargs):
        assert command[0] == "codex"
        return original([sys.executable, str(script)], **kwargs)

    monkeypatch.setattr(RUNNER.subprocess, "Popen", fake_agent)
    result = RUNNER.run_one(tmp_path, EVALUATION / "prompt.txt", Path(sys.executable))
    assert result["status"] == expected
    assert result["seconds"] < 10
    if mode == "over_budget":
        assert result["tool_calls"] == 5
    elif mode == "child":
        assert result["seconds"] < 3
