import importlib.util
import json
import sys
from pathlib import Path

import pytest

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
    artifact.write_text("{}")
    evidence = {
        "provenance": {"experiment_sha256": SCORER.digest(artifact)},
        "independent_evaluation": {"runs": [{"accuracy": 1.0, "worst_group": 1.0}] * 9},
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
    ("mode", "expected"), [("failure", "infrastructure_or_agent_failure"), ("over_budget", "tool_budget_exceeded")]
)
def test_runner_records_failures_and_stops_batched_tool_events(tmp_path, monkeypatch, mode, expected):
    script = tmp_path / "fake_agent.py"
    script.write_text(
        "import json,sys,time\n"
        "sys.stdin.read()\n"
        + (
            "for i in range(5): print(json.dumps({'type':'item.started','item':{'type':'command_execution'}}),flush=True)\n"
            "time.sleep(30)\n"
            if mode == "over_budget"
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
