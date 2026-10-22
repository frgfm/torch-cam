import re
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("streamlit")

from demo import inference


def install_transformers(monkeypatch):
    calls = []

    def loader(name):
        return SimpleNamespace(
            from_pretrained=lambda checkpoint, **options: calls.append((name, checkpoint, options)) or MagicMock()
        )

    module = ModuleType("transformers")
    for name in (
        "AutoModelForCausalLM",
        "AutoModelForImageTextToText",
        "AutoModelForSequenceClassification",
        "AutoProcessor",
        "AutoTokenizer",
        "BlipForConditionalGeneration",
    ):
        setattr(module, name, loader(name))
    monkeypatch.setitem(sys.modules, "transformers", module)
    return calls


@pytest.mark.parametrize(
    ("family", "model_name", "loader_name"),
    [
        ("vlm", inference.VLM_ID, "BlipForConditionalGeneration"),
        ("vlm", "Qwen/Qwen3-VL-2B-Instruct", "AutoModelForImageTextToText"),
        ("vlm", "google/gemma-4-E2B-it", "AutoModelForImageTextToText"),
        ("decision", "minilm", "AutoModelForSequenceClassification"),
        ("decision", "deberta", "AutoModelForSequenceClassification"),
    ],
)
def test_huggingface_loaders_pin_matching_safe_model_and_processor(monkeypatch, family, model_name, loader_name):
    calls = install_transformers(monkeypatch)
    if model_name.startswith("google/gemma-4"):
        monkeypatch.setattr(inference, "vlm_device_supported", lambda _name, _device: True)

    inference.load_resource.clear()
    _, _processor, _ = inference.load_resource(family, model_name, "cpu")

    model_call = next(call for call in calls if call[0] == loader_name)
    checkpoint = model_name if family == "vlm" else inference.NLI_MODELS[model_name]
    revisions = inference.VLM_REVISIONS if family == "vlm" else inference.NLI_REVISIONS
    revision = revisions[checkpoint]
    model_options = model_call[2]
    assert re.fullmatch(r"[a-f0-9]{40}", revision)
    assert model_call[1] == checkpoint
    assert model_options["revision"] == revision
    assert model_options["trust_remote_code"] is False
    assert model_options.get("use_safetensors", False) is (model_name != inference.VLM_ID)
    processor_name = "AutoProcessor" if family == "vlm" else "AutoTokenizer"
    processor_call = next(call for call in calls if call[0] == processor_name)
    assert processor_call[1] == checkpoint
    assert processor_call[2]["revision"] == revision
    assert processor_call[2]["trust_remote_code"] is False


def test_laya_agent_receives_reviewed_revision_without_transformers_remote_code(monkeypatch):
    calls = []
    agent = SimpleNamespace(model=MagicMock(), tok=object())
    laya = ModuleType("laya")
    laya.Agent = lambda model_id, **options: calls.append((model_id, options)) or agent
    monkeypatch.setitem(sys.modules, "laya", laya)

    inference.load_resource.clear()
    model, processor, _ = inference.load_resource("decision", "laya", "cpu")

    assert model is agent
    assert processor is agent.tok
    assert calls == [(inference.LAYA_ID, {"device": "cpu", "backend": "eager", "revision": inference.LAYA_REVISION})]


def test_decider_tokenizer_model_and_metadata_share_reviewed_revision(monkeypatch, tmp_path):
    calls = install_transformers(monkeypatch)
    metadata = tmp_path / "decider_config.json"
    metadata.write_text('{"temperature": 1.0}')
    hub = ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda checkpoint, filename, revision: (
        calls.append(("metadata", checkpoint, {"filename": filename, "revision": revision})) or str(metadata)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    inference.load_resource.clear()
    inference.load_resource("decision", "decider", "cpu")

    assert {call[2]["revision"] for call in calls} == {inference.DECIDER_REVISION}
    assert all(call[1] == inference.DECIDER_ID for call in calls)
    assert next(call for call in calls if call[0] == "AutoModelForCausalLM")[2]["use_safetensors"] is True


@pytest.mark.parametrize(
    ("call", "args"),
    [
        (inference.explain_decision, ("x" * 2001, "minilm", "cpu")),
        (inference.generate_answer, (object(), "x" * 257, "cpu")),
    ],
)
def test_direct_inference_limits_reject_before_model_load(monkeypatch, call, args):
    monkeypatch.setattr(inference, "load_resource", lambda *_args: pytest.fail("model loaded before validation"))

    with pytest.raises(ValueError):
        call(*args)
