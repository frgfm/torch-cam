import pytest
import torch

from scripts import dexar_example
from torchcam.methods import DEXAR


@pytest.fixture
def qwen_inputs():
    # Optional architectural check: tiny random weights, no downloads, and no mandatory Transformers dependency.
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.51.3":
        pytest.skip("the example targets Transformers 4.51.3")
    config = transformers.Qwen2_5_VLConfig(
        vocab_size=32,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=3,
        num_key_value_heads=1,
        rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
        image_token_id=29,
        vision_start_token_id=28,
        vision_end_token_id=30,
        video_token_id=31,
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 24,
            "window_size": 8,
            "fullatt_block_indexes": [0],
        },
        attn_implementation="eager",
    )
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = transformers.Qwen2_5_VLForConditionalGeneration(config).eval().requires_grad_(False)
        pixels = torch.randn(24, 24)
    # Nonuniform final norm weights expose accidental double normalization.
    model.model.norm.weight.copy_(torch.linspace(0.5, 1.5, 24))
    ids = torch.tensor([[1, 28, 29, 29, 29, 29, 29, 29, 30, 3]])
    inputs = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "image_grid_thw": torch.tensor([[1, 4, 6]]),
        "pixel_values": pixels,
    }
    return model, inputs


def test_qwen_token_alignment_intermediate_logits_and_hook_cleanup(qwen_inputs, monkeypatch):
    model, inputs = qwen_inputs
    answer = torch.tensor([[4, 5, 4]])
    prefixes = []
    current = None

    def capture(_module, _args, kwargs, output):
        nonlocal current
        prefixes.append(kwargs["input_ids"].clone())
        assert output.logits.shape == (1, 1, 32)  # Replay projects only the predicting state.
        current = output

    original_call = DEXAR.__call__

    def check_logits(self, scores, attentions, visual_mask, **kwargs):
        token = int(answer[0, len(prefixes) - 1])
        expected = model.lm_head(model.model.norm(current.hidden_states[1][:, -1]))[:, token]
        torch.testing.assert_close(scores[0], expected)
        torch.testing.assert_close(scores[-1], current.logits[:, -1, token])
        return original_call(self, scores, attentions, visual_mask, **kwargs)

    monkeypatch.setattr(DEXAR, "__call__", check_logits)
    handle = model.register_forward_hook(capture, with_kwargs=True)
    try:
        maps, weights, sequence, timings = dexar_example.explain_qwen(model, inputs, answer)
    finally:
        handle.remove()
    for step, prefix in enumerate(prefixes):
        torch.testing.assert_close(prefix, torch.cat([inputs["input_ids"], answer[:, :step]], dim=1))
    assert maps.shape == (1, 3, 2, 3)
    assert weights.shape == (1, 3)
    assert sequence.shape == (1, 2, 3)
    assert sequence.isfinite().all()
    assert all(value >= 0 for value in timings.values())
    assert inputs["input_ids"].shape[1] == 10
    assert not model.get_input_embeddings()._forward_hooks
    assert not model.lm_head._forward_pre_hooks
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_qwen_hook_cleanup_on_failed_forward(qwen_inputs, monkeypatch):
    model, inputs = qwen_inputs

    def fail(**_kwargs):
        raise RuntimeError("failed forward")

    monkeypatch.setattr(model, "forward", fail)
    with pytest.raises(RuntimeError, match="failed forward"):
        dexar_example.explain_qwen(model, inputs, torch.tensor([[4]]))
    assert not model.get_input_embeddings()._forward_hooks
    assert not model.lm_head._forward_pre_hooks


def test_qwen_rejects_multiple_frames(qwen_inputs):
    model, inputs = qwen_inputs
    inputs["image_grid_thw"][0, 0] = 2
    with pytest.raises(ValueError, match="still image"):
        dexar_example.explain_qwen(model, inputs, torch.tensor([[4]]))
