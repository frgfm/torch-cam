# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Qwen2.5-VL example: install Transformers 4.51.3, then pass --image and --output (outside Git)."""

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from torchcam.methods import DEXAR, TAM
from torchcam.utils import overlay_mask


@contextmanager
def timer(timings, key, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings[key] = timings.get(key, 0) + time.perf_counter() - started


@torch.enable_grad()
def explain_qwen(model, inputs, answer_ids, special_ids):  # noqa: PLR0915
    """Replay the actual generated IDs; each prefix excludes the token currently being explained.

    Returns:
        token maps, relevance weights, sequence map, TAM token maps, and separate attribution timings

    Raises:
        ValueError: if the model or inputs fall outside this example's supported configuration
    """
    if model.training or model.config._attn_implementation != "eager":  # noqa: SLF001
        raise ValueError("use eval mode and attn_implementation='eager'")
    ids = inputs["input_ids"]
    grid = inputs["image_grid_thw"]
    if (
        ids.shape[0] != 1
        or grid.shape != (1, 3)
        or int(grid[0, 0]) != 1
        or not inputs["attention_mask"].all()
        or "pixel_values_videos" in inputs
    ):
        raise ValueError("this example requires batch size one and one unpadded still image")
    if answer_ids.ndim != 2 or answer_ids.shape[0] != 1 or answer_ids.shape[1] == 0:
        raise ValueError("provide at least one generated token, shaped (1, tokens)")
    decoder = model.model  # Transformers 4.51.3 Qwen2.5-VL layout.
    num_layers = len(decoder.layers)
    merge = model.config.vision_config.spatial_merge_size
    height, width = (int(dim) // merge for dim in grid[0, 1:])
    image_mask = ids[0] == model.config.image_token_id
    if int(image_mask.sum()) != height * width:
        raise ValueError("image placeholders must match the merged visual grid")
    head = model.get_output_embeddings()
    extractor, tam = DEXAR((height, width)), TAM(head)
    text_mask = ~image_mask & ~torch.isin(ids[0], torch.tensor(special_ids, device=ids.device))
    prefix = dict(inputs)
    maps, weights, tam_maps = [], [], []
    timings = {}
    # Differentiate frozen embeddings; avoid projecting every prefix position onto the full vocabulary.
    handles = [
        model.get_input_embeddings().register_forward_hook(lambda _module, _args, output: output.requires_grad_()),
        head.register_forward_pre_hook(lambda _module, args: (args[0][:, -1:] if args[0].ndim == 3 else args[0],)),
    ]
    try:
        for step in range(answer_ids.shape[1]):
            token_id = int(answer_ids[0, step])
            with timer(timings, "forward", ids.device):
                output = model(**prefix, output_attentions=True, output_hidden_states=True, use_cache=False)
            with timer(timings, "dexar", ids.device):
                layer_scores = []
                for layer in range(num_layers):
                    state = output.hidden_states[layer + 1][:, -1]
                    if layer < num_layers - 1:  # Final returned state already includes decoder.norm.
                        state = decoder.norm(state)
                    bias = None if head.bias is None else head.bias[token_id : token_id + 1]
                    layer_scores.append(F.linear(state, head.weight[token_id : token_id + 1], bias).squeeze(-1))
                mask = prefix["input_ids"][0] == model.config.image_token_id
                token_map, weight = extractor(layer_scores, output.attentions, mask)
            maps.append(token_map)
            weights.append(weight)

            with timer(timings, "tam", ids.device):
                final_states = output.hidden_states[-1].detach()
                if step == 0:
                    visual = final_states[:, image_mask].reshape(1, height, width, final_states.shape[-1])
                    context, context_ids = final_states[:, text_mask], ids[:, text_mask]
                tam_maps.append(tam(token_id, visual, context, context_ids))
                # TorchCAM TAM pairs earlier IDs with the final states that predicted them.
                context = torch.cat([context, final_states[:, -1:]], dim=1)
                context_ids = torch.cat([context_ids, answer_ids[:, step : step + 1]], dim=1)
            # Append only after attributing this token.
            prefix["input_ids"] = torch.cat([prefix["input_ids"], answer_ids[:, step : step + 1]], dim=1)
            prefix["attention_mask"] = torch.ones_like(prefix["input_ids"])
            del output, layer_scores, state
    finally:
        for handle in handles:
            handle.remove()
    maps, weights = torch.stack(maps, dim=1), torch.stack(weights, dim=1)
    with timer(timings, "dexar", ids.device):
        sequence = extractor.aggregate(maps, weights)
    return maps, weights, sequence, torch.stack(tam_maps, dim=1), timings


def save_overlays(image, tokens, maps, weights, sequence, tam_maps, output_dir):
    fig, axes = plt.subplots(len(tokens) + 1, 2, figsize=(8, 2.5 * (len(tokens) + 1)), squeeze=False)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input image")
    axes[0, 1].imshow(overlay_mask(image, Image.fromarray(sequence[0].cpu().numpy(), mode="F")))
    axes[0, 1].set_title("DEX-AR sequence")
    for idx, token in enumerate(tokens):
        for column, (method, heatmaps) in enumerate((("DEX-AR", maps), ("TorchCAM TAM", tam_maps))):
            axes[idx + 1, column].imshow(
                overlay_mask(image, Image.fromarray(heatmaps[0, idx].float().cpu().numpy(), mode="F"))
            )
            suffix = f"; weight={float(weights[0, idx]):.3g}" if column == 0 else ""
            axes[idx + 1, column].set_title(f"{idx}: {token!r} — {method}{suffix}")
    for axis in axes.flat:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "overlays.png", dpi=120)
    plt.close(fig)
    image.save(output_dir / "input.png")


def main(args):
    import transformers  # noqa: PLC0415
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: PLC0415

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    image = Image.open(args.image).convert("RGB")
    timings = {}
    with timer(timings, "load", device):
        processor = AutoProcessor.from_pretrained(
            args.model, revision=args.revision, min_pixels=4 * 28 * 28, max_pixels=args.max_pixels
        )
        model = (
            Qwen2_5_VLForConditionalGeneration
            .from_pretrained(args.model, revision=args.revision, torch_dtype=dtype, attn_implementation="eager")
            .to(device)
            .eval()
            .requires_grad_(False)
        )
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    with timer(timings, "generation", device), torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
    answer_ids = generated[:, inputs["input_ids"].shape[1] :]
    # Retain original generated IDs; never decode and retokenize an answer for attribution.
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos or []
    eos_terminated = bool(answer_ids.shape[1] and int(answer_ids[0, -1]) in eos)
    if eos_terminated:
        answer_ids = answer_ids[:, :-1]
    maps, weights, sequence, tam_maps, attribution = explain_qwen(
        model, inputs, answer_ids, processor.tokenizer.all_special_ids
    )
    answer = processor.tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    tokens = [processor.tokenizer.decode([int(token)]) for token in answer_ids[0]]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_overlays(image, tokens, maps, weights, sequence, tam_maps, output_dir)
    report = {
        **vars(args),
        "revision": model.config._commit_hash,  # noqa: SLF001
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "answer": answer,
        "generated_ids": generated[0, inputs["input_ids"].shape[1] :].tolist(),
        "explained_ids": answer_ids[0].tolist(),
        "token_weights": weights[0].tolist(),
        "grid_shape": list(maps.shape[-2:]),
        "layers": len(model.model.layers),
        "eos_terminated": eos_terminated,
        "inference": {"attention": "eager", "do_sample": False, "generation_cache": True, "attribution_cache": False},
        "seconds": {**timings, **attribution, "attribution": attribution["forward"] + attribution["dexar"]},
        "note": "One demonstration, not an accuracy benchmark. TAM timing excludes shared prefix forwards.",
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True, help="artifact directory outside the repository")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--prompt", default="Describe the image in one short sentence.")
    parser.add_argument("--max-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    main(parser.parse_args())
