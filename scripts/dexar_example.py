# Copyright (C) 2020-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Generate and explain an answer with Qwen2.5-VL (Transformers 4.51.3).

Optional setup: uv pip install 'transformers==4.51.3'
Run: python scripts/dexar_example.py --image /path/to/image.jpg --output /tmp/dexar-demo
Weights and output artifacts stay outside Git. One image, one unpadded prompt, no video or quantization.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from torchcam.methods import DEXAR, TAM
from torchcam.utils import overlay_mask


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def explain_qwen(model, inputs, answer_ids, special_ids, start_layer=0):  # noqa: PLR0915
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
    if ids.shape[0] != 1 or grid.shape != (1, 3) or int(grid[0, 0]) != 1:
        raise ValueError("this example supports batch size one and one still image")
    if not inputs["attention_mask"].all() or "pixel_values_videos" in inputs:
        raise ValueError("this example requires an unpadded image prompt without video")
    if answer_ids.ndim != 2 or answer_ids.shape[0] != 1 or answer_ids.shape[1] == 0:
        raise ValueError("provide at least one generated token, shaped (1, tokens)")
    decoder = model.model  # Transformers 4.51.3 Qwen2.5-VL layout.
    num_layers = len(decoder.layers)
    if not 0 <= start_layer < num_layers:
        raise ValueError("start_layer must select at least one decoder layer")
    merge = model.config.vision_config.spatial_merge_size
    if (grid[0, 1:] % merge).any():
        raise ValueError("image grid dimensions must be divisible by spatial_merge_size")
    height, width = (int(dim) // merge for dim in grid[0, 1:])
    image_mask = ids[0] == model.config.image_token_id
    if int(image_mask.sum()) != height * width:
        raise ValueError("image placeholders must match the merged visual grid")
    extractor, tam = DEXAR((height, width)), TAM(model.get_output_embeddings())
    text_mask = ~image_mask & ~torch.isin(ids[0], torch.tensor(special_ids, device=ids.device))
    prefix = dict(inputs)
    maps, weights, tam_maps = [], [], []
    timings = {"attribution_forward_seconds": 0.0, "dexar_seconds": 0.0, "tam_seconds": 0.0}
    # Frozen weights save memory. Make the decoder's embedding output differentiable without changing parameters.
    handle = model.get_input_embeddings().register_forward_hook(lambda _module, _args, output: output.requires_grad_())
    try:
        with torch.enable_grad():
            for step in range(answer_ids.shape[1]):
                token_id = int(answer_ids[0, step])
                synchronize(ids.device)
                started = time.perf_counter()
                output = model(**prefix, output_attentions=True, output_hidden_states=True, use_cache=False)
                synchronize(ids.device)
                timings["attribution_forward_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                layer_scores = []
                for layer in range(start_layer, num_layers):
                    state = output.hidden_states[layer + 1][:, -1]
                    # The final returned state already includes decoder.norm.
                    if layer < num_layers - 1:
                        state = decoder.norm(state)
                    head = model.get_output_embeddings()
                    bias = None if head.bias is None else head.bias[token_id : token_id + 1]
                    layer_scores.append(F.linear(state, head.weight[token_id : token_id + 1], bias).squeeze(-1))
                # Appending after attribution keeps token t aligned with the state predicting it.
                mask = prefix["input_ids"][0] == model.config.image_token_id
                token_map, weight = extractor(layer_scores, output.attentions[start_layer:], mask)
                synchronize(ids.device)
                timings["dexar_seconds"] += time.perf_counter() - started
                maps.append(token_map)
                weights.append(weight)

                started = time.perf_counter()
                final_states = output.hidden_states[-1].detach()
                if step == 0:
                    visual = final_states[:, image_mask].reshape(1, height, width, final_states.shape[-1])
                    context = final_states[:, text_mask]
                    context_ids = ids[:, text_mask]
                tam_maps.append(tam(token_id, visual, context, context_ids))
                # Match TorchCAM TAM's documented generated-context convention: states that predicted earlier IDs.
                context = torch.cat([context, final_states[:, -1:]], dim=1)
                context_ids = torch.cat([context_ids, answer_ids[:, step : step + 1]], dim=1)
                synchronize(ids.device)
                timings["tam_seconds"] += time.perf_counter() - started
                prefix["input_ids"] = torch.cat([prefix["input_ids"], answer_ids[:, step : step + 1]], dim=1)
                prefix["attention_mask"] = torch.ones_like(prefix["input_ids"])
                # Drop the previous graph before constructing the next prefix's graph.
                del output, layer_scores, state
    finally:
        handle.remove()
    maps, weights = torch.stack(maps, dim=1), torch.stack(weights, dim=1)
    started = time.perf_counter()
    sequence = extractor.aggregate(maps, weights)
    synchronize(ids.device)
    timings["dexar_seconds"] += time.perf_counter() - started
    return maps, weights, sequence, torch.stack(tam_maps, dim=1), timings


def save_overlays(image, tokens, maps, weights, sequence, tam_maps, output_dir):
    # Show all actual tokens, including zero-weight tokens, alongside the real TorchCAM TAM implementation.
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
    started = time.perf_counter()
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
    if not isinstance(model.get_output_embeddings(), nn.Linear):
        raise TypeError("this example requires an unquantized linear vocabulary head")
    synchronize(device)
    load_seconds = time.perf_counter() - started
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": args.prompt}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
    synchronize(device)
    generation_seconds = time.perf_counter() - started
    answer_ids = generated[:, inputs["input_ids"].shape[1] :]
    # Retain original generated IDs; never decode and retokenize an answer for attribution.
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else eos
    if answer_ids.shape[1] and int(answer_ids[0, -1]) in eos:
        answer_ids = answer_ids[:, :-1]
    maps, weights, sequence, tam_maps, timings = explain_qwen(
        model, inputs, answer_ids, processor.tokenizer.all_special_ids, args.start_layer
    )
    answer = processor.tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    tokens = [processor.tokenizer.decode([int(token)]) for token in answer_ids[0]]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_overlays(image, tokens, maps, weights, sequence, tam_maps, output_dir)
    report = {
        "model": args.model,
        "revision": model.config._commit_hash,  # noqa: SLF001
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "dtype": args.dtype,
        "attention": "eager",
        "prompt": args.prompt,
        "answer": answer,
        "generated_ids": generated[0, inputs["input_ids"].shape[1] :].tolist(),
        "explained_ids": answer_ids[0].tolist(),
        "tokens": tokens,
        "token_weights": weights[0].tolist(),
        "image_size": image.size,
        "image_grid_thw": inputs["image_grid_thw"].tolist(),
        "grid_shape": list(maps.shape[-2:]),
        "layers": [args.start_layer, len(model.model.layers)],
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "generation_use_cache": True,
        "attribution_use_cache": False,
        "tam_kernel_size": 3,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        **timings,
        "attribution_seconds": timings["attribution_forward_seconds"] + timings["dexar_seconds"],
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
    parser.add_argument("--start-layer", type=int, default=0)
    main(parser.parse_args())
