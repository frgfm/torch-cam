"""Produce one genuine pretrained-Qwen prediction and a TorchCAM TAM overlay."""

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from torchcam.methods import TAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct", help="Hub ID or local checkpoint directory")
    parser.add_argument("--image", type=Path, default=Path(__file__).with_name("input.png"))
    parser.add_argument("--target-token", default="cats", help="Exact decoded answer token, ignoring surrounding whitespace")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    image = Image.open(args.image).convert("RGB")
    prompt = "Describe this image in one sentence."

    print(f"Loading pretrained {args.model} on {device}...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=56 * 56, max_pixels=448 * 448)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    print("Generating a greedy caption and retaining final language-model states...", flush=True)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True,
            return_dict_in_generate=True, output_hidden_states=True
        )
    generation_seconds = time.perf_counter() - started
    answer_ids = generated.sequences[:, inputs["input_ids"].shape[1]:]
    prediction = processor.batch_decode(answer_ids, skip_special_tokens=True)[0].strip()
    tokens = [processor.tokenizer.decode([idx]) for idx in answer_ids[0].tolist()]
    record = {
        "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
        "trained_checkpoint": True, "device": device, "dtype": str(dtype), "prompt": prompt,
        "prediction": prediction, "answer_ids": answer_ids[0].tolist(), "decoded_tokens": tokens,
        "input_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "input_source": "https://github.com/huggingface/transformers/blob/main/tests/fixtures/tests_samples/COCO/000000039769.png",
        "generation_seconds": generation_seconds,
    }
    (args.output_dir / "prediction.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Prediction: {prediction}", flush=True)
    matches = [i for i, token in enumerate(tokens) if token.strip().casefold() == args.target_token.casefold()]
    if not matches:
        raise ValueError(f"Requested token {args.target_token!r} not present. Actual tokens: {tokens!r}")
    step = matches[0]
    target_id = int(answer_ids[0, step])

    # The initial multimodal forward supplies the spatial visual states and prompt states.
    states = generated.hidden_states[0][-1]
    image_mask = inputs["input_ids"][0] == model.config.image_token_id
    special_ids = torch.tensor(processor.tokenizer.all_special_ids, device=device)
    text_mask = ~torch.isin(inputs["input_ids"][0], special_ids) & inputs["attention_mask"][0].bool()
    text_mask &= ~image_mask
    grid = inputs["image_grid_thw"][0]
    if int(grid[0]) != 1 or inputs["image_grid_thw"].shape[0] != 1:
        raise ValueError("This demo requires exactly one still image.")
    merge = model.config.vision_config.spatial_merge_size
    height, width = (int(dim) // merge for dim in grid[1:])
    visual = states[:, image_mask].reshape(1, height, width, states.shape[-1])
    context, context_ids = states[:, text_mask], inputs["input_ids"][:, text_mask]
    if step:
        # Align earlier generated IDs with the states that predicted those IDs.
        previous_states = torch.cat([generated.hidden_states[i][-1][:, -1:] for i in range(step)], dim=1)
        previous_ids = answer_ids[:, :step]
        keep = ~torch.isin(previous_ids[0], special_ids)
        context = torch.cat((context, previous_states[:, keep]), dim=1)
        context_ids = torch.cat((context_ids, previous_ids[:, keep]), dim=1)

    started = time.perf_counter()
    maps = TAM(model.get_output_embeddings())(target_id, visual, context, context_ids)
    if device == "cuda":
        torch.cuda.synchronize()
    map_seconds = time.perf_counter() - started
    raw = maps[0].float().cpu().numpy()
    np.save(args.output_dir / "tam.npy", raw)
    resized = F.interpolate(maps.float().unsqueeze(1), image.size[::-1], mode="bilinear", align_corners=False)
    overlay_values = resized[0, 0].cpu().numpy()

    # Save only the vocabulary rows used by this explanation, for an offline replay.
    needed_ids, inverse = torch.unique(torch.cat((context_ids.flatten(), context_ids.new_tensor([target_id]))), return_inverse=True)
    np.savez_compressed(
        args.output_dir / "replay.npz", visual=visual.float().cpu().numpy(), context=context.float().cpu().numpy(),
        compact_context_ids=inverse[:-1].reshape(context_ids.shape).cpu().numpy(),
        compact_target_id=inverse[-1:].cpu().numpy(), original_vocabulary_ids=needed_ids.cpu().numpy(),
        head_weights=model.get_output_embeddings().weight[needed_ids].detach().float().cpu().numpy(),
    )
    record.update({
        "selected_token": tokens[step], "selected_token_id": target_id, "answer_token_position": step,
        "grid_shape": [height, width], "context_ids": context_ids[0].tolist(),
        "tam_seconds": map_seconds, "tam_min": float(raw.min()), "tam_max": float(raw.max()),
        "normalization": "independent min-max per image", "smoothing_kernel": 3,
    })
    (args.output_dir / "prediction.json").write_text(json.dumps(record, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), layout="constrained")
    axes[0].imshow(image)
    axes[0].set_title("Input photograph")
    axes[1].imshow(image)
    colored = axes[1].imshow(overlay_values, cmap="turbo", vmin=0, vmax=1, alpha=0.45)
    axes[1].set_title(f"TAM for token {tokens[step].strip()!r} · {height}×{width} grid")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(colored, ax=axes[1], fraction=0.035, pad=0.02, label="Relative activation")
    fig.suptitle("Prediction: " + textwrap.fill(prediction, width=100), fontsize=13)
    fig.savefig(args.output_dir / "result.png", dpi=170, facecolor="white")
    plt.close(fig)
    print(f"Selected token: {tokens[step]!r}, ID {target_id}, grid {height}×{width}", flush=True)
    print(f"Saved {args.output_dir / 'result.png'} and prediction.json", flush=True)


if __name__ == "__main__":
    main()
