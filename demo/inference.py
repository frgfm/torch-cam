# Copyright (C) 2021-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import json
from operator import itemgetter
from pathlib import Path
from threading import Lock
from time import perf_counter

import streamlit as st
import torch
from PIL import ImageOps
from torchvision.transforms.functional import to_pil_image

from torchcam.methods import LayerCAM, TokenAblation

DECISION_MODELS = {
    "laya": "Laya",
    "minilm": "MiniLM · zero-shot classifier",
    "deberta": "DeBERTa · zero-shot classifier",
    "decider": "Decider · 0.8B",
}
DECIDER_ID = "Mapika/decider-0.8b"
NLI_MODELS = {
    "minilm": "cross-encoder/nli-MiniLM2-L6-H768",
    "deberta": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
}
VLM_ID = "Salesforce/blip-image-captioning-base"
VLM_MODELS = {
    VLM_ID: "BLIP captioner",
    "Qwen/Qwen3-VL-2B-Instruct": "Qwen3-VL · 2B",
    "google/gemma-4-E2B-it": "Gemma 4 · E2B",
}
QUESTION = {
    "type": "choice",
    "instructions": "Route this support ticket by the issue described in the message.",
    "criteria": {
        "Billing": "Charges, payments, invoices, or refunds.",
        "Account access": "Signing in, passwords, or account access.",
        "Delivery": "Shipment tracking, delays, or missing deliveries.",
    },
}
MODEL_INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "position_ids", "option_ids")


@st.cache_resource(show_spinner=False, max_entries=1)
def load_resource(family, model_name, device):
    """Load only a requested model; share one cache across the three pages.

    Returns:
        the model, its processor and a lock for serialized inference

    Raises:
        ValueError: if the requested model is unsupported
    """
    if family == "image":
        from torchvision.models import get_model, get_model_weights  # noqa: PLC0415

        weights = get_model_weights(model_name).DEFAULT
        model, processor = get_model(model_name, weights=weights).eval().to(device), weights
    elif family == "decision" and model_name == "laya":
        from laya import Agent  # noqa: PLC0415

        model = Agent("convaiinnovations/laya", device=device, backend="eager")
        # Match attribution precision and the native decoder's probabilities, including on CUDA.
        model.model.float().requires_grad_(False)
        model.amp_enabled = False
        model.dtype = torch.float32
        processor = model.tok
    else:
        from transformers import (  # noqa: PLC0415
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoModelForSequenceClassification,
            AutoProcessor,
            AutoTokenizer,
            BlipForConditionalGeneration,
        )

        if family == "vlm" and model_name in VLM_MODELS:
            processor = AutoProcessor.from_pretrained(model_name)
            loader = BlipForConditionalGeneration if model_name == VLM_ID else AutoModelForImageTextToText
            full_precision = model_name == VLM_ID or (device == "cpu" and model_name.startswith("Qwen/"))
            dtype = torch.float32 if full_precision else torch.bfloat16
            model = loader.from_pretrained(model_name, dtype=dtype, attn_implementation="eager").eval().to(device)
        elif family == "decision" and model_name == "decider":
            from huggingface_hub import hf_hub_download  # noqa: PLC0415

            processor = AutoTokenizer.from_pretrained(DECIDER_ID, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(DECIDER_ID, dtype=torch.float32).eval().to(device)
            metadata = json.loads(
                Path(
                    hf_hub_download(
                        DECIDER_ID,
                        "decider_config.json",
                        revision=model.config._commit_hash,  # noqa: SLF001
                    )
                ).read_text(encoding="utf-8")
            )
            model.config.decision_temperature = (metadata.get("temperature_by_type") or {}).get(
                "choice", metadata["temperature"]
            )
        elif family == "decision" and model_name in NLI_MODELS:
            checkpoint = NLI_MODELS[model_name]
            processor = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
            model = (
                AutoModelForSequenceClassification.from_pretrained(checkpoint, dtype=torch.float32).eval().to(device)
            )
        else:
            raise ValueError("Unknown demo model")
        model.requires_grad_(False)
    return model, processor, Lock()


def explain_laya(agent, text):
    from laya.common import collate_items, render_options  # noqa: PLC0415

    if agent.tok.mask_token and agent.tok.mask_token in text:
        raise ValueError("Remove the reserved mask token from the ticket before explaining it.")
    agent._check_question("route", QUESTION)  # noqa: SLF001
    internal = {"route": agent._to_internal(QUESTION)}  # noqa: SLF001
    (item,) = agent._encode_state(text, ["route"], internal)  # noqa: SLF001
    if item["state_stats"]["state_tokens_dropped"]:
        raise ValueError("This ticket exceeds Laya's context. Shorten it so every displayed token can be explained.")
    if item["options"]["options_distinct"] != len(render_options(internal["route"])):
        raise ValueError("Laya's context shortened the routing options. Use a checkpoint with a larger context.")
    encoded = agent.tok(text, add_special_tokens=False, return_offsets_mapping=True)
    start = len(item["ids"]) - len(encoded["input_ids"]) - 1
    state_slice = slice(start, len(item["ids"]) - 1)
    if item["ids"][state_slice] != encoded["input_ids"]:
        raise ValueError("Ticket tokens do not align with Laya's encoded input.")
    batch = collate_items([[item]], agent.tok.pad_token_id)
    inputs = {key: batch[key].to(agent.device) for key in MODEL_INPUTS if key in batch}
    mask = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
    mask[:, state_slice] = True
    result = ablate_ticket(agent.model, inputs, mask, agent.tok.mask_token_id, itemgetter(0))
    prediction = agent.predict(text, {"route": QUESTION})["answers"]["route"]
    probabilities = [prediction["probabilities"][label] for label in QUESTION["criteria"]]
    maps = result.attributions[0, :, state_slice].cpu().tolist()
    offsets = [encoded["offset_mapping"]] * len(maps)
    return result, maps, offsets, probabilities, "option logit"


def explain_nli(model, tokenizer, text):
    hypotheses = [f"This ticket is about {description.lower()}" for description in QUESTION["criteria"].values()]
    encoded = tokenizer(
        [text] * len(hypotheses),
        hypotheses,
        padding=True,
        truncation=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    limit = min(tokenizer.model_max_length, model.config.max_position_embeddings - 2)
    if encoded["input_ids"].shape[1] > limit:
        raise ValueError(
            "This ticket exceeds the model's context. Shorten it so every displayed token can be explained."
        )
    offsets = encoded.pop("offset_mapping").tolist()
    inputs = {key: value.to(model.device) for key, value in encoded.items()}
    mask = torch.tensor(
        [[sequence == 0 for sequence in encoded.sequence_ids(row)] for row in range(len(hypotheses))],
        device=model.device,
    )
    entailment = next(index for index, label in model.config.id2label.items() if label.lower() == "entailment")
    result = ablate_ticket(
        model, inputs, mask, tokenizer.mask_token_id, lambda output: output.logits[:, entailment : entailment + 1]
    )
    probabilities = result.scores[:, 0].softmax(0).cpu().tolist()
    maps = result.attributions[:, 0].cpu().tolist()
    offsets = [
        [offset if keep else [0, 0] for offset, keep in zip(row, selected, strict=True)]
        for row, selected in zip(offsets, mask.cpu().tolist(), strict=True)
    ]
    return result, maps, offsets, probabilities, "entailment logit"


def ablate_ticket(model, inputs, mask, replacement_id, output_transform):
    if mask.any(0).sum() > 64:
        raise ValueError("This ticket is too long for an interactive explanation. Please shorten it.")
    return TokenAblation(model, output_transform)(inputs, replacement_id, token_mask=mask)


def explain_decider(model, tokenizer, text):
    # Native plain state-first layout: tokenize context and question separately, with no generated answer letters.
    header = "Context:\n"
    state = tokenizer(header + text, add_special_tokens=False, return_offsets_mapping=True)
    question = f"\n\nQuestion: {QUESTION['instructions']}\nOptions:"
    question += "".join(
        f"\n({letter}) {label}: {description}"
        for letter, (label, description) in zip(
            "ABC",
            QUESTION["criteria"].items(),
            strict=True,
        )
    )
    tail = tokenizer.encode(question + "\nAnswer: (", add_special_tokens=False)
    ids = state["input_ids"] + tail
    if len(ids) > model.config.get_text_config().max_position_embeddings:
        raise ValueError("This ticket exceeds the model's context. Shorten it before explaining.")
    offsets = [[max(0, start - len(header)), max(0, end - len(header))] for start, end in state["offset_mapping"]]
    offsets += [[0, 0]] * len(tail)
    inputs = {"input_ids": torch.tensor([ids], device=model.device), "use_cache": False, "logits_to_keep": 1}
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    mask = torch.tensor([[end > start for start, end in offsets]], device=model.device)
    letters = [tokenizer.encode(letter, add_special_tokens=False) for letter in "ABC"]
    space = tokenizer.encode(" ", add_special_tokens=False)
    if any(len(ids) != 1 for ids in [*letters, space]) or len({ids[0] for ids in letters}) != 3:
        raise ValueError("Decider requires single-token option letters and a single-token space baseline")
    columns = [ids[0] for ids in letters]
    result = ablate_ticket(model, inputs, mask, space[0], lambda output: output.logits[:, -1, columns].float())
    temperature = float(model.config.decision_temperature)
    if not 0 < temperature < float("inf"):
        raise ValueError("Invalid decision temperature")
    probabilities = (result.scores[0] / temperature).softmax(-1).cpu().tolist()
    maps = result.attributions[0].cpu().tolist()
    return result, maps, [offsets] * len(maps), probabilities, "option logit"


def explain_decision(text, model_name, device):
    model, processor, lock = load_resource("decision", model_name, device)
    with lock:
        started_at = perf_counter()
        if model_name == "laya":
            result, maps, offsets, probabilities, target = explain_laya(model, text)
        else:
            explain = explain_decider if model_name == "decider" else explain_nli
            result, maps, offsets, probabilities, target = explain(model, processor, text)
    return {
        "model": DECISION_MODELS[model_name],
        "text": text,
        "method": "TokenAblation",
        "target": target,
        "question": QUESTION,
        "baseline": "one ticket token replaced at a time; question/options held fixed",
        "replacement_token": " " if model_name == "decider" else processor.mask_token,
        "labels": list(QUESTION["criteria"]),
        "probabilities": probabilities,
        "attributions": maps,
        "offsets": offsets,
        "scores": result.scores.cpu().flatten().tolist(),
        "elapsed": perf_counter() - started_at,
    }


def prepare_vlm_inputs(model, processor, image, prompt, device):
    if model.config.model_type == "blip":
        size = model.config.vision_config.image_size
        model_input = ImageOps.fit(image, (size, size))
        inputs = processor(text=prompt, images=model_input, return_tensors="pt").to(device)
        side = size // model.config.vision_config.patch_size
        return inputs, model_input, (side, side)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the image in one short sentence."},
            ],
        }
    ]
    image_kwargs = {"max_pixels": 384**2} if model.config.model_type == "qwen3_vl" else {"max_soft_tokens": 70}
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        **image_kwargs,
    ).to(device)
    _, _, indices, (height, width) = vlm_target(model, inputs)
    patch = model.config.vision_config.patch_size
    pixels = inputs["pixel_values"].detach().float().cpu().reshape(-1, inputs["pixel_values"].shape[-1])[indices.cpu()]
    if model.config.model_type == "qwen3_vl":
        temporal = model.config.vision_config.temporal_patch_size
        pixels = pixels.reshape(height, width, 3, temporal, patch, patch)[:, :, :, 0].permute(2, 0, 3, 1, 4)
        pixels = pixels.reshape(3, height * patch, width * patch)
        mean = torch.tensor(processor.image_processor.image_mean)[:, None, None]
        std = torch.tensor(processor.image_processor.image_std)[:, None, None]
        pixels = pixels * std + mean
    else:
        pixels = pixels.reshape(height, width, patch, patch, 3).permute(4, 0, 2, 1, 3)
        pixels = pixels.reshape(3, height * patch, width * patch)
    return inputs, to_pil_image(pixels.clamp(0, 1)), (height, width)


def vlm_target(model, inputs):
    """Adapt native patch coordinates to LayerCAM.

    Returns:
        target layer, reshape function, spatially ordered patch indices and grid dimensions

    Raises:
        ValueError: if the input is not one image with a supported rectangular layout
    """
    if model.config.model_type == "qwen3_vl":
        if inputs["image_grid_thw"].shape != (1, 3) or inputs["image_grid_thw"][0, 0] != 1:
            raise ValueError("The demo expects one still image")
        _, height, width = inputs["image_grid_thw"][0].tolist()
        merge = model.config.vision_config.spatial_merge_size
        indices = torch.arange(height * width, device=inputs["pixel_values"].device)
        indices = indices.reshape(height // merge, width // merge, merge, merge).permute(0, 2, 1, 3).flatten()
        # Before DeepStack branches: gradients include every visual route into the language model.
        layer = model.model.visual.patch_embed
    elif model.config.model_type == "gemma4":
        positions = inputs["image_position_ids"]
        if positions.shape[0] != 1:
            raise ValueError("The demo expects one still image")
        valid = (positions[0] >= 0).all(-1)
        width, height = (positions[0, valid].amax(0) + 1).tolist()
        indices = valid.nonzero().flatten()
        indices = indices[(positions[0, valid, 1] * width + positions[0, valid, 0]).argsort()]
        if indices.numel() != height * width:
            raise ValueError("Image patches must form a complete rectangular grid")
        layer = model.model.vision_tower.patch_embedder
    else:
        raise ValueError("Unsupported visual feature layout")
    reshape = lambda states: states.float().reshape(-1, states.shape[-1])[indices].T.reshape(1, -1, height, width)
    return layer, reshape, indices, (height, width)


def answer_inputs(result, token_index, device):
    """Replay the prefix, preserving image metadata and multimodal token types.

    Returns:
        model inputs with regular tensors, including a differentiable pixel tensor
    """
    inputs = {key: value.to(device).clone() for key, value in result["inputs"].items()}
    inputs["input_ids"] = torch.tensor([result["generated_ids"][:token_index]], device=device)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    for key in ("mm_token_type_ids", "token_type_ids"):
        if key in inputs:
            inputs[key] = torch.nn.functional.pad(inputs[key], (0, token_index - inputs[key].shape[1]))
    inputs["pixel_values"].requires_grad_(True)
    return inputs


def generate_answer(image, prompt, device, model_name=VLM_ID):
    model, processor, lock = load_resource("vlm", model_name, device)
    with lock, torch.inference_mode():
        inputs, model_input, grid = prepare_vlm_inputs(model, processor, image, prompt, device)
        blip = model.config.model_type == "blip"
        start = inputs["input_ids"].shape[1] - int(blip)  # BLIP replaces CLS with BOS and removes the final SEP.
        sequences = model.generate(
            **inputs,
            do_sample=False,
            num_beams=3 if blip else 1,
            use_cache=not blip,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            max_new_tokens=24,
            return_dict_in_generate=False,
        )
        ids = sequences[0].tolist()
        choices = [index for index in range(start, len(ids)) if ids[index] not in processor.tokenizer.all_special_ids]
        if not choices:
            raise ValueError("The model produced no caption tokens. Try a different image.")
        return {
            "model": VLM_MODELS[model_name],
            "model_id": model_name,
            "prompt": prompt,
            "input_image": model_input,
            "grid_shape": grid,
            "inputs": {key: value.cpu() for key, value in inputs.items()},
            "pixel_values": inputs["pixel_values"].cpu(),
            "generated_ids": ids,
            "answer": processor.tokenizer.decode(ids[start:], skip_special_tokens=True),
            "choices": choices,
            "token_labels": {
                index: f"{number} · {processor.tokenizer.decode([ids[index]])!r}"
                for number, index in enumerate(choices, 1)
            },
        }


def explain_answer_token(result, token_index, device):
    model, _, lock = load_resource("vlm", result["model_id"], device)
    with lock, torch.enable_grad():
        if token_index not in result["choices"]:
            raise ValueError("Select a generated caption token")
        inputs = answer_inputs(result, token_index, device)
        token_id = result["generated_ids"][token_index]
        if model.config.model_type == "blip":
            layer = model.text_decoder.bert.encoder.layer[-1].crossattention.self.dropout
            # Last text query attends to CLS followed by the row-major patch grid; CLS has no pixel location.
            reshape = lambda states: states[:, :, -1, 1:].reshape(
                states.shape[0], states.shape[1], *result["grid_shape"]
            )
        else:
            layer, reshape, _, _ = vlm_target(model, inputs)
        with LayerCAM(model, layer, reshape_transform=reshape) as extractor:
            output = model(**inputs, use_cache=False, logits_to_keep=1)
            maps = extractor(token_id, output.logits[:, -1])[0]
        probability = float(output.logits[0, -1].detach().float().softmax(-1)[token_id])
    return maps.float().cpu(), probability
