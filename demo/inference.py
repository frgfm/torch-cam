# Copyright (C) 2021-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from operator import itemgetter
from threading import Lock
from time import perf_counter

import streamlit as st
import torch
from PIL import ImageOps

from torchcam.methods import LayerCAM, TokenAblation

DECISION_MODELS = {"laya": "Laya", "minilm": "MiniLM · zero-shot classifier"}
VLM_ID = "Salesforce/blip-image-captioning-base"
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
            AutoModelForSequenceClassification,
            AutoProcessor,
            AutoTokenizer,
            BlipForConditionalGeneration,
        )

        if family == "vlm":
            processor = AutoProcessor.from_pretrained(VLM_ID)
            model = BlipForConditionalGeneration.from_pretrained(VLM_ID).eval().to(device)
        elif family == "decision" and model_name == "minilm":
            checkpoint = "cross-encoder/nli-MiniLM2-L6-H768"
            processor = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint).eval().to(device)
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


def explain_minilm(model, tokenizer, text):
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
        raise ValueError("This ticket exceeds MiniLM's context. Shorten it so every displayed token can be explained.")
    offsets = encoded.pop("offset_mapping").tolist()
    inputs = {key: value.to(model.device) for key, value in encoded.items()}
    mask = torch.tensor(
        [[sequence == 0 for sequence in encoded.sequence_ids(row)] for row in range(len(hypotheses))],
        device=model.device,
    )
    # This checkpoint's published label order is contradiction, entailment, neutral.
    result = ablate_ticket(model, inputs, mask, tokenizer.mask_token_id, lambda output: output.logits[:, 1:2])
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


def explain_decision(text, model_name, device):
    model, processor, lock = load_resource("decision", model_name, device)
    with lock:
        started_at = perf_counter()
        result, maps, offsets, probabilities, target = (
            explain_laya(model, text) if model_name == "laya" else explain_minilm(model, processor, text)
        )
    return {
        "model": DECISION_MODELS[model_name],
        "text": text,
        "method": "TokenAblation",
        "target": target,
        "question": QUESTION,
        "baseline": "one ticket token masked at a time; question/options held fixed",
        "replacement_token": processor.mask_token,
        "labels": list(QUESTION["criteria"]),
        "probabilities": probabilities,
        "attributions": maps,
        "offsets": offsets,
        "scores": result.scores.cpu().flatten().tolist(),
        "elapsed": perf_counter() - started_at,
    }


def prepare_vlm_inputs(model, processor, image, prompt, device):
    size = model.config.vision_config.image_size
    model_input = ImageOps.fit(image, (size, size))
    inputs = processor(text=prompt, images=model_input, return_tensors="pt").to(device)
    side = size // model.config.vision_config.patch_size
    return inputs, model_input, side


def generate_answer(image, prompt, device):
    model, processor, lock = load_resource("vlm", VLM_ID, device)
    with lock, torch.inference_mode():
        inputs, model_input, side = prepare_vlm_inputs(model, processor, image, prompt, device)
        start = inputs["input_ids"].shape[1] - 1  # BLIP replaces CLS with BOS and removes the final SEP.
        sequences = model.generate(
            **inputs,
            do_sample=False,
            num_beams=3,
            use_cache=False,
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
            "model": "BLIP captioner",
            "prompt": prompt,
            "input_image": model_input,
            "grid_side": side,
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
    model, _, lock = load_resource("vlm", VLM_ID, device)
    with lock, torch.enable_grad():
        prefix = torch.tensor([result["generated_ids"][:token_index]], device=device, dtype=torch.long)
        inputs = {
            "pixel_values": result["pixel_values"].to(device).clone().requires_grad_(True),
            "input_ids": prefix,
            "attention_mask": torch.ones_like(prefix),
        }
        side = result["grid_side"]
        token_id = result["generated_ids"][token_index]
        layer = model.text_decoder.bert.encoder.layer[-1].crossattention.self.dropout
        # Last text query attends to CLS followed by the row-major patch grid; CLS has no pixel location.
        reshape = lambda states: states[:, :, -1, 1:].reshape(states.shape[0], states.shape[1], side, side)
        with LayerCAM(model, layer, reshape_transform=reshape) as extractor:
            output = model(**inputs, use_cache=False, logits_to_keep=1)
            maps = extractor(token_id, output.logits[:, -1])[0]
        probability = float(output.logits[0, -1].detach().float().softmax(-1)[token_id])
    return maps.float().cpu(), probability
