# Copyright (C) 2021-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from operator import itemgetter
from threading import Lock
from time import perf_counter

import streamlit as st
import torch
from PIL import ImageOps

from torchcam.methods import TAM, TokenIG

DECISION_MODELS = {"laya": "Laya", "minilm": "MiniLM · zero-shot classifier"}
VLM_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
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
            AutoModelForImageTextToText,
            AutoModelForSequenceClassification,
            AutoProcessor,
            AutoTokenizer,
        )

        if family == "vlm":
            processor = AutoProcessor.from_pretrained(VLM_ID)
            model = AutoModelForImageTextToText.from_pretrained(VLM_ID, dtype=torch.float32).eval().to(device)
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
    result = TokenIG(agent.model, agent.model.encoder.get_input_embeddings(), itemgetter(0))(
        inputs, token_mask=mask, max_steps=512
    )
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
    result = TokenIG(model, model.get_input_embeddings(), lambda output: output.logits)(
        inputs, 1, token_mask=mask, max_steps=512
    )
    probabilities = result.scores[:, 1].softmax(0).cpu().tolist()
    maps = result.attributions[:, 0].cpu().tolist()
    offsets = [
        [offset if keep else [0, 0] for offset, keep in zip(row, selected, strict=True)]
        for row, selected in zip(offsets, mask.cpu().tolist(), strict=True)
    ]
    return result, maps, offsets, probabilities, "entailment logit"


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
        "method": "TokenIG",
        "target": target,
        "question": QUESTION,
        "baseline": "zero ticket embeddings; question/options held fixed",
        "labels": list(QUESTION["criteria"]),
        "probabilities": probabilities,
        "attributions": maps,
        "offsets": offsets,
        "steps": result.steps,
        "scores": result.scores[:, result.class_indices].cpu().flatten().tolist(),
        "baseline_scores": result.baseline_scores[:, result.class_indices].cpu().flatten().tolist(),
        "completeness_error": result.completeness_error.cpu().flatten().tolist(),
        "elapsed": perf_counter() - started_at,
    }


def prepare_vlm_inputs(model, processor, image, prompt, device):
    size = model.config.vision_config.image_size
    model_input = ImageOps.fit(image, (size, size))
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=rendered,
        images=[model_input],
        do_image_splitting=False,
        size={"longest_edge": size},
        max_image_size={"longest_edge": size},
        return_tensors="pt",
    ).to(device)
    side = size // model.config.vision_config.patch_size // model.config.scale_factor
    if (
        inputs["pixel_values"].shape[1] != 1
        or int((inputs["input_ids"] == model.config.image_token_id).sum()) != side**2
    ):
        raise ValueError("The VLM input must contain one square image grid.")
    return inputs, model_input, side


def generate_answer(image, prompt, device):
    model, processor, lock = load_resource("vlm", VLM_ID, device)
    with lock, torch.inference_mode():
        inputs, model_input, side = prepare_vlm_inputs(model, processor, image, prompt, device)
        sequences = model.generate(**inputs, do_sample=False, max_new_tokens=48, return_dict_in_generate=False)
        ids = sequences[0, inputs["input_ids"].shape[1] :].tolist()
        choices = [index for index, token in enumerate(ids) if token not in processor.tokenizer.all_special_ids]
        if not choices:
            raise ValueError("The model produced no answer tokens. Try a different image or prompt.")
        return {
            "model": "SmolVLM-256M",
            "prompt": prompt,
            "input_image": model_input,
            "grid_side": side,
            "inputs": {key: value.cpu() for key, value in inputs.items()},
            "generated_ids": ids,
            "answer": processor.tokenizer.decode(ids, skip_special_tokens=True),
            "choices": choices,
            "token_labels": {index: f"{index + 1} · {processor.tokenizer.decode([ids[index]])!r}" for index in choices},
        }


def explain_answer_token(result, token_index, device):
    model, processor, lock = load_resource("vlm", VLM_ID, device)
    with lock, torch.inference_mode():
        inputs = {key: value.to(device) for key, value in result["inputs"].items()}
        prefix = torch.tensor([result["generated_ids"][:token_index]], device=device, dtype=torch.long)
        inputs["input_ids"] = torch.cat((inputs["input_ids"], prefix), -1)
        inputs["attention_mask"] = torch.cat((inputs["attention_mask"], torch.ones_like(prefix)), -1)
        output = model(**inputs, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        states = output.hidden_states[-1]
        ids = inputs["input_ids"][0]
        image_mask = ids == model.config.image_token_id
        special_ids = torch.tensor(processor.tokenizer.all_special_ids, device=device)
        context = inputs["attention_mask"][0].bool() & ~image_mask & ~torch.isin(ids, special_ids)
        side = result["grid_side"]
        token_id = result["generated_ids"][token_index]
        maps = TAM(model.get_output_embeddings())(
            token_id,
            states[:, image_mask].reshape(1, side, side, states.shape[-1]),
            states[:, context],
            inputs["input_ids"][:, context],
        )
        probability = float(output.logits[0, -1].float().softmax(-1)[token_id])
    return maps.float().cpu(), probability
