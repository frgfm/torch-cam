# Copyright (C) 2021-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import gc
import html
import json
import logging
import re
import struct
import sys
import tempfile
import zlib
from io import BytesIO
from pathlib import Path
from threading import Lock
from unittest.mock import patch
from urllib.request import Request, urlopen

import inference
import torch
from laya import Agent
from laya.common import DecisionModel, temp_bucket
from PIL import Image, PngImagePlugin, UnidentifiedImageError, features
from safetensors.torch import save_file
from streamlit.testing.v1 import AppTest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from torchvision.models import get_model
from transformers import (
    BlipConfig,
    BlipForConditionalGeneration,
    BlipImageProcessor,
    BlipProcessor,
    ModernBertConfig,
    ModernBertModel,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
)

import app

LOGGER = logging.getLogger(__name__)
TEXT_CONFIG = {
    "vocab_size": 256,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def hook_count(model):
    return sum(
        len(module._forward_hooks) + len(module._forward_pre_hooks) + len(module._backward_hooks)  # noqa: SLF001
        for module in model.modules()
    )


def image_upload(image, image_format, **kwargs):
    source = BytesIO()
    source.name = "upload.png"
    image.save(source, format=image_format, **kwargs)
    return source


def check_image_decoding():
    for image_format, mode in (("JPEG", "L"), ("PNG", "RGBA")):
        image = Image.new(mode, (8, 4))
        source = image_upload(image, image_format)
        decoded = app.read_image(source)
        source.close()
        check(decoded.mode == "RGB", f"{image_format} was not converted to RGB")
        check(decoded.size == image.size, f"{image_format} dimensions changed")
        decoded.load()

    image = Image.new("RGB", (8, 4))
    exif = Image.Exif()
    exif[274] = 6  # Rotate 90 degrees clockwise.
    oriented = app.read_image(image_upload(image, "JPEG", exif=exif))
    check(oriented.size == (4, 8), "EXIF orientation was not applied")

    # Filename filtering alone accepts these formats when renamed to .png.
    unsupported_formats = ["GIF", "BMP"]
    if features.check_codec("jpg_2000"):
        unsupported_formats.append("JPEG2000")
    for image_format in unsupported_formats:
        try:
            app.read_image(image_upload(image, image_format))
        except UnidentifiedImageError:
            pass
        else:
            raise AssertionError(f"Disguised {image_format} upload was accepted")

    try:
        app.read_image(BytesIO(b"\x89PNG\r\n\x1a\ntruncated"))
    except OSError:
        pass
    else:
        raise AssertionError("Truncated PNG upload was accepted")

    pixel_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = 24  # The 32-pixel image should trigger the warning, not the error.
        try:
            app.read_image(image_upload(image, "PNG"))
        except Image.DecompressionBombWarning:
            pass
        else:
            raise AssertionError("Oversized PNG upload was accepted")
    finally:
        Image.MAX_IMAGE_PIXELS = pixel_limit


def check_malformed_png_metadata():
    data = image_upload(Image.new("RGB", (1, 1)), "PNG").getvalue()
    end = data.index(b"IEND") - 4
    metadata_chunks = (
        (b"zTXt", b"key\0\1bad"),
        (b"gAMA", b""),
        (b"iCCP", b""),
        (b"eXIf", b"garbage"),
        (b"zTXt", b"key\0\0" + zlib.compress(b"a" * 33)),
    )
    with patch.object(PngImagePlugin, "MAX_TEXT_CHUNK", 32):
        for kind, payload in metadata_chunks:
            chunk = struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
            upload = data[:end] + chunk + data[end:]
            # Each file passes identification, then fails during loading or EXIF parsing.
            with Image.open(BytesIO(upload), formats=("JPEG", "PNG")) as image:
                check(image.format == "PNG", "Malformed fixture was not identified as PNG")
            try:
                app.read_image(BytesIO(upload))
            except OSError as exc:
                check(exc.__cause__ is not None, "Pillow parse error was not normalized")
            else:
                raise AssertionError(f"Malformed {kind!r} metadata was accepted")

            with patch("streamlit.file_uploader", return_value=BytesIO(upload)):
                ui = AppTest.from_file(Path(__file__).with_name("app.py")).run()
            check(len(ui.exception) == 0, "Malformed image crashed the demo")
            check(
                any("This image cannot be opened safely" in error.value for error in ui.error),
                "Malformed image did not show the safe-image error",
            )


def fixture_tokenizer():
    special = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    vocabulary = {token: index for index, token in enumerate(special)}
    corpus = (
        json.dumps(inference.QUESTION)
        + " Customer José charged twice refund duplicate payment user assistant describe image"
    )
    for token in re.findall(r"\w+|[^\w\s]", corpus):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    core = Tokenizer(WordLevel(vocabulary, unk_token=special[-1]))
    core.pre_tokenizer = Whitespace()
    core.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 1), ("[SEP]", 2)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        pad_token=special[0],
        cls_token=special[1],
        sep_token=special[2],
        mask_token=special[3],
        unk_token=special[4],
        bos_token=special[1],
        eos_token=special[2],
        model_max_length=256,
    )


def check_decision_adapters():
    # Native architectures with random weights verify adapters, not learned routing behavior.
    torch.manual_seed(19)
    tokenizer = fixture_tokenizer()
    text = "Customer José: charged twice refund duplicate payment."
    config = RobertaConfig(**TEXT_CONFIG, max_position_embeddings=256, num_labels=3)
    model = RobertaForSequenceClassification(config).eval().requires_grad_(False)
    result, maps, offsets, probabilities, _ = inference.explain_minilm(model, tokenizer, text)
    check(len(maps) == len(offsets) == len(probabilities) == 3, "MiniLM routing rows do not align")
    check(abs(sum(probabilities) - 1) < 1e-6, "MiniLM probabilities do not sum to one")
    check(bool(result.attributions.isfinite().all()), "MiniLM token effects are not finite")
    for row, scores in zip(offsets, maps, strict=True):
        check(
            all(score == 0 for offset, score in zip(row, scores, strict=True) if offset == [0, 0]),
            "MiniLM attributed fixed hypothesis or special tokens",
        )
    check(hook_count(model) == 0, "MiniLM adapter leaked hooks")
    with patch.object(inference, "load_resource", return_value=(model, tokenizer, Lock())):
        report = inference.explain_decision(text, "minilm", "cpu")
    check(len(report["scores"]) == 3 and report["method"] == "TokenAblation", "Routing score columns do not align")
    json.dumps(report, allow_nan=False)

    for layout in ("sequential", "parallel"):
        with tempfile.TemporaryDirectory(prefix="torchcam-laya-") as directory:
            path = Path(directory)
            config = ModernBertConfig(
                **TEXT_CONFIG,
                max_position_embeddings=512,
                local_attention=32,
                reference_compile=False,
                cls_token_id=1,
                sep_token_id=2,
            )
            native = DecisionModel(ModernBertModel(config), head_layers=2).eval()
            tokenizer.save_pretrained(path / "tokenizer")
            config.save_pretrained(path / "encoder")
            (path / "rl_agent_config.json").write_text(
                json.dumps({
                    "encoder": "offline-fixture",
                    "head_layers": 2,
                    "max_len": 256,
                    "head_max_len": 160,
                    "act_costs": {"escalate": 0.5},
                    "option_layout": layout,
                    "temperature": [1.6, 1.2, 1.8],
                    "temperature_by_options": {temp_bucket(0, 3): 1.3},
                }),
                encoding="utf-8",
            )
            save_file(native.state_dict(), str(path / "model.safetensors"))
            agent = Agent(str(path), device="cpu", backend="eager")
            result, maps, offsets, probabilities, _ = inference.explain_laya(agent, text)
            temperature = agent.temperature_by_options[temp_bucket(0, 3)]
            expected = (result.scores[0] / temperature).softmax(-1)
            torch.testing.assert_close(torch.tensor(probabilities), expected, atol=5.1e-5, rtol=0)
            check(
                all(len(row) == len(offset) for row, offset in zip(maps, offsets, strict=True)),
                "Laya state offsets do not align",
            )
            check(bool(result.attributions.isfinite().all()), "Laya token effects are not finite")
            check(hook_count(agent.model) == 0, "Laya adapter leaked hooks")
            try:
                inference.explain_laya(agent, "refund " * 400)
            except ValueError as error:
                check("Shorten" in str(error), "Unexpected truncation error")
            else:
                raise AssertionError("Laya silently explained a truncated ticket")


def check_vlm_adapter():
    torch.manual_seed(7)
    tokenizer = fixture_tokenizer()
    processor = BlipProcessor(BlipImageProcessor(size={"height": 32, "width": 32}), tokenizer)
    config = BlipConfig(
        vision_config={
            "image_size": 32,
            "patch_size": 8,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
        },
        text_config={**TEXT_CONFIG, "encoder_hidden_size": 16, "sep_token_id": 2, "is_decoder": True},
    )
    model = BlipForConditionalGeneration(config).eval().requires_grad_(False)
    # Keep synthetic decoding inside the known regular vocabulary, rather than producing only special IDs.
    model.generation_config.suppress_tokens = [index for index in range(256) if index not in range(5, 20)]
    with patch.object(inference, "load_resource", return_value=(model, processor, Lock())):
        result = inference.generate_answer(Image.new("RGB", (48, 24), "red"), "describe image", "cpu")
        index = result["choices"][2]
        cam, probability = inference.explain_answer_token(result, index, "cpu")
        prefix = torch.tensor([result["generated_ids"][:index]])
        with torch.no_grad():
            reference = model(pixel_values=result["pixel_values"], input_ids=prefix, use_cache=False)
        expected = float(reference.logits[0, -1].softmax(-1)[result["generated_ids"][index]])
        check(abs(probability - expected) < 1e-6, "VLM probability differs from native incremental generation")
        check(tuple(cam.shape) == (1, 4, 4), "VLM map does not match the native visual-token grid")
        check(bool(cam.isfinite().all()) and 0 < probability < 1, "VLM outputs are not finite")
        check(result["input_image"].size == (32, 32), "VLM overlay does not show the actual model crop")
        changed = {**result, "generated_ids": result["generated_ids"][: index + 1] + [6] * 5}
        future_cam, future_probability = inference.explain_answer_token(changed, index, "cpu")
        torch.testing.assert_close(cam, future_cam, atol=0, rtol=0)
        check(probability == future_probability, "VLM explanation leaked future answer tokens")
        check(hook_count(model) == 0, "VLM adapter leaked hooks")


def check_navigation_and_text_ui():
    with patch.object(
        inference, "load_resource", side_effect=AssertionError("Navigation must not load a model")
    ) as loader:
        ui = AppTest.from_file(Path(__file__).with_name("app.py")).run()
        ui.session_state["ticket-input"] = "Edited support ticket"
        ui.session_state["decision-model"] = "minilm"
        ui.session_state["image-model"] = "resnet50"
        ui.session_state["image-method-resnet50"] = "GradCAM"
        for section in ("Decision models", "VLMs", "Image classifiers", "Decision models", "VLMs", "Image classifiers"):
            ui.session_state["explorer"] = section
            ui.run()
            check(len(ui.exception) == 0, f"Navigation failed in {section}")
            if section == "Decision models":
                check(ui.text_area[0].value == "Edited support ticket", "Tab navigation reset the ticket")
                check(ui.selectbox[0].value == "minilm", "Tab navigation reset the decision model")
            elif section == "Image classifiers":
                check(ui.selectbox[0].value == "resnet50", "Tab navigation reset the image model")
                check(ui.selectbox[1].value == "GradCAM", "Tab navigation reset the CAM method")
        loader.assert_not_called()
    text = 'José 😀 <script>alert("ticket")</script>'
    rendered = app.token_heatmap(text, [(0, 4), (5, 6)], [1, -2])
    check(
        "<script>" not in rendered and "&lt;" in rendered and "😀" in rendered, "Ticket HTML or Unicode was corrupted"
    )
    check(html.unescape(re.sub(r"<[^>]*>", "", rendered)) == text, "Heatmap changed the original ticket")

    report = {
        "model": "Local verification fixture",
        "method": "TokenAblation",
        "text": text,
        "elapsed": 0.1,
        "labels": list(inference.QUESTION["criteria"]),
        "probabilities": [0.7, 0.2, 0.1],
        "offsets": [[(0, 4)]] * 3,
        "attributions": [[1], [-1], [0]],
    }
    with patch.object(app, "load_resource"), patch.object(app, "explain_decision", return_value=report):
        ui = AppTest.from_string("import app\napp.decision_page()").run()
        ui.text_area[0].set_value(text)
        ui.button[0].click().run()
        check(len(ui.exception) == 0 and len(ui.get("html")) == 3, "Decision maps did not render")
        ui.text_area[0].set_value("A different ticket").run()
        check(len(ui.get("html")) == 0, "Stale maps survived a changed ticket")
    with patch.object(app, "load_resource"), patch.object(app, "explain_decision", side_effect=RuntimeError("Failure")):
        ui.button[0].click().run()
        check(any("could not be explained" in error.value for error in ui.error), "Explanation failure was hidden")
        check(len(ui.get("html")) == 0, "Explanation failure displayed a stale map")


def check_pretrained():
    """Check the deployed models on public examples; run explicitly with --pretrained."""
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)
    for model_name in inference.DECISION_MODELS:
        for text, expected in (
            ("I was charged twice. Please refund the duplicate payment.", "Billing"),
            ("My password reset link has expired. Help me sign in.", "Account access"),
            ("My shipment has not arrived. The tracking page shows a delivery delay.", "Delivery"),
        ):
            result = inference.explain_decision(text, model_name, "cpu")
            predicted = result["labels"][max(range(3), key=result["probabilities"].__getitem__)]
            check(predicted == expected, f"{model_name} routed {expected} as {predicted}")
            check(bool(torch.tensor(result["attributions"]).isfinite().all()), "Nonfinite pretrained effects")
            LOGGER.info("Pretrained %s / %s: %.2fs", model_name, expected, result["elapsed"])
    url = "https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg"
    with urlopen(Request(url, headers={"User-Agent": "TorchCAM demo validation"}), timeout=60) as response:
        image = Image.open(BytesIO(response.read())).convert("RGB")
    result = inference.generate_answer(image, "a picture of", "cpu")
    model, processor, _ = inference.load_resource("vlm", inference.VLM_ID, "cpu")
    index = next(i for i in result["choices"] if processor.tokenizer.decode([result["generated_ids"][i]]) == "dog")
    cam, _ = inference.explain_answer_token(result, index, "cpu")
    inputs = {"pixel_values": result["pixel_values"], "input_ids": torch.tensor([result["generated_ids"][:index]])}
    occluded = []
    side, size = result["grid_side"], result["input_image"].width
    with torch.no_grad():
        for largest in (True, False):
            mask = torch.zeros(cam.numel(), dtype=torch.bool)
            mask[cam.flatten().topk(max(1, cam.numel() // 5), largest=largest).indices] = True
            mask = mask.reshape(side, side).repeat_interleave(size // side, 0).repeat_interleave(size // side, 1)
            altered = {**inputs, "pixel_values": inputs["pixel_values"].clone()}
            altered["pixel_values"][..., mask] = 0
            occluded.append(
                float(model(**altered, logits_to_keep=1, use_cache=False).logits[0, -1, result["generated_ids"][index]])
            )
    check(occluded[0] < occluded[1], "BLIP's high-ranked regions are less informative than low-ranked regions")
    LOGGER.info("Pretrained BLIP caption and token-occlusion check passed: %s", result["answer"])


def main():
    check_image_decoding()
    check_malformed_png_metadata()
    check_navigation_and_text_ui()
    check_decision_adapters()
    check_vlm_adapter()
    check(app.compatible_methods("vit_b_16") == ("LeGrad",), "ViT compatibility changed")
    check("LeGrad" not in app.compatible_methods("resnet18"), "LeGrad must stay ViT-only")
    check("FinerCAM" in app.compatible_methods("resnet18"), "FinerCAM is missing")
    check("RefineCAM" in app.compatible_methods("resnet18"), "RefineCAM is missing")

    for model_name in app.MODEL_LABELS:
        model = get_model(model_name, weights=None)
        module_names = dict(model.named_modules())
        for method_name in app.compatible_methods(model_name):
            preset = app.target_layer_preset(model_name, method_name)
            for layer in preset:
                check(layer in module_names, f"Unknown {model_name} preset: {layer}")
            if method_name == "CAM":
                with app.build_extractor(model, method_name, list(preset)):
                    pass
        check(hook_count(model) == 0, f"Hooks leaked while checking {model_name}")
        del model
        gc.collect()

    model = get_model("resnet18", weights=None).train()
    module_modes = [module.training for module in model.modules()]
    parameter_flags = [parameter.requires_grad for parameter in model.parameters()]
    input_tensor = torch.rand(3, 64, 64)

    cam, _, _, _ = app.extract_cam(model, Lock(), input_tensor, "GradCAM", ["layer4"])
    check(tuple(cam.shape) == (1, 2, 2), "Unexpected GradCAM shape")
    check(hook_count(model) == 0, "Hooks leaked after successful extraction")
    check([module.training for module in model.modules()] == module_modes, "Module modes were not restored")
    check(
        [parameter.requires_grad for parameter in model.parameters()] == parameter_flags,
        "Parameter flags were not restored",
    )
    check(all(parameter.grad is None for parameter in model.parameters()), "Parameter gradients were not cleared")

    try:
        app.extract_cam(model, Lock(), input_tensor, "GradCAM", ["layer4"], class_idx=1000)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid class index did not fail")
    check(hook_count(model) == 0, "Hooks leaked after failed extraction")
    check([module.training for module in model.modules()] == module_modes, "Failure changed module modes")
    check(
        [parameter.requires_grad for parameter in model.parameters()] == parameter_flags,
        "Failure changed parameter flags",
    )
    check(all(parameter.grad is None for parameter in model.parameters()), "Failure left parameter gradients")


if __name__ == "__main__":
    check_pretrained() if "--pretrained" in sys.argv else main()
