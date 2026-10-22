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
from threading import Event, Lock, Thread
from time import monotonic, sleep
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch
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
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSequenceClassification,
    BlipForConditionalGeneration,
    BlipImageProcessor,
    BlipProcessor,
    ModernBertConfig,
    ModernBertModel,
    PreTrainedTokenizerFast,
)
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

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


def check_upload_limits():
    upload = image_upload(Image.new("RGB", (2, 2)), "PNG")
    with (
        patch.object(app, "MAX_UPLOAD_BYTES", 1),
        patch.object(app.Image, "open", side_effect=AssertionError("oversized upload reached the decoder")),
    ):
        try:
            app.read_image(upload)
        except OSError as exc:
            check("8 MiB" in str(exc), "Oversized upload had an unclear error")
        else:
            raise AssertionError("Oversized upload was accepted")

    with (
        patch.object(app, "MAX_IMAGE_PIXELS", 3),
        patch.object(Image.Image, "load", side_effect=AssertionError("oversized image was decoded")),
    ):
        try:
            app.read_image(upload)
        except OSError as exc:
            check("8-megapixel" in str(exc), "Oversized image had an unclear error")
        else:
            raise AssertionError("Oversized image was accepted")


def check_algorithm_limits():
    try:
        app.parse_target_layers("layer1+layer2+layer3+layer4+layer5", "resnet18", "RefineCAM")
    except ValueError as exc:
        check("four" in str(exc), "Layer limit had an unclear error")
    else:
        raise AssertionError("Excessive target layers were accepted")

    try:
        app.compute_result(None, "resnet18", "ScoreCAM", "layer4", None, None, None, {"batch_size": 33})
    except ValueError as exc:
        check("between 1 and 32" in str(exc), "Invalid method settings had an unclear error")
    else:
        raise AssertionError("Excessive masked-input batch size was accepted")


def check_compute_admission():
    with patch.object(app, "COMPUTE_WAIT_SECONDS", 0.01), app.expensive_work():
        app.st.cache_resource.clear()
        try:
            with app.expensive_work():
                pass
        except app.ComputeBusyError:
            pass
        else:
            raise AssertionError("A concurrent compute request did not time out")
    with app.expensive_work():
        pass

    for error in (RuntimeError("fixture failure"), KeyboardInterrupt()):
        with TestCase().assertRaises(type(error)), app.expensive_work():  # noqa: PT027 - dependency-free smoke check.
            raise error
        with app.expensive_work():
            pass

    admissions = app.COMPUTE_ADMISSIONS
    waiting_started = Event()
    waiting_entered = Event()

    def wait_for_slot():
        waiting_started.set()
        with app.expensive_work():
            waiting_entered.set()

    with patch.object(app, "COMPUTE_WAIT_SECONDS", 2), app.expensive_work():
        waiter = Thread(target=wait_for_slot)
        waiter.start()
        check(waiting_started.wait(1), "Queued compute request did not start")
        deadline = monotonic() + 1
        while admissions._value and monotonic() < deadline:  # noqa: SLF001
            sleep(0.001)
        check(admissions._value == 0, "Admission did not reserve one active and one waiting slot")  # noqa: SLF001
        try:
            with app.expensive_work():
                pass
        except app.ComputeBusyError:
            pass
        else:
            raise AssertionError("An extra compute request exceeded the bounded queue")
    waiter.join(timeout=2)
    check(waiting_entered.is_set() and not waiter.is_alive(), "Queued compute request did not run after release")


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


def check_decision_report(report, model):
    check(
        len(report["probabilities"]) == len(report["attributions"]) == len(report["offsets"]) == 3,
        "Routing rows do not align",
    )
    # Native Laya probabilities are rounded to four decimals.
    check(abs(sum(report["probabilities"]) - 1) < 1.5e-4, "Routing probabilities do not sum to one")
    for offsets, values in zip(report["offsets"], report["attributions"], strict=True):
        check(all(0 <= a <= b <= len(report["text"]) for a, b in offsets), "Offsets exceed the original ticket")
        check(
            all(v == 0 for offset, v in zip(offsets, values, strict=True) if offset == [0, 0]),
            "Attributed fixed context",
        )
    check(hook_count(model) == 0, "Decision adapter leaked hooks")
    json.dumps(report, allow_nan=False)


def check_decision_adapters():
    # Native architectures with random weights verify adapters, not learned routing behavior.
    torch.manual_seed(19)
    tokenizer = fixture_tokenizer()
    text = "Customer José: charged twice refund duplicate payment."
    for name, architecture, labels in (
        ("minilm", "roberta", ("contradiction", "entailment", "neutral")),
        ("deberta", "deberta-v2", ("entailment", "not_entailment")),
    ):
        config = AutoConfig.for_model(
            architecture, **TEXT_CONFIG, max_position_embeddings=256, id2label=dict(enumerate(labels))
        )
        model = AutoModelForSequenceClassification.from_config(config).eval().requires_grad_(False)
        with patch.object(inference, "load_resource", return_value=(model, tokenizer, Lock())):
            report = inference.explain_decision(text, name, "cpu")
        check_decision_report(report, model)
        hypotheses = [
            f"This ticket is about {description.lower()}" for description in inference.QUESTION["criteria"].values()
        ]
        inputs = tokenizer([text] * 3, hypotheses, padding=True, return_tensors="pt")
        with torch.no_grad():
            expected = model(**inputs).logits[:, labels.index("entailment")]
        torch.testing.assert_close(torch.tensor(report["scores"]), expected)
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
            with patch.object(inference, "load_resource", return_value=(agent, tokenizer, Lock())):
                report = inference.explain_decision(text, "laya", "cpu")
            check_decision_report(report, agent.model)
            temperature = agent.temperature_by_options[temp_bucket(0, 3)]
            expected = (torch.tensor(report["scores"]) / temperature).softmax(-1)
            torch.testing.assert_close(torch.tensor(report["probabilities"]), expected, atol=5.1e-5, rtol=0)
            with TestCase().assertRaisesRegex(ValueError, "Shorten"):  # noqa: PT027 - demo extras do not include pytest.
                inference.explain_laya(agent, "refund " * 400)


def check_decider_adapter():
    tokenizer = fixture_tokenizer()
    tokenizer.add_tokens(["A", "B", "C"])
    config = AutoConfig.for_model(
        "qwen3_5_text",
        **TEXT_CONFIG,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        layer_types=["linear_attention", "full_attention"],
    )
    model = AutoModelForCausalLM.from_config(config).eval().requires_grad_(False)
    model.config.decision_temperature = 1.03
    text = "Customer José: charged twice refund duplicate payment."
    encode = tokenizer.encode
    # The synthetic whitespace tokenizer needs a stand-in ID; the pretrained gate checks the real space token.
    with (
        patch.object(tokenizer, "encode", side_effect=lambda value, **kw: [3] if value == " " else encode(value, **kw)),
        patch.object(inference, "load_resource", return_value=(model, tokenizer, Lock())),
    ):
        report = inference.explain_decision(text, "decider", "cpu")
    expected = (torch.tensor(report["scores"]) / model.config.decision_temperature).softmax(-1)
    torch.testing.assert_close(torch.tensor(report["probabilities"]), expected)
    check_decision_report(report, model)
    check(report["replacement_token"].isspace(), "Invalid Decider replacement baseline")


def check_vlm_result(model, processor, result, index):
    with patch.object(inference, "load_resource", return_value=(model, processor, Lock())):
        cam, probability = inference.explain_answer_token(result, index, "cpu")
        check(tuple(cam.shape) == (1, *result["grid_shape"]) and cam.isfinite().all(), "Invalid native VLM map")
        inputs = {key: value.clone() for key, value in result["inputs"].items()}
        inputs["input_ids"] = torch.tensor([result["generated_ids"][:index]])
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        for key in ("mm_token_type_ids", "token_type_ids"):
            if key in inputs:
                inputs[key] = torch.nn.functional.pad(inputs[key], (0, index - inputs[key].shape[1]))
        with torch.no_grad():
            output = model(**inputs, use_cache=False, logits_to_keep=1)
            expected = output.logits[0, -1].float().softmax(-1)[result["generated_ids"][index]]
        check(abs(float(expected) - probability) < 1e-6, "VLM score differs from native prefix replay")
        changed = {**result, "generated_ids": [*result["generated_ids"][: index + 1], 99]}
        future, future_probability = inference.explain_answer_token(changed, index, "cpu")
        torch.testing.assert_close(cam, future, atol=0, rtol=0)
        check(probability == future_probability and hook_count(model) == 0, "Future-token leakage or leftover hooks")


def check_vlm_adapter():
    torch.manual_seed(7)
    tokenizer = fixture_tokenizer()
    processor = BlipProcessor(BlipImageProcessor(size={"height": 32, "width": 32}), tokenizer)
    config = AutoConfig.for_model(
        "blip",
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
        check(result["input_image"].size == (32, 32) and result["grid_shape"] == (4, 4), "Incorrect BLIP crop/grid")
        check_vlm_result(model, processor, result, result["choices"][2])


def check_generation_limits():
    image = Image.new("RGB", (2, 2))
    inputs = {"input_ids": torch.ones((1, 2), dtype=torch.long)}
    processor = SimpleNamespace(tokenizer=SimpleNamespace(all_special_ids=[], decode=lambda *_args, **_kwargs: "word"))
    for model_id, model_type, limit in (
        (inference.VLM_ID, "blip", 24),
        ("Qwen/Qwen3-VL-2B-Instruct", "qwen3_vl", 48),
        ("google/gemma-4-E2B-it", "gemma4", 48),
    ):
        model = SimpleNamespace(
            config=SimpleNamespace(model_type=model_type),
            generate=MagicMock(return_value=torch.tensor([[1, 5, 6]])),
        )
        with (
            patch.object(inference, "load_resource", return_value=(model, processor, Lock())),
            patch.object(inference, "prepare_vlm_inputs", return_value=(inputs, image, (1, 1))),
        ):
            inference.generate_answer(image, "a picture of", "cpu", model_id)
        check(model.generate.call_args.kwargs["max_new_tokens"] == limit, f"{model_id} generation was unbounded")


def check_vlm_families():
    inference.load_resource.clear()
    with (
        patch.object(inference, "vlm_device_supported", return_value=False),
        TestCase().assertRaisesRegex(ValueError, "BF16"),  # noqa: PT027 - keep this smoke check dependency-free.
    ):
        inference.load_resource("vlm", "google/gemma-4-E2B-it", "cpu")
    torch.manual_seed(13)
    image = Image.fromarray(torch.randint(0, 256, (48, 64, 3), dtype=torch.uint8).numpy())
    text = {**TEXT_CONFIG, "num_key_value_heads": 2, "head_dim": 8}
    for family in ["qwen", "gemma"]:
        if family == "qwen":
            proc = Qwen2VLImageProcessor(patch_size=8, merge_size=2, temporal_patch_size=2, do_resize=False)
            pixels = proc(images=[image], return_tensors="pt")
            config = AutoConfig.for_model(
                "qwen3_vl",
                image_token_id=20,
                vision_start_token_id=21,
                vision_end_token_id=22,
                text_config={**text, "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
                vision_config={
                    "depth": 2,
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_heads": 2,
                    "out_hidden_size": 32,
                    "patch_size": 8,
                    "temporal_patch_size": 2,
                    "spatial_merge_size": 2,
                    "num_position_embeddings": 16,
                    "deepstack_visual_indexes": [0],
                },
            )
        else:
            proc = Gemma4ImageProcessor(patch_size=8, pooling_kernel_size=2, max_soft_tokens=70, do_resize=False)
            pixels = proc(images=[image], return_tensors="pt")
            pixels.pop("num_soft_tokens_per_image")
            config = AutoConfig.for_model(
                "gemma4",
                image_token_id=20,
                boi_token_id=21,
                eoi_token_id=22,
                text_config={
                    **text,
                    "vocab_size_per_layer_input": 256,
                    "hidden_size_per_layer_input": 8,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "use_bidirectional_attention": "vision",
                },
                vision_config={
                    "hidden_size": 16,
                    "intermediate_size": 32,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "patch_size": 8,
                    "pooling_kernel_size": 2,
                    "position_embedding_size": 64,
                },
            )
        model = AutoModelForImageTextToText.from_config(config).eval().requires_grad_(False)
        ids = [1, 21] + [20] * 12 + [22, 10, 11]
        pixels["input_ids"] = torch.tensor([ids])
        pixels["attention_mask"] = torch.ones_like(pixels["input_ids"])
        pixels["mm_token_type_ids"] = (pixels["input_ids"] == 20).long()
        processor = SimpleNamespace(image_processor=proc, apply_chat_template=lambda *_, batch=pixels, **__: batch)
        inputs, display, grid = inference.prepare_vlm_inputs(model, processor, image, "unused", "cpu")
        check(
            grid == ((3, 4) if family == "qwen" else (6, 8)) and display.size == image.size,
            "Non-square patch geometry changed",
        )
        check(
            max(abs(a - b) for a, b in zip(display.tobytes(), image.tobytes(), strict=True)) <= 1,
            "Patch projection transposes or scrambles pixels",
        )
        result = {
            "inputs": inputs,
            "generated_ids": [*ids, 12, 13, 14],
            "choices": [len(ids), len(ids) + 1, len(ids) + 2],
            "grid_shape": grid,
            "model_id": family,
        }
        check_vlm_result(model, processor, result, len(ids) + 1)


def check_navigation_and_text_ui():
    with patch.object(
        inference, "load_resource", side_effect=AssertionError("Navigation must not load a model")
    ) as loader:
        ui = AppTest.from_file(Path(__file__).with_name("app.py")).run()
        ui.session_state["ticket-input"] = "Edited support ticket"
        ui.session_state["decision-model"] = "minilm"
        ui.session_state["vlm-model"] = inference.VLM_ID
        ui.session_state["image-model"] = "resnet50"
        ui.session_state["image-method-resnet50"] = "GradCAM"
        for section in ("Decision models", "VLMs", "Image classifiers", "Decision models", "VLMs", "Image classifiers"):
            ui.session_state["explorer"] = section
            ui.run()
            check(len(ui.exception) == 0, f"Navigation failed in {section}")
            check(len(ui.tabs) == 3, "Task tabs disappeared")
            if section == "Decision models":
                check(ui.text_area[0].value == "Edited support ticket", "Tab navigation reset the ticket")
                check(ui.selectbox[0].value == "minilm", "Tab navigation reset the decision model")
            elif section == "VLMs":
                check(ui.selectbox[0].label == "VLM model", "VLM model selector disappeared")
                check(ui.selectbox[0].value == inference.VLM_ID, "Tab navigation reset the VLM model")
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
        "replacement_token": fixture_tokenizer().mask_token,
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
    for model_id in inference.VLM_MODELS:
        if not inference.vlm_device_supported(model_id, "cpu"):
            LOGGER.info("Skipping %s: this CPU lacks BF16 acceleration", model_id)
            continue
        inference.load_resource.clear()
        gc.collect()
        result = inference.generate_answer(image, "a picture of", "cpu", model_id)
        model, processor, _ = inference.load_resource("vlm", model_id, "cpu")
        index = next(
            i
            for i in result["choices"]
            if processor.tokenizer.decode([result["generated_ids"][i]]).strip().lower() in {"dog", "border"}
        )
        cam, probability = inference.explain_answer_token(result, index, "cpu")
        check(bool(cam.isfinite().all()) and bool(cam.any()), "Invalid pretrained token map")
        inputs = inference.answer_inputs(result, index, "cpu")
        inputs["pixel_values"] = inputs["pixel_values"].detach()
        baseline = float(torch.tensor(probability).log())
        drops = {True: [], False: []}
        with torch.no_grad():
            for fraction in (0.2, 0.5):
                for largest in (True, False):
                    selected = cam.flatten().topk(max(1, int(cam.numel() * fraction)), largest=largest).indices
                    altered = {**inputs, "pixel_values": inputs["pixel_values"].clone()}
                    if model.config.model_type == "blip":
                        height, width = result["grid_shape"]
                        mask = torch.zeros(height * width, dtype=torch.bool)
                        mask[selected] = True
                        mask = mask.reshape(height, width).repeat_interleave(result["input_image"].height // height, 0)
                        mask = mask.repeat_interleave(result["input_image"].width // width, 1)
                        altered["pixel_values"][..., mask] = 0
                    else:
                        _, _, groups, _ = inference.vlm_target(model, inputs)
                        pixels = altered["pixel_values"].reshape(-1, altered["pixel_values"].shape[-1])
                        pixels[groups[selected].flatten()] = 0 if model.config.model_type == "qwen3_vl" else 0.5
                    scores = model(**altered, logits_to_keep=1, use_cache=False).logits[0, -1].float()
                    drops[largest].append(baseline - float(scores.log_softmax(-1)[result["generated_ids"][index]]))
        check(sum(drops[True]) > max(0, sum(drops[False])), f"{model_id}: region-removal rank check failed")
        LOGGER.info("Pretrained %s caption and token-occlusion check passed: %s", model_id, result["answer"])
        del model, processor, result, cam, inputs, altered


def main():
    check_image_decoding()
    check_upload_limits()
    check_algorithm_limits()
    check_compute_admission()
    check_malformed_png_metadata()
    check_navigation_and_text_ui()
    check_decision_adapters()
    check_decider_adapter()
    check_vlm_adapter()
    check_generation_limits()
    check_vlm_families()
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
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    gradient_reference = parameter.grad
    saved_gradient = parameter.grad.clone()
    input_tensor = torch.rand(3, 64, 64)

    cam, _, _, _ = app.extract_cam(model, Lock(), input_tensor, "GradCAM", ["layer4"])
    check(tuple(cam.shape) == (1, 2, 2), "Unexpected GradCAM shape")
    check(hook_count(model) == 0, "Hooks leaked after successful extraction")
    check([module.training for module in model.modules()] == module_modes, "Module modes were not restored")
    check(
        [parameter.requires_grad for parameter in model.parameters()] == parameter_flags,
        "Parameter flags were not restored",
    )
    check(parameter.grad is gradient_reference, "Existing gradient reference was not restored")
    torch.testing.assert_close(parameter.grad, saved_gradient)

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
    check(parameter.grad is gradient_reference, "Failure replaced an existing gradient")
    torch.testing.assert_close(parameter.grad, saved_gradient)


if __name__ == "__main__":
    check_pretrained() if "--pretrained" in sys.argv else main()
