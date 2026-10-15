# Copyright (C) 2021-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import html
import json
import logging
import struct
import warnings
from contextlib import contextmanager
from io import BytesIO
from time import perf_counter

import streamlit as st
import torch
from inference import DECISION_MODELS, explain_answer_token, explain_decision, generate_answer, load_resource
from matplotlib import colormaps
from PIL import Image, ImageOps
from torchvision.models import get_model_weights
from torchvision.transforms.functional import to_pil_image

from torchcam import methods
from torchcam.utils import overlay_mask

LOGGER = logging.getLogger(__name__)
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "resnet50": "ResNet-50",
    "mobilenet_v3_small": "MobileNetV3 Small",
    "mobilenet_v3_large": "MobileNetV3 Large",
    "regnet_y_400mf": "RegNet-Y-400MF",
    "convnext_tiny": "ConvNeXt Tiny",
    "convnext_small": "ConvNeXt Small",
    "vit_b_16": "ViT-B/16",
}
MODEL_STAGES = {
    "resnet18": ("layer1", "layer2", "layer3", "layer4"),
    "resnet50": ("layer1", "layer2", "layer3", "layer4"),
    "mobilenet_v3_small": ("features.1", "features.3", "features.8", "features.12"),
    "mobilenet_v3_large": ("features.3", "features.6", "features.12", "features.16"),
    "regnet_y_400mf": (
        "trunk_output.block1",
        "trunk_output.block2",
        "trunk_output.block3",
        "trunk_output.block4",
    ),
    "convnext_tiny": ("features.1", "features.3", "features.5", "features.7"),
    "convnext_small": ("features.1", "features.3", "features.5", "features.7"),
    "vit_b_16": tuple(f"encoder.layers.encoder_layer_{idx}" for idx in range(8, 12)),
}
CNN_METHODS = (
    "CAM",
    "GradCAM",
    "GradCAMpp",
    "SmoothGradCAMpp",
    "ScoreCAM",
    "SSCAM",
    "ISCAM",
    "XGradCAM",
    "LayerCAM",
    "FinerCAM",
    "RefineCAM",
)
SLOW_METHODS = {"ScoreCAM", "SSCAM", "ISCAM"}


def compatible_methods(model_name):
    if model_name == "vit_b_16":
        return ("LeGrad",)
    return CNN_METHODS[1:] if model_name.startswith("mobilenet_v3") else CNN_METHODS


def target_layer_preset(model_name, method_name):
    stages = MODEL_STAGES[model_name]
    return stages if method_name in {"RefineCAM", "LeGrad"} else stages[-1:]


def parse_target_layers(value, model_name, method_name):
    if method_name not in compatible_methods(model_name):
        raise ValueError(f"{method_name} is not supported with {MODEL_LABELS[model_name]}")

    layers = [layer.strip() for layer in value.split("+")]
    if any(not layer for layer in layers):
        raise ValueError("Enter target layer names separated by '+'")
    if len(set(layers)) != len(layers):
        raise ValueError("Target layers must be unique")
    if method_name == "RefineCAM" and len(layers) < 2:
        raise ValueError("RefineCAM requires at least two target layers")
    if method_name == "LeGrad" and any(not layer.startswith("encoder.layers.encoder_layer_") for layer in layers):
        raise ValueError("LeGrad target layers must be ViT encoder blocks")
    return layers


def read_image(source):
    source.seek(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        try:
            with Image.open(source, formats=("JPEG", "PNG")) as image:
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
        except (ValueError, SyntaxError, IndexError, TypeError, struct.error) as exc:
            raise OSError("Invalid JPEG or PNG image data") from exc


def preprocess_image(image, weights):
    transform = weights.transforms()
    input_tensor = transform(image)
    mean = input_tensor.new_tensor(transform.mean).view(-1, 1, 1)
    std = input_tensor.new_tensor(transform.std).view(-1, 1, 1)
    model_input = to_pil_image(input_tensor.mul(std).add(mean).clamp_(0, 1))
    return input_tensor, model_input


@contextmanager
def preserve_model_state(model):
    module_modes = [(module, module.training) for module in model.modules()]
    parameter_flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    model.zero_grad(set_to_none=True)
    try:
        model.eval()
        with torch.enable_grad():
            yield
    finally:
        model.zero_grad(set_to_none=True)
        for module, training in module_modes:
            module.training = training
        for parameter, requires_grad in parameter_flags:
            parameter.requires_grad_(requires_grad)


def build_extractor(model, method_name, target_layers, **method_kwargs):
    if method_name == "CAM":
        # ponytail: supported torchvision models register the class-output Linear last; use explicit heads if that changes.
        method_kwargs["fc_layer"] = next(
            name for name, module in reversed(tuple(model.named_modules())) if isinstance(module, torch.nn.Linear)
        )
    return getattr(methods, method_name)(model, target_layer=target_layers, **method_kwargs)


def extract_cam(
    model,
    lock,
    input_tensor,
    method_name,
    target_layers,
    class_idx=None,
    **method_kwargs,
):
    unknown_layers = [layer for layer in target_layers if layer not in dict(model.named_modules())]
    if unknown_layers:
        raise ValueError(f"Unknown target layer(s): {', '.join(unknown_layers)}")

    device = next(model.parameters()).device
    # ponytail: shared cached models serialize hook mutation; use per-session models if throughput becomes a constraint.
    with lock, preserve_model_state(model):
        started_at = perf_counter()
        with build_extractor(model, method_name, target_layers, **method_kwargs) as extractor:
            scores = model(input_tensor.unsqueeze(0).to(device))
            target_idx = int(scores.argmax(dim=1).item()) if class_idx is None else class_idx
            if not 0 <= target_idx < scores.shape[1]:
                raise ValueError(f"Class index must be between 0 and {scores.shape[1] - 1}")
            if method_name == "RefineCAM":
                cams = extractor(target_idx, scores, target_shape=tuple(input_tensor.shape[-2:]))
            elif method_name == "LeGrad":
                cams = extractor(target_idx)
            else:
                cams = extractor(target_idx, scores)
            if len(cams) == 1:
                cam = cams[0]
            else:
                base_extractor = getattr(extractor, "base_cam", extractor)
                cam = base_extractor.fuse_cams(cams)
        elapsed = perf_counter() - started_at
        confidence = float(scores.softmax(dim=1)[0, target_idx].item())
        return cam.detach().float().cpu(), target_idx, confidence, elapsed


def colorize_cam(cam):
    array = cam.squeeze(0).numpy()
    return Image.fromarray((255 * colormaps["jet"](array)[..., :3]).astype("uint8"))


def image_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def tensor_bytes(tensor):
    buffer = BytesIO()
    torch.save(tensor, buffer)
    return buffer.getvalue()


def compute_result(
    selected_image,
    model_name,
    method_name,
    target_layer_value,
    weights,
    categories,
    explicit_class_idx,
    method_kwargs,
):
    target_layers = parse_target_layers(target_layer_value, model_name, method_name)
    input_tensor, model_input = preprocess_image(selected_image, weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with st.spinner(f"Loading {MODEL_LABELS[model_name]}…"):
            model, _, lock = load_resource("image", model_name, device)
    except Exception as exc:
        raise ConnectionError(
            "The pretrained model could not be loaded. Check connectivity, the model cache, and available memory, then retry."
        ) from exc

    with st.spinner("Computing the activation map…"):
        cam, class_idx, confidence, elapsed = extract_cam(
            model,
            lock,
            input_tensor,
            method_name,
            target_layers,
            explicit_class_idx,
            **method_kwargs,
        )
    raw_image = colorize_cam(cam)
    overlay_image = overlay_mask(
        model_input,
        to_pil_image(cam, mode="F"),
        alpha=0.5,
    )
    file_stem = method_name.lower()
    return {
        "model": MODEL_LABELS[model_name],
        "method": method_name,
        "class_name": categories[class_idx],
        "confidence": confidence,
        "layers": " + ".join(target_layers),
        "elapsed": elapsed,
        "input_image": model_input,
        "raw_image": raw_image,
        "overlay_image": overlay_image,
        "tensor_bytes": tensor_bytes(cam),
        "overlay_bytes": image_bytes(overlay_image),
        "tensor_filename": f"torchcam-{file_stem}.pt",
        "overlay_filename": f"torchcam-{file_stem}-overlay.png",
    }


def render_comparison(selected_image):
    result = st.session_state.get("result")
    st.subheader("Latest result" if result else "CAM comparison")
    if result:
        st.caption(
            f"{result['model']} · {result['method']} · {result['class_name']} "
            f"({result['confidence']:.1%}) · {result['layers']} · {result['elapsed']:.2f} s"
        )

    columns = st.columns(3, gap="medium")
    columns[0].markdown("#### Model input")
    if result:
        columns[0].image(result["input_image"], width="stretch")
    elif selected_image is not None:
        columns[0].image(selected_image, width="stretch")
    else:
        columns[0].info("Upload a PNG or JPEG image to begin.")

    columns[1].markdown("#### Raw CAM")
    columns[2].markdown("#### Overlay")
    if not result:
        columns[1].info("Compute a CAM to see the activation map.")
        columns[2].info("Compute a CAM to see the overlay.")
        return

    columns[1].container(key="raw-cam").image(result["raw_image"], width="stretch", output_format="PNG")
    columns[1].download_button(
        "Download raw tensor",
        data=result["tensor_bytes"],
        file_name=result["tensor_filename"],
        mime="application/octet-stream",
        width="stretch",
    )
    columns[2].image(result["overlay_image"], width="stretch")
    columns[2].download_button(
        "Download overlay",
        data=result["overlay_bytes"],
        file_name=result["overlay_filename"],
        mime="image/png",
        width="stretch",
    )


def image_page():
    st.caption("Compare class activation maps from pretrained torchvision models.")

    with st.sidebar:
        uploaded_file = st.file_uploader("Input image", type=("png", "jpg", "jpeg"))
        try:
            selected_image = read_image(uploaded_file) if uploaded_file is not None else None
        except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
            selected_image = None
            st.error("This image cannot be opened safely. Choose a valid PNG or JPEG file.")

        st.header("Configuration")
        model_name = st.selectbox(
            "Classification model",
            tuple(MODEL_LABELS),
            format_func=MODEL_LABELS.__getitem__,
            key="image-model",
        )
        method_name = st.selectbox("CAM method", compatible_methods(model_name), key=f"image-method-{model_name}")
        weights = get_model_weights(model_name).DEFAULT
        categories = weights.meta["categories"]

        class_mode = st.radio("Target class", ("Predicted class", "Choose a class"), key="image-class-mode")
        explicit_class_idx = None
        if class_mode == "Choose a class":
            explicit_class_idx = st.selectbox(
                "ImageNet class",
                range(len(categories)),
                format_func=categories.__getitem__,
                key="image-class-index",
            )

        method_kwargs = {}
        preset = "+".join(target_layer_preset(model_name, method_name))
        with st.expander("Method settings"):
            target_layer_value = st.text_input(
                "Target layers",
                value=preset,
                key=f"target-layers-{model_name}-{method_name}",
                help="Separate multiple module names with '+'.",
            )
            if method_name == "FinerCAM":
                method_kwargs["gamma"] = st.number_input(
                    "Comparison strength", min_value=0.0, value=0.6, step=0.1, key="image-gamma"
                )
                method_kwargs["num_references"] = st.number_input(
                    "Automatic comparison classes",
                    min_value=1,
                    max_value=len(categories) - 1,
                    value=3,
                    step=1,
                    key="image-reference-count",
                )
            if method_name in SLOW_METHODS:
                method_kwargs["batch_size"] = st.number_input(
                    "Masked-input batch size",
                    min_value=1,
                    value=32,
                    step=1,
                    key="image-batch-size",
                    help="Lower values use less memory but may change extraction latency.",
                )

        if method_name in SLOW_METHODS:
            st.warning(f"{method_name} performs extra forward passes and can be slow on CPU.")
        elif method_name == "LeGrad":
            st.info("LeGrad is limited to ViT-B/16 and its final four encoder blocks in this demo.")

        compute = st.button("Compute CAM", type="primary", width="stretch", disabled=selected_image is None)

    if compute:
        try:
            result = compute_result(
                selected_image,
                model_name,
                method_name,
                target_layer_value,
                weights,
                categories,
                explicit_class_idx,
                method_kwargs,
            )
        except ConnectionError as exc:
            LOGGER.exception("Unable to load pretrained model")
            st.error(str(exc))
        except (AssertionError, RuntimeError, TypeError, ValueError) as exc:
            st.error(f"The CAM could not be computed: {exc}")
        except Exception:
            LOGGER.exception("Unexpected CAM extraction failure")
            st.error("The CAM could not be computed. Check the selected layers and retry.")
        else:
            st.session_state["result"] = result

    render_comparison(selected_image)


def token_heatmap(text, offsets, contributions):
    # Use Python character offsets and escape the complete input, including unhighlighted gaps.
    values = [0.0] * len(text)
    for (start, end), contribution in zip(offsets, contributions, strict=True):
        if not 0 <= start <= end <= len(text):
            raise ValueError("Token offsets must lie within the original ticket")
        for index in range(start, end):
            values[index] += max(float(contribution), 0)
    maximum = max(values, default=0) or 1
    pieces = []
    for character, value in zip(text, values, strict=True):
        opacity = value / maximum * 0.5
        pieces.append(f'<span style="background-color:rgba(6,182,212,{opacity:.4f})">{html.escape(character)}</span>')
    return '<div style="white-space:pre-wrap;line-height:1.9">' + "".join(pieces) + "</div>"


def decision_page():
    st.caption("See which ticket tokens support each routing decision.")
    with st.sidebar:
        model_name = st.selectbox(
            "Decision model", tuple(DECISION_MODELS), format_func=DECISION_MODELS.__getitem__, key="decision-model"
        )
    text = st.text_area(
        "Support ticket",
        value="Customer: Alex Johnson. I was charged twice for the same order. Please refund the duplicate payment.",
        height=120,
        max_chars=2000,
        key="ticket-input",
    )
    st.caption("Routing options: Billing · Account access · Delivery")
    request = (model_name, text)
    if st.button("Explain decision", type="primary", disabled=not text.strip()):
        st.session_state.pop("decision_result", None)
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            with st.spinner(f"Loading {DECISION_MODELS[model_name]}…"):
                load_resource("decision", model_name, device)
            with st.spinner("Measuring each ticket token's effect…"):
                result = explain_decision(text, model_name, device)
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Decision explanation failed")
            st.error("The decision could not be explained. Check model availability and available memory, then retry.")
        else:
            st.session_state["decision_result"] = {**result, "request": request}
    result = st.session_state.get("decision_result")
    if not result or result["request"] != request:
        return
    st.caption(f"{result['model']} · Token ablation on class scores · {result['elapsed']:.2f} s")
    st.caption("Highlights show how much masking a token lowers a decision score, scaled within each decision.")
    for label, probability, offsets, scores in zip(
        result["labels"],
        result["probabilities"],
        result["offsets"],
        result["attributions"],
        strict=True,
    ):
        st.markdown(f"**{label} · {probability:.1%}**")
        st.html(token_heatmap(result["text"], offsets, scores))
    report = {key: value for key, value in result.items() if key != "request"}
    st.download_button(
        "Download signed contributions",
        json.dumps(report, ensure_ascii=False, indent=2),
        file_name="torchcam-ticket-attribution.json",
        mime="application/json",
    )


def vlm_page():
    st.caption("Generate a caption and explore image evidence for one generated token.")
    with st.sidebar:
        st.markdown("**BLIP captioner**")
        uploaded = st.file_uploader("Input image", type=("png", "jpg", "jpeg"), key="vlm-image")
    try:
        image = read_image(uploaded) if uploaded is not None else None
    except (OSError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        st.error("This image cannot be opened safely. Choose a valid PNG or JPEG file.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    request = uploaded.getvalue() if uploaded is not None else None
    if st.button("Generate caption", type="primary", disabled=image is None):
        st.session_state.pop("vlm_result", None)
        st.session_state.pop("vlm_map", None)
        st.session_state.pop("vlm-token", None)
        try:
            with st.spinner("Loading BLIP and generating a caption…"):
                result = generate_answer(image, "a picture of", device)
        except Exception:
            LOGGER.exception("VLM generation failed")
            st.error("The caption could not be generated. Check model availability and available memory, then retry.")
        else:
            st.session_state["vlm_result"] = {**result, "request": request}
    result = st.session_state.get("vlm_result")
    if not result or result["request"] != request:
        if image is not None:
            st.image(image, width=400)
        return
    st.markdown("**Caption**")
    st.write(result["answer"])
    token_index = st.selectbox(
        "Caption token",
        result["choices"],
        index=None,
        placeholder="Choose a caption token to explain",
        format_func=result["token_labels"].__getitem__,
        key="vlm-token",
    )
    if token_index is None:
        st.image(result["input_image"], width=400)
        return
    current_map = st.session_state.get("vlm_map")
    if current_map is None or current_map["index"] != token_index:
        try:
            with st.spinner("Computing the token activation map…"):
                cam, probability = explain_answer_token(result, token_index, device)
        except Exception:
            LOGGER.exception("VLM attribution failed")
            st.error("The visual map could not be computed for this token.")
            return
        current_map = {"index": token_index, "cam": cam, "probability": probability}
        st.session_state["vlm_map"] = current_map
    cam = current_map["cam"]
    st.caption(f"LayerCAM · raw next-token probability {current_map['probability']:.1%}")
    st.caption(
        "Colors show relative image influence for this token, not an object mask. The square crop is the model input."
    )
    if not cam.any():
        st.info("The model has no positive visual attribution for this token. Try another caption token.")
        return
    columns = st.columns(3)
    for column, label, display in zip(
        columns,
        ("Model input", "Token activation", "Overlay"),
        (
            result["input_image"],
            colorize_cam(cam),
            overlay_mask(result["input_image"], to_pil_image(cam, mode="F"), alpha=0.5),
        ),
        strict=True,
    ):
        column.markdown(f"#### {label}")
        column.image(display, width="stretch")
    st.download_button(
        "Download token map",
        tensor_bytes(cam),
        file_name="torchcam-vlm-token.pt",
        mime="application/octet-stream",
    )


def main():
    st.set_page_config(page_title="TorchCAM Explorer", page_icon="🔎", layout="wide")
    st.html("<style>.st-key-raw-cam img { image-rendering: pixelated; }</style>")
    st.title("TorchCAM Explorer")
    st.caption("Explore what supports a model's output: image regions, visual tokens, or text tokens.")
    # Keep editable controls alive while their tab is not rendered; uploaders cannot be assigned through session state.
    for key in tuple(st.session_state):
        if key.startswith(("image-", "target-layers-", "decision-", "ticket-")) or key == "vlm-token":
            st.session_state[key] = st.session_state[key]
    tabs = st.tabs(("Image classifiers", "VLMs", "Decision models"), key="explorer", on_change="rerun")
    for tab, page in zip(tabs, (image_page, vlm_page, decision_page), strict=True):
        if tab.open:
            with tab:
                page()


if __name__ == "__main__":
    main()
