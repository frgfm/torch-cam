# Copyright (C) 2021-2026, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

import logging
import warnings
from contextlib import contextmanager
from io import BytesIO
from threading import Lock
from time import perf_counter

import streamlit as st
import torch
from matplotlib import colormaps
from PIL import Image, ImageOps
from torchvision.models import get_model, get_model_weights
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
        with Image.open(source) as image:
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_image(image, weights):
    transform = weights.transforms()
    input_tensor = transform(image)
    mean = input_tensor.new_tensor(transform.mean).view(-1, 1, 1)
    std = input_tensor.new_tensor(transform.std).view(-1, 1, 1)
    model_input = to_pil_image(input_tensor.mul(std).add(mean).clamp_(0, 1))
    return input_tensor, model_input


@st.cache_resource(show_spinner=False, max_entries=2)
def load_model(model_name, device):
    weights = get_model_weights(model_name).DEFAULT
    return get_model(model_name, weights=weights).eval().to(device), Lock()


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


def build_extractor(model, method_name, target_layers, finer_gamma=0.6, finer_references=3):
    if method_name == "CAM":
        # ponytail: supported torchvision models register the class-output Linear last; use explicit heads if that changes.
        fc_layer = next(
            name for name, module in reversed(tuple(model.named_modules())) if isinstance(module, torch.nn.Linear)
        )
        return methods.CAM(model, target_layer=target_layers, fc_layer=fc_layer)
    if method_name == "FinerCAM":
        return methods.FinerCAM(
            model,
            target_layer=target_layers,
            gamma=finer_gamma,
            num_references=finer_references,
        )
    if method_name == "RefineCAM":
        return methods.RefineCAM(model, target_layer=target_layers)
    if method_name == "LeGrad":
        return methods.LeGrad(model, target_layer=target_layers)
    return getattr(methods, method_name)(model, target_layer=target_layers)


def extract_cam(
    model,
    lock,
    input_tensor,
    method_name,
    target_layers,
    class_idx=None,
    finer_gamma=0.6,
    finer_references=3,
):
    unknown_layers = [layer for layer in target_layers if layer not in dict(model.named_modules())]
    if unknown_layers:
        raise ValueError(f"Unknown target layer(s): {', '.join(unknown_layers)}")

    device = next(model.parameters()).device
    # ponytail: shared cached models serialize hook mutation; use per-session models if throughput becomes a constraint.
    with lock, preserve_model_state(model):
        started_at = perf_counter()
        with build_extractor(
            model,
            method_name,
            target_layers,
            finer_gamma,
            finer_references,
        ) as extractor:
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
    finer_gamma,
    finer_references,
):
    target_layers = parse_target_layers(target_layer_value, model_name, method_name)
    input_tensor, model_input = preprocess_image(selected_image, weights)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with st.spinner(f"Loading {MODEL_LABELS[model_name]}…"):
            model, lock = load_model(model_name, device)
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
            finer_gamma,
            int(finer_references),
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

    columns[1].image(result["raw_image"], width="stretch")
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


def main():
    st.set_page_config(page_title="TorchCAM Explorer", page_icon="🔎", layout="wide")
    st.title("TorchCAM Explorer")
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
        )
        method_name = st.selectbox("CAM method", compatible_methods(model_name))
        weights = get_model_weights(model_name).DEFAULT
        categories = weights.meta["categories"]

        class_mode = st.radio("Target class", ("Predicted class", "Choose a class"))
        explicit_class_idx = None
        if class_mode == "Choose a class":
            explicit_class_idx = st.selectbox(
                "ImageNet class",
                range(len(categories)),
                format_func=categories.__getitem__,
            )

        finer_gamma = 0.6
        finer_references = 3
        preset = "+".join(target_layer_preset(model_name, method_name))
        with st.expander("Method settings"):
            target_layer_value = st.text_input(
                "Target layers",
                value=preset,
                key=f"target-layers-{model_name}-{method_name}",
                help="Separate multiple module names with '+'.",
            )
            if method_name == "FinerCAM":
                finer_gamma = st.number_input("Comparison strength", min_value=0.0, value=0.6, step=0.1)
                finer_references = st.number_input(
                    "Automatic comparison classes",
                    min_value=1,
                    max_value=len(categories) - 1,
                    value=3,
                    step=1,
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
                finer_gamma,
                finer_references,
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


if __name__ == "__main__":
    main()
