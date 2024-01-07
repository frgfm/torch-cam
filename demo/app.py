# Copyright (C) 2021-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from io import BytesIO

import matplotlib.pyplot as plt
import requests
import streamlit as st
import torch
from PIL import Image
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor

from torchcam import methods
from torchcam.methods._utils import locate_candidate_layer
from torchcam.utils import overlay_mask

CAM_METHODS = [
    "CAM",
    "GradCAM",
    "GradCAMpp",
    "SmoothGradCAMpp",
    "ScoreCAM",
    "SSCAM",
    "ISCAM",
    "XGradCAM",
    "LayerCAM",
]
TV_MODELS = [
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "regnet_y_400mf",
    "convnext_tiny",
    "convnext_small",
]
LABEL_MAP = requests.get(
    "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json",
    timeout=10,
).json()


def main():
    # Wide mode
    st.set_page_config(page_title="TorchCAM - Class activation explorer", layout="wide")

    # Designing the interface
    st.title("TorchCAM: class activation explorer")
    # For newline
    st.write("\n")
    # Set the columns
    cols = st.columns((1, 1, 1))
    cols[0].header("Input image")
    cols[1].header("Raw CAM")
    cols[-1].header("Overlayed CAM")

    # Sidebar
    # File selection
    st.sidebar.title("Input selection")
    # Disabling warning
    st.set_option("deprecation.showfileUploaderEncoding", False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["png", "jpeg", "jpg"])
    if uploaded_file is not None:
        img = Image.open(BytesIO(uploaded_file.read()), mode="r").convert("RGB")

        cols[0].image(img, use_column_width=True)

    # Model selection
    st.sidebar.title("Setup")
    tv_model = st.sidebar.selectbox(
        "Classification model",
        TV_MODELS,
        help="Supported models from Torchvision",
    )
    default_layer = ""
    if tv_model is not None:
        with st.spinner("Loading model..."):
            model = models.__dict__[tv_model](pretrained=True).eval()
        default_layer = locate_candidate_layer(model, (3, 224, 224))

    if torch.cuda.is_available():
        model = model.cuda()

    target_layer = st.sidebar.text_input(
        "Target layer",
        default_layer,
        help='If you want to target several layers, add a "+" separator (e.g. "layer3+layer4")',
    )
    cam_method = st.sidebar.selectbox(
        "CAM method",
        CAM_METHODS,
        help="The way your class activation map will be computed",
    )
    if cam_method is not None:
        cam_extractor = methods.__dict__[cam_method](
            model,
            target_layer=[s.strip() for s in target_layer.split("+")] if len(target_layer) > 0 else None,
        )

    class_choices = [f"{idx + 1} - {class_name}" for idx, class_name in enumerate(LABEL_MAP)]
    class_selection = st.sidebar.selectbox("Class selection", ["Predicted class (argmax)", *class_choices])

    # For newline
    st.sidebar.write("\n")

    if st.sidebar.button("Compute CAM"):
        if uploaded_file is None:
            st.sidebar.error("Please upload an image first")

        else:
            with st.spinner("Analyzing..."):
                # Preprocess image
                img_tensor = normalize(
                    to_tensor(resize(img, (224, 224))),
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                )

                if torch.cuda.is_available():
                    img_tensor = img_tensor.cuda()

                # Forward the image to the model
                out = model(img_tensor.unsqueeze(0))
                # Select the target class
                if class_selection == "Predicted class (argmax)":
                    class_idx = out.squeeze(0).argmax().item()
                else:
                    class_idx = LABEL_MAP.index(class_selection.rpartition(" - ")[-1])
                # Retrieve the CAM
                act_maps = cam_extractor(class_idx, out)
                # Fuse the CAMs if there are several
                activation_map = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(activation_map.squeeze(0).cpu().numpy())
                ax.axis("off")
                cols[1].pyplot(fig)

                # Overlayed CAM
                fig, ax = plt.subplots()
                result = overlay_mask(img, to_pil_image(activation_map, mode="F"), alpha=0.5)
                ax.imshow(result)
                ax.axis("off")
                cols[-1].pyplot(fig)


if __name__ == "__main__":
    main()
