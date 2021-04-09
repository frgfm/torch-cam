import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from torchvision import models
from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image

from torchcam import cams
from torchcam.utils import overlay_mask


CAM_METHODS = ["CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "SSCAM", "ISCAM", "XGradCAM"]
TV_MODELS = ["resnet18", "resnet50", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]


def main():

    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("TorchCAM: class activation explorer")
    # For newline
    st.write('\n')
    # Set the columns
    cols = st.beta_columns((1, 1, 1))
    cols[0].header("Input image")
    cols[1].header("Raw CAM")
    cols[-1].header("Overlayed CAM")

    # Sidebar
    # File selection
    st.sidebar.title("Input selection")
    # Disabling warning
    st.set_option('deprecation.showfileUploaderEncoding', False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=['png', 'jpeg', 'jpg'])
    if uploaded_file is not None:
        img = Image.open(BytesIO(uploaded_file.read()), mode='r').convert('RGB')

        cols[0].image(img, use_column_width=True)

    # Model selection
    st.sidebar.title("Model selection")
    tv_model = st.sidebar.selectbox("Classification model", TV_MODELS)
    target_layer = st.sidebar.text_input("Target layer", "")

    st.sidebar.title("Method selection")
    cam_method = st.sidebar.selectbox("CAM method", CAM_METHODS)

    if tv_model is not None:
        with st.spinner('Loading model...'):
            model = models.__dict__[tv_model](pretrained=True).eval()

        cam_extractor = cams.__dict__[cam_method](
            model,
            target_layer=target_layer if len(target_layer) > 0 else None
        )

    # For newline
    st.sidebar.write('\n')

    if st.sidebar.button("Compute CAM"):

        if uploaded_file is None:
            st.sidebar.write("Please upload an image")

        else:
            with st.spinner('Analyzing...'):

                # Preprocess image
                img_tensor = normalize(to_tensor(resize(img, (224, 224))), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

                # Forward the image to the model
                out = model(img_tensor.unsqueeze(0))
                # Retrieve the CAM
                activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
                # Plot the raw heatmap
                fig, ax = plt.subplots()
                ax.imshow(activation_map.numpy())
                ax.axis('off')
                cols[1].pyplot(fig)

                # Overlayed CAM
                fig, ax = plt.subplots()
                result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
                ax.imshow(result)
                ax.axis('off')
                cols[-1].pyplot(fig)


if __name__ == '__main__':
    main()
