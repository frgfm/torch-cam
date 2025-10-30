# TorchCAM: class activation explorer

TorchCAM provides a minimal yet flexible way to explore the spatial importance of features on your PyTorch model outputs. Check out the live demo on [HuggingFace Spaces](https://huggingface.co/spaces/frgfm/torch-cam) 🤗

<p align="center">
    <img src="https://github.com/frgfm/torch-cam/releases/download/v0.3.1/example.png" alt="CAM visualization" width="70%">
</p>
<p align="center">
    <em>Source: image from <a href="https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg">woopets</a> (activation maps created with a pretrained <a href="https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18">Resnet-18</a>)</em>
</p>

This project is meant for:

* ⚡ **exploration**: easily assess the influence of spatial features on your model's outputs
* 👩‍🔬 **research**: quickly implement your own ideas for new CAM methods

## Installation

Create and activate a virtual environment and then install TorchCAM:

```shell
pip install torchcam
```

Check out the [installation guide](getting-started/installation.md) for more options

## Quick start

Get an image and a model:

```python
from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights

weights = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
preprocess = weights.transforms()

img_path = "path/to/your/image.jpg"

img = decode_image(img_path)
input_tensor = preprocess(img)
```

Compute the class activation map:

```python hl_lines="3 6"
from torchcam.methods import LayerCAM

with LayerCAM(model) as cam_extractor:
  out = model(input_tensor.unsqueeze(0))
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
```

Display it:

```python hl_lines="3 6"
import matplotlib.pyplot as plt
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
```

![overlayed_heatmap](https://github.com/frgfm/torch-cam/releases/download/v0.1.2/overlayed_heatmap.png)

## CAM zoo

### Activation-based methods
   * CAM from ["Learning Deep Features for Discriminative Localization"](https://arxiv.org/pdf/1512.04150.pdf)
   * Score-CAM from ["Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks"](https://arxiv.org/pdf/1910.01279.pdf)
   * SS-CAM from ["SS-CAM: Smoothed Score-CAM for Sharper Visual Feature Localization"](https://arxiv.org/pdf/2006.14255.pdf)
   * IS-CAM from ["IS-CAM: Integrated Score-CAM for axiomatic-based explanations"](https://arxiv.org/pdf/2010.03023.pdf)

### Gradient-based methods
   * Grad-CAM from ["Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"](https://arxiv.org/pdf/1610.02391.pdf)
   * Grad-CAM++ from ["Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"](https://arxiv.org/pdf/1710.11063.pdf)
   * Smooth Grad-CAM++ from ["Smooth Grad-CAM++: An Enhanced Inference Level Visualization Technique for Deep Convolutional Neural Network Models"](https://arxiv.org/pdf/1908.01224.pdf)
   * X-Grad-CAM from ["Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs"](https://arxiv.org/pdf/2008.02312.pdf)
   * Layer-CAM from ["LayerCAM: Exploring Hierarchical Class Activation Maps for Localization"](http://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf)
