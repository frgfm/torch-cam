<h1 align="center">
  TorchCAM: class activation explorer
</h1>

<p align="center">
  <a href="https://github.com/frgfm/torch-cam/actions/workflows/build.yml">
    <img alt="CI Status" src="https://img.shields.io/github/actions/workflow/status/frgfm/torch-cam/build.yml?branch=main&label=CI&logo=github&style=flat-square">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/Linter-Ruff-FCC21B?style=flat-square&logo=ruff&logoColor=white" alt="ruff">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/Formatter-Ruff-FCC21B?style=flat-square&logo=Python&logoColor=white" alt="ruff">
  </a>
  <a href="https://www.codacy.com/gh/frgfm/torch-cam/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/torch-cam&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/87eaeec3e15442188f96c36bace5faf4"/></a>
  <a href="https://codecov.io/gh/frgfm/torch-cam">
    <img src="https://img.shields.io/codecov/c/github/frgfm/torch-cam.svg?logo=codecov&style=flat-square&label=Coverage" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/torchcam/">
    <img src="https://img.shields.io/pypi/v/torchcam.svg?logo=PyPI&logoColor=fff&style=flat-square&label=PyPI" alt="PyPi Version">
  </a>
  <a href="https://anaconda.org/frgfm/torchcam">
    <img src="https://img.shields.io/conda/v/frgfm/torchcam.svg?logo=anaconda&label=Conda&logoColor=fff&style=flat-square" alt="Conda Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/torchcam.svg?logo=Python&label=Python&logoColor=fff&style=flat-square" alt="pyversions">
  <a href="https://github.com/frgfm/torch-cam/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/frgfm/torch-cam.svg?label=License&logoColor=fff&style=flat-square" alt="License">
  </a>
</p>
<p align="center">
  <a href="https://huggingface.co/spaces/frgfm/torch-cam">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Huggingface Spaces">
  </a>
  <a href="https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/quicktour.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
  </a>
</p>
<p align="center">
  <a href="https://frgfm.github.io/torch-cam">
    <img src="https://img.shields.io/github/actions/workflow/status/frgfm/torch-cam/page-build.yml?branch=main&label=Documentation&logo=read-the-docs&logoColor=white&style=flat-square" alt="Documentation Status">
  </a>
</p>

Simple way to leverage the class-specific activation of convolutional layers in PyTorch.

<p align="center">
    <a alt="cam_examples">
        <img src="https://github.com/frgfm/torch-cam/releases/download/v0.3.1/example.png" /></a>
</p>
<p align="center">
    <em>Source: image from <a href="https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg">woopets</a> (activation maps created with a pretrained <a href="https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18">Resnet-18</a>)</em>
</p>


## Quick Tour

### Setting your CAM

TorchCAM leverages [PyTorch hooking mechanisms](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks) to seamlessly retrieve all required information to produce the class activation without additional efforts from the user. Each CAM object acts as a wrapper around your model.

You can find the exhaustive list of supported CAM methods in the [documentation](https://frgfm.github.io/torch-cam/methods.html), then use it as follows:

```python
# Define your model
from torchvision.models import resnet18
model = resnet18(pretrained=True).eval()

# Set your CAM extractor
from torchcam.methods import SmoothGradCAMpp
cam_extractor = SmoothGradCAMpp(model)
```

*Please note that by default, the layer at which the CAM is retrieved is set to the last non-reduced convolutional layer. If you wish to investigate a specific layer, use the `target_layer` argument in the constructor.*



### Retrieving the class activation map

Once your CAM extractor is set, you only need to use your model to infer on your data as usual. If any additional information is required, the extractor will get it for you automatically.

```python
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp

model = resnet18(pretrained=True).eval()
# Get your input
img = read_image("path/to/your/image.png")
# Preprocess it for your chosen model
input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

with SmoothGradCAMpp(model) as cam_extractor:
  # Preprocess your data and feed it to the model
  out = model(input_tensor.unsqueeze(0))
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
```

If you want to visualize your heatmap, you only need to cast the CAM to a numpy ndarray:

```python
import matplotlib.pyplot as plt
# Visualize the raw CAM
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
```

![raw_heatmap](https://github.com/frgfm/torch-cam/releases/download/v0.1.2/raw_heatmap.png)

Or if you wish to overlay it on your input image:

```python
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
```

![overlayed_heatmap](https://github.com/frgfm/torch-cam/releases/download/v0.1.2/overlayed_heatmap.png)

## Setup

Python 3.9 (or higher) and [uv](https://docs.astral.sh/uv/)/[conda](https://docs.conda.io/en/latest/miniconda.html) are required to install TorchCAM.

### Stable release

You can install the last stable release of the package using [pypi](https://pypi.org/project/torchcam/) as follows:

```shell
pip install torchcam
```

or using [conda](https://anaconda.org/frgfm/torchcam):

```shell
conda install -c frgfm torchcam
```

### Developer installation

Alternatively, if you wish to use the latest features of the project that haven't made their way to a release yet, you can install the package from source:

```shell
git clone https://github.com/frgfm/torch-cam.git
pip install -e torch-cam/.
```



## CAM Zoo

This project is developed and maintained by the repo owner, but the implementation was based on the following research papers:

- [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150): the original CAM paper
- [Grad-CAM](https://arxiv.org/abs/1610.02391): GradCAM paper, generalizing CAM to models without global average pooling.
- [Grad-CAM++](https://arxiv.org/abs/1710.11063): improvement of GradCAM++ for more accurate pixel-level contribution to the activation.
- [Smooth Grad-CAM++](https://arxiv.org/abs/1908.01224): SmoothGrad mechanism coupled with GradCAM.
- [Score-CAM](https://arxiv.org/abs/1910.01279): score-weighting of class activation for better interpretability.
- [SS-CAM](https://arxiv.org/abs/2006.14255): SmoothGrad mechanism coupled with Score-CAM.
- [IS-CAM](https://arxiv.org/abs/2010.03023): integration-based variant of Score-CAM.
- [XGrad-CAM](https://arxiv.org/abs/2008.02312): improved version of Grad-CAM in terms of sensitivity and conservation.
- [Layer-CAM](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf): Grad-CAM alternative leveraging pixel-wise contribution of the gradient to the activation.

<p align="center">
    <a alt="wallaby_video_cam">
        <img src="https://github.com/frgfm/torch-cam/releases/download/v0.2.0/video_example_wallaby.gif" /></a>
</p>
<p align="center">
    <em>Source: <a href="https://www.youtube.com/watch?v=hZJN5BzKfxk">YouTube video</a> (activation maps created by <a href="https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.LayerCAM">Layer-CAM</a> with a pretrained <a href="https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18">ResNet-18</a>)</em>
</p>



## What else

### Documentation

The full package documentation is available [here](https://frgfm.github.io/torch-cam/) for detailed specifications.

### Demo app

A minimal demo app is provided for you to play with the supported CAM methods! Feel free to check out the live demo on [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/frgfm/torch-cam)

If you prefer running the demo by yourself, you will need an extra dependency ([Streamlit](https://streamlit.io/)) for the app to run:

```
pip install -e ".[demo]"
```

You can then easily run your app in your default browser by running:

```
streamlit run demo/app.py
```

![torchcam_demo](https://github.com/frgfm/torch-cam/releases/download/v0.2.0/torchcam_demo.png)

### Example script

An example script is provided for you to benchmark the heatmaps produced by multiple CAM approaches on the same image:

```shell
python scripts/cam_example.py --arch resnet18 --class-idx 232 --rows 2
```

![gradcam_sample](https://github.com/frgfm/torch-cam/releases/download/v0.3.1/example.png)

*All script arguments can be checked using `python scripts/cam_example.py --help`*



### Latency benchmark

You crave for beautiful activation maps, but you don't know whether it fits your needs in terms of latency?

In the table below, you will find a latency benchmark (forward pass not included) for all CAM methods:

| CAM method                                                   | Arch               | GPU mean (std)     | CPU mean (std)       |
| ------------------------------------------------------------ | ------------------ | ------------------ | -------------------- |
| [CAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.CAM) | resnet18           | 0.11ms (0.02ms)    | 0.14ms (0.03ms)      |
| [GradCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.GradCAM) | resnet18           | 3.71ms (1.11ms)    | 40.66ms (1.82ms)     |
| [GradCAMpp](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.GradCAMpp) | resnet18           | 5.21ms (1.22ms)    | 41.61ms (3.24ms)     |
| [SmoothGradCAMpp](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.SmoothGradCAMpp) | resnet18           | 33.67ms (2.51ms)   | 239.27ms (7.85ms)    |
| [ScoreCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.ScoreCAM) | resnet18           | 304.74ms (11.54ms) | 6796.89ms (415.14ms) |
| [SSCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.SSCAM) | resnet18           |                    |                      |
| [ISCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.ISCAM) | resnet18           |                    |                      |
| [XGradCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.XGradCAM) | resnet18           | 3.78ms (0.96ms)    | 40.63ms (2.03ms)     |
| [LayerCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.LayerCAM) | resnet18           | 3.65ms (1.04ms)    | 40.91ms (1.79ms)     |
| [CAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.CAM) | mobilenet_v3_large | N/A*               | N/A*                 |
| [GradCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.GradCAM) | mobilenet_v3_large | 8.61ms (1.04ms)    | 26.64ms (3.46ms)     |
| [GradCAMpp](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.GradCAMpp) | mobilenet_v3_large | 8.83ms (1.29ms)    | 25.50ms (3.10ms)     |
| [SmoothGradCAMpp](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.SmoothGradCAMpp) | mobilenet_v3_large | 77.38ms (3.83ms)   | 156.25ms (4.89ms)    |
| [ScoreCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.ScoreCAM) | mobilenet_v3_large | 35.19ms (2.11ms)   | 679.16ms (55.04ms)   |
| [SSCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.SSCAM) | mobilenet_v3_large |                    |                      |
| [ISCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.ISCAM) | mobilenet_v3_large |                    |                      |
| [XGradCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.XGradCAM) | mobilenet_v3_large | 8.41ms (0.98ms)    | 24.21ms (2.94ms)     |
| [LayerCAM](https://frgfm.github.io/torch-cam/latest/methods.html#torchcam.methods.LayerCAM) | mobilenet_v3_large | 8.02ms (0.95ms)    | 25.14ms (3.17ms)     |

**The base CAM method cannot work with architectures that have multiple fully-connected layers*

This benchmark was performed over 100 iterations on (224, 224) inputs, on a laptop to better reflect performances that can be expected by common users. The hardware setup includes an [Intel(R) Core(TM) i7-10750H](https://ark.intel.com/content/www/us/en/ark/products/201837/intel-core-i710750h-processor-12m-cache-up-to-5-00-ghz.html) for the CPU, and a [NVIDIA GeForce RTX 2070 with Max-Q Design](https://www.nvidia.com/fr-fr/geforce/graphics-cards/rtx-2070/) for the GPU.

You can run this latency benchmark for any CAM method  on your hardware as follows:

```bash
python scripts/eval_latency.py SmoothGradCAMpp
```

*All script arguments can be checked using `python scripts/eval_latency.py --help`*

### Example notebooks

Looking for more illustrations of TorchCAM features?
You might want to check the [Jupyter notebooks](notebooks) designed to give you a broader overview.



## Citation

If you wish to cite this project, feel free to use this [BibTeX](http://www.bibtex.org/) reference:

```bibtex
@misc{torcham2020,
    title={TorchCAM: class activation explorer},
    author={François-Guillaume Fernandez},
    year={2020},
    month={March},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/frgfm/torch-cam}}
}
```



## Contributing

Feeling like extending the range of possibilities of CAM? Or perhaps submitting a paper implementation? Any sort of contribution is greatly appreciated!

You can find a short guide in [`CONTRIBUTING`](CONTRIBUTING.md) to help grow this project!



## License

Distributed under the Apache 2.0 License. See [`LICENSE`](LICENSE) for more information.

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Ffrgfm%2Ftorch-cam.svg?type=large&issueType=license)](https://app.fossa.com/projects/git%2Bgithub.com%2Ffrgfm%2Ftorch-cam?ref=badge_large&issueType=license)
