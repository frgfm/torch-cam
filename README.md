<h1 align="center">
  TorchCAM: class activation explorer
</h1>

<p align="center">
  <strong>English</strong> | <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/frgfm/torch-cam/actions/workflows/package.yml">
    <img alt="CI Status" src="https://img.shields.io/github/actions/workflow/status/frgfm/torch-cam/package.yml?branch=main&label=CI&logo=github&style=flat-square">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/Linter-Ruff-FCC21B?style=flat-square&logo=ruff&logoColor=white" alt="ruff">
  </a>
  <a href="https://github.com/astral-sh/ty">
    <img src="https://img.shields.io/badge/Typecheck-Ty-261230?style=flat-square&logo=astral&logoColor=white" alt="ty">
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
  <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/frgfm/torch-cam?label=Release&logo=github">
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
    <img src="https://img.shields.io/github/actions/workflow/status/frgfm/torch-cam/docs.yml?branch=main&label=Documentation&logo=read-the-docs&logoColor=white&style=flat-square" alt="Documentation Status">
  </a>
</p>

Simple way to leverage the class-specific activation of convolutional and transformer layers in PyTorch.

Debugging one surprising classifier result? Use the [predicted-versus-expected agent workflow](https://frgfm.github.io/torch-cam/getting-started/debug-prediction/), install the [portable skill](https://github.com/frgfm/torch-cam/blob/main/.agents/skills/torchcam-debug-prediction/SKILL.md) in your coding agent with `npx skills add frgfm/torch-cam`, or start from [`llms.txt`](https://frgfm.github.io/torch-cam/llms.txt).

<p align="center">
    <a alt="cam_examples">
        <img src="https://github.com/frgfm/torch-cam/releases/download/v0.3.1/example.png" /></a>
</p>
<p align="center">
    <em>Source: image from <a href="https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg">woopets</a> (activation maps created with a pretrained <a href="https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18">Resnet-18</a>)</em>
</p>


## Why TorchCAM

- **12 CAM methods, one API**: from CAM and Grad-CAM to Finer-CAM, LeGrad and RefineCAM.
- **CNNs and Vision Transformers**: automatic target-layer resolution for CNNs, LeGrad and reshape transforms for ViTs.
- **Lean**: fully typed, with only 4 runtime dependencies (PyTorch, NumPy, Pillow, Matplotlib).
- **Measurable**: built-in faithfulness metrics (average drop, increase in confidence, deletion/insertion).
- **Agent-ready**: `explain()` saves a manifest-backed evidence bundle that humans and coding agents can verify.

## Quick Tour

### Explain a prediction

```python
from urllib.request import urlretrieve
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
from torchcam.explain import explain

urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
image = Image.open("dog.jpg").convert("RGB")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights).eval()

result = explain(model, weights.transforms()(image).unsqueeze(0), class_names=weights.meta["categories"])
result.save("torchcam-explanation", image)  # CAMs, heatmaps, overlays and manifest.json
```

Pass `expected_class_idx` to compare the prediction with the class you expected. See [Debug one prediction](https://frgfm.github.io/torch-cam/getting-started/debug-prediction/) for the full contract.

### Setting your CAM

TorchCAM leverages [PyTorch hooking mechanisms](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks) to seamlessly retrieve all required information to produce the class activation without additional efforts from the user. Each CAM object acts as a wrapper around your model.

You can find the exhaustive list of supported CAM methods in the [documentation](https://frgfm.github.io/torch-cam/reference/methods/), then use it as follows:

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM

# Define your model
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
# Set your CAM extractor
cam_extractor = LayerCAM(model)
```

*Please note that by default, the layer at which the CAM is retrieved is set to the last non-reduced convolutional layer. If you wish to investigate a specific layer, use the `target_layer` argument in the constructor.*

### Retrieving the class activation map

Once your CAM extractor is set, you only need to use your model to infer on your data as usual. If any additional information is required, the extractor will get it for you automatically.

<!-- --8<-- [start:quickstart-input] -->
```python
from urllib.request import urlretrieve
from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights

# Get a model and an image
weights = get_model_weights("resnet18").DEFAULT
model = get_model("resnet18", weights=weights).eval()
preprocess = weights.transforms()
urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
img = decode_image("dog.jpg")

input_tensor = preprocess(img)
```
<!-- --8<-- [end:quickstart-input] -->

Compute the class activation map:

<!-- --8<-- [start:quickstart-cam] -->
```python hl_lines="3 6"
from torchcam.methods import LayerCAM

with LayerCAM(model) as cam_extractor:
  out = model(input_tensor.unsqueeze(0))
  # Retrieve the CAM by passing the class index and the model output
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
```
<!-- --8<-- [end:quickstart-cam] -->

Here `class_idx` (the first argument) is the index in the model's output logits of the class you want to explain — `out.squeeze(0).argmax().item()` picks the top prediction, but you can pass any class index. The extractor returns one activation map per target layer.

If you want to visualize your heatmap, you only need to cast the CAM to a numpy ndarray:

```python
import matplotlib.pyplot as plt
# Visualize the raw CAM
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
```

![raw_heatmap](https://github.com/frgfm/torch-cam/releases/download/v0.1.2/raw_heatmap.png)

Or if you wish to overlay it on your input image:

<!-- --8<-- [start:quickstart-overlay] -->
```python hl_lines="3 6"
import matplotlib.pyplot as plt
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask

# Resize the CAM and overlay it
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
```
<!-- --8<-- [end:quickstart-overlay] -->

![overlayed_heatmap](https://github.com/frgfm/torch-cam/releases/download/v0.1.2/overlayed_heatmap.png)

> [!TIP]
> Using your own (non-torchvision) model, a Vision Transformer, 3D/video data, or batched inputs? Read the [**Advanced usage guide**](https://frgfm.github.io/torch-cam/getting-started/advanced-usage/) — it also covers how to choose the right `target_layer` and CAM method.
>
> Hitting a `cannot register a hook ...` / `requires grad` error, a `NaN`, or a blank heatmap? See [**Troubleshooting**](https://frgfm.github.io/torch-cam/getting-started/troubleshooting/).

## Setup

Python 3.11 (or higher) and [uv](https://docs.astral.sh/uv/)/[pip](https://pip.pypa.io/en/stable/installation/) are required to install TorchCAM. TorchCAM 0.5.0 requires PyTorch 2.4.1 or higher (`torch>=2.4.1,<3`) and supports NumPy 1.26.4 and NumPy 2 (`numpy>=1.26.4,<3`). Use the matching torchvision release.

On macOS, TorchCAM 0.5.0 requires Apple Silicon when using published PyTorch wheels. Intel Mac users can use Python 3.11 or 3.12 and install the previous release with `pip install "torchcam==0.4.1"`.

### Stable release

You can install the last stable release of the package using [pypi](https://pypi.org/project/torchcam/) as follows:

```shell
pip install torchcam
```

### Latest version

Alternatively, if you wish to use the latest features of the project that haven't made their way to a release yet, you can install the package from source:

```shell
pip install "torchcam @ git+https://github.com/frgfm/torch-cam.git"
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
- [Finer-CAM](https://arxiv.org/abs/2501.11309): contrastive CAM objective highlighting differences between similar classes.
- [LeGrad](https://arxiv.org/abs/2404.03214): layerwise positive attention-gradient maps for Vision Transformers.
- [RefineCAM](https://arxiv.org/abs/2605.14641): multi-layer refinement producing high-resolution activation maps.

*Not sure which one to use? See [Choosing a CAM method](https://frgfm.github.io/torch-cam/getting-started/advanced-usage/#choosing-a-cam-method).*

<p align="center">
    <a alt="wallaby_video_cam">
        <img src="https://github.com/frgfm/torch-cam/releases/download/v0.2.0/video_example_wallaby.gif" /></a>
</p>
<p align="center">
    <em>Source: <a href="https://www.youtube.com/watch?v=hZJN5BzKfxk">YouTube video</a> (activation maps created by <a href="https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.LayerCAM">Layer-CAM</a> with a pretrained <a href="https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18">ResNet-18</a>)</em>
</p>



## What else

### Documentation

The full package documentation is available [here](https://frgfm.github.io/torch-cam/) for detailed specifications.

### Playground app

A minimal demo app is provided for you to play with the supported CAM methods! Feel free to check out the live demo on [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/frgfm/torch-cam)

The demo accepts JPEG and PNG uploads.

If you prefer running the demo by yourself, you will need an extra dependency ([Streamlit](https://streamlit.io/)) for the app to run:

```
pip install -e ".[demo]"
```

You can then easily run your app in your default browser by running:

```
streamlit run demo/app.py
```

![torchcam_demo](https://github.com/frgfm/torch-cam/releases/download/v0.2.0/torchcam_demo.png)

### Visualization script

An example script is provided for you to benchmark the heatmaps produced by multiple CAM approaches on the same image:

```shell
python scripts/cam_example.py --arch resnet18 --class-idx 232 --rows 2
```

![gradcam_sample](https://github.com/frgfm/torch-cam/releases/download/v0.3.1/example.png)

*All script arguments can be checked using `python scripts/cam_example.py --help`*

### Performance benchmarks

The purpose of CAM methods is to provide interpretability and they do so by pointing the biggest influence factors on the model outputs. Ideally the CAM should pinpoint all the visual cues that have any influence of the output classification score.
For this, we use two metrics:

- [Increase in Confidence](https://frgfm.github.io/torch-cam/reference/metrics/#torchcam.metrics.ClassificationMetric) (higher is better): the fraction of inputs for which masking with the CAM increases the probability of the original predicted class.
- [Average Drop](https://frgfm.github.io/torch-cam/reference/metrics/#torchcam.metrics.ClassificationMetric) (lower is better): the mean relative decrease in that class probability after masking, with increases counted as zero drop.

The table below records **historical October 2025 results**, obtained with earlier metric and CAM implementations. These values have not been revalidated with the current implementation. The ResNet-18 LayerCAM row has been corrected to match the recorded CSV and [original benchmark notebook](https://github.com/frgfm/notebooks/blob/main/torch-cam/performance_benchmark.ipynb).

| CAM method | Arch | Average drop (↓) | Increase in confidence (↑) |
| ---------- | ---- | ---------------- | -------------------------- |
| [GradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAM) | resnet18 | 0.2686 | 0.2250 |
| [GradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAMpp) | resnet18 | 0.5271 | 0.1962 |
| [SmoothGradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.SmoothGradCAMpp) | resnet18 | 0.2088 | 0.2499 |
| [LayerCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.LayerCAM) | resnet18 | 0.1805 | 0.2894 |
| [GradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAM) | mobilenet_v3_large | 0.2678 | 0.3483 |
| [GradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAMpp) | mobilenet_v3_large | 0.3182 | 0.2535 |
| [SmoothGradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.SmoothGradCAMpp) | mobilenet_v3_large | 0.2681 | 0.2678 |
| [LayerCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.LayerCAM) | mobilenet_v3_large | 0.2526 | 0.2882 |

The recorded protocol used the validation set of [imagenette2-320](https://github.com/fastai/imagenette), `Resize(256)` followed by `CenterCrop(224)`, and target layers `layer4` for ResNet-18 and `features` for MobileNet V3 Large. It evaluated the original predicted class. Masking multiplies the ImageNet-normalized input by the CAM: a zero mask therefore corresponds to the ImageNet mean RGB color, not black. No original random seed was recorded; stochastic CAMs and changes to CAM or metric numerics require a fresh benchmark run.

You can run a new benchmark on your hardware with an explicit seed and weight version as follows:

```bash
python scripts/eval_perf.py ~/Downloads/imagenette2-320 LayerCAM --arch mobilenet_v3_large --seed 0 --weights IMAGENET1K_V2
```

The optional `--deletion-insertion` evaluation uses a normalized zero baseline for both curves and 20 perturbation steps by default. It generates CAMs separately from the classification metrics, consuming extra random draws that can change subsequent stochastic classification CAMs even with the same seed; compare stochastic methods using the same flags. This differs from the original RISE protocol, which uses a blurred image as its insertion baseline; scores from different protocols are not directly comparable.

*All script arguments can be checked using `python scripts/eval_perf.py --help`*

### Latency benchmark

You crave for beautiful activation maps, but you don't know whether it fits your needs in terms of latency?

The table below preserves **historical October 2021 CPU latency measurements** (initial forward pass not included), from the [original benchmark commit](https://github.com/frgfm/torch-cam/commit/8237aef5756daeb85a49f87293cafef35277c295). They have not been revalidated with current implementations. The GPU column has been retired because the original timer did not synchronize CUDA operations; the current `scripts/eval_latency.py` synchronizes CUDA before and after timing.

| CAM method | Arch | CPU mean (std) |
| ---------- | ---- | -------------- |
| [CAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.CAM) | resnet18           | 0.14ms (0.03ms)      |
| [GradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAM) | resnet18           | 40.66ms (1.82ms)     |
| [GradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAMpp) | resnet18           | 41.61ms (3.24ms)     |
| [SmoothGradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.SmoothGradCAMpp) | resnet18           | 239.27ms (7.85ms)    |
| [ScoreCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.ScoreCAM) | resnet18           | 6796.89ms (415.14ms) |
| [XGradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.XGradCAM) | resnet18           | 40.63ms (2.03ms)     |
| [LayerCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.LayerCAM) | resnet18           | 40.91ms (1.79ms)     |
| [CAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.CAM) | mobilenet_v3_large | N/A*                 |
| [GradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAM) | mobilenet_v3_large | 26.64ms (3.46ms)     |
| [GradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAMpp) | mobilenet_v3_large | 25.50ms (3.10ms)     |
| [SmoothGradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.SmoothGradCAMpp) | mobilenet_v3_large | 156.25ms (4.89ms)    |
| [ScoreCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.ScoreCAM) | mobilenet_v3_large | 679.16ms (55.04ms)   |
| [XGradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.XGradCAM) | mobilenet_v3_large | 24.21ms (2.94ms)     |
| [LayerCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.LayerCAM) | mobilenet_v3_large | 25.14ms (3.17ms)     |

**The base CAM method cannot work with architectures that have multiple fully-connected layers*

These CPU measurements used 100 iterations on (224, 224) inputs and a laptop with an [Intel(R) Core(TM) i7-10750H](https://ark.intel.com/content/www/us/en/ark/products/201837/intel-core-i710750h-processor-12m-cache-up-to-5-00-ghz.html).

You can run this latency benchmark for any CAM method  on your hardware as follows:

```bash
uv run --extra scripts python scripts/eval_latency.py SmoothGradCAMpp --device cpu --weights none --output latency.json
```

Each command runs five fresh processes with one CPU thread. It reports the first call after extractor setup, then runs 10 full CAM warm-up calls before collecting 100 samples. Use `--repeat`, `--threads`, `--warmup`, and `--it` to change these settings. `--weights none` uses an untrained model and avoids downloads; omit it to use pretrained weights.

The default `--scope extractor` excludes the initial model forward pass. `--scope end-to-end` includes the forward pass and CAM extraction; both exclude image loading, preprocessing, device transfers, and extractor setup. The JSON report saves raw samples, median/p95 latency, resolved layers and weights, versions, revision, and script hash. Peak RSS is the highest RAM use of a worker through setup and measurement, including output checks. CUDA allocated and reserved memory are reported separately. RSS is unavailable on Windows.

ViT and Swin spatial methods use the same reshape setup as the example script. LeGrad needs an explicit transformer block (`--target-layer encoder.layers.encoder_layer_11` for `vit_b_16`). RefineCAM needs at least two repeated `--target-layer` arguments. The example script accepts these layers as `--target layer3,layer4`.

*All script arguments can be checked using `python scripts/eval_latency.py --help`*

### Example notebooks

Explore these runnable notebooks, hosted in [frgfm/notebooks](https://github.com/frgfm/notebooks/tree/main/torch-cam):

Each notebook includes setup cells that pin a development snapshot with fixes after TorchCAM 0.5.0. Run those cells to use the intended implementation.

| Notebook | Use case | Run |
|:---------|:---------|:----|
| [Quicktour](https://github.com/frgfm/notebooks/blob/main/torch-cam/quicktour.ipynb) | Extract CAMs, create overlays, and fuse multiple layers | [Colab](https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/quicktour.ipynb) |
| [Debug a prediction](https://github.com/frgfm/notebooks/blob/main/torch-cam/debug_prediction.ipynb) | Compare predicted and expected classes and save an evidence bundle | [Colab](https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/debug_prediction.ipynb) |
| [Vision Transformers](https://github.com/frgfm/notebooks/blob/main/torch-cam/vision_transformers.ipynb) | Explain a torchvision ViT with LeGrad and token-based GradCAM | [Colab](https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/vision_transformers.ipynb) |
| [Latency benchmark](https://github.com/frgfm/notebooks/blob/main/torch-cam/latency_benchmark.ipynb) | Measure CAM extraction and end-to-end latency on your hardware | [Colab](https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/latency_benchmark.ipynb) |
| [Performance benchmark](https://github.com/frgfm/notebooks/blob/main/torch-cam/performance_benchmark.ipynb) | Evaluate confidence and deletion/insertion faithfulness with an explicit baseline | [Colab](https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/performance_benchmark.ipynb) |

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
