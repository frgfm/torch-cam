<h1 align="center">
  TorchCAM：类激活图探索工具
</h1>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

> [!NOTE]
> 本文是英文 README 的简体中文翻译。若两者内容不一致，请以[英文版](README.md)为准。

<p align="center">
  <a href="https://github.com/frgfm/torch-cam/actions/workflows/package.yml">
    <img alt="CI 状态" src="https://img.shields.io/github/actions/workflow/status/frgfm/torch-cam/package.yml?branch=main&label=CI&logo=github&style=flat-square">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/Linter-Ruff-FCC21B?style=flat-square&logo=ruff&logoColor=white" alt="ruff">
  </a>
  <a href="https://github.com/astral-sh/ty">
    <img src="https://img.shields.io/badge/Typecheck-Ty-261230?style=flat-square&logo=astral&logoColor=white" alt="ty">
  </a>
  <a href="https://www.codacy.com/gh/frgfm/torch-cam/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=frgfm/torch-cam&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/87eaeec3e15442188f96c36bace5faf4"/></a>
  <a href="https://codecov.io/gh/frgfm/torch-cam">
    <img src="https://img.shields.io/codecov/c/github/frgfm/torch-cam.svg?logo=codecov&style=flat-square&label=Coverage" alt="测试覆盖率">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/torchcam/">
    <img src="https://img.shields.io/pypi/v/torchcam.svg?logo=PyPI&logoColor=fff&style=flat-square&label=PyPI" alt="PyPI 版本">
  </a>
  <img alt="GitHub 最新版本" src="https://img.shields.io/github/v/release/frgfm/torch-cam?label=Release&logo=github">
  <img src="https://img.shields.io/pypi/pyversions/torchcam.svg?logo=Python&label=Python&logoColor=fff&style=flat-square" alt="Python 版本">
  <a href="https://github.com/frgfm/torch-cam/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/frgfm/torch-cam.svg?label=License&logoColor=fff&style=flat-square" alt="许可证">
  </a>
</p>
<p align="center">
  <a href="https://huggingface.co/spaces/frgfm/torch-cam">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces">
  </a>
  <a href="https://colab.research.google.com/github/frgfm/notebooks/blob/main/torch-cam/quicktour.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 中打开">
  </a>
</p>
<p align="center">
  <a href="https://frgfm.github.io/torch-cam">
    <img src="https://img.shields.io/github/actions/workflow/status/frgfm/torch-cam/page-build.yml?branch=main&label=Documentation&logo=read-the-docs&logoColor=white&style=flat-square" alt="文档状态">
  </a>
</p>

一种简单的方法，用于提取 PyTorch 卷积层中特定类别的激活信息。

<p align="center">
    <a alt="CAM 示例">
        <img src="https://github.com/frgfm/torch-cam/releases/download/v0.3.1/example.png" /></a>
</p>
<p align="center">
    <em>来源：<a href="https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg">woopets</a> 的图片（激活图由预训练的 <a href="https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18">ResNet-18</a> 生成）</em>
</p>


## 快速上手

### 配置 CAM

TorchCAM 利用 [PyTorch 的钩子机制](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks)，自动获取生成类激活图所需的全部信息，无需用户执行额外操作。每个 CAM 对象都是模型的一个封装器。

你可以在[文档](https://frgfm.github.io/torch-cam/reference/methods/)中查看所有支持的 CAM 方法，然后按如下方式使用：

```python
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM

# 定义模型
model = get_model("resnet18", weights=get_model_weights("resnet18").DEFAULT).eval()
# 配置 CAM 提取器
cam_extractor = LayerCAM(model)
```

*请注意，默认情况下，TorchCAM 会从最后一个尚未降维的卷积层提取 CAM。如果要分析指定层，请在构造函数中传入 `target_layer` 参数。*

### 获取类激活图

配置好 CAM 提取器后，只需像往常一样使用模型对数据进行推理。如果需要其他信息，提取器会自动获取。

```python
from torchvision.io import decode_image
from torchvision.models import get_model, get_model_weights
from torchcam.methods import LayerCAM

# 获取模型和图片
weights = get_model_weights("resnet18").DEFAULT
model = get_model("resnet18", weights=weights).eval()
preprocess = weights.transforms()
img = decode_image("path/to/your/image.jpg")

input_tensor = preprocess(img)

with LayerCAM(model) as cam_extractor:
  out = model(input_tensor.unsqueeze(0))
  # 传入类别索引和模型输出以获取 CAM
  activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
```

这里的 `class_idx`（第一个参数）是要解释的类别在模型输出 logits 中的索引。`out.squeeze(0).argmax().item()` 会选择预测概率最高的类别，你也可以传入任意类别索引。提取器会为每个目标层返回一张激活图。

要显示热力图，只需将 CAM 转换为 NumPy 数组：

```python
import matplotlib.pyplot as plt
# 显示原始 CAM
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()
```

![原始热力图](https://github.com/frgfm/torch-cam/releases/download/v0.1.2/raw_heatmap.png)

也可以将热力图叠加到输入图片上：

```python
import matplotlib.pyplot as plt
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.utils import overlay_mask

# 调整 CAM 大小并叠加到图片上
result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
```

![叠加热力图](https://github.com/frgfm/torch-cam/releases/download/v0.1.2/overlayed_heatmap.png)

> [!TIP]
> 正在使用自己的模型（而非 torchvision 模型）、Vision Transformer、3D/视频数据或批量输入？请阅读[**高级用法指南**](https://frgfm.github.io/torch-cam/getting-started/advanced-usage/)，其中还介绍了如何选择合适的 `target_layer` 和 CAM 方法。
>
> 遇到 `cannot register a hook ...`、`requires grad`、`NaN` 或空白热力图？请参阅[**故障排查指南**](https://frgfm.github.io/torch-cam/getting-started/troubleshooting/)。

## 安装

安装 TorchCAM 需要 Python 3.11 或更高版本，以及 [uv](https://docs.astral.sh/uv/) 或 [pip](https://pip.pypa.io/en/stable/installation/)。TorchCAM 支持 PyTorch 2.x（`torch>=2.0`）及其对应版本的 torchvision。

### 稳定版本

可以通过 [PyPI](https://pypi.org/project/torchcam/) 安装最新的稳定版本：

```shell
pip install torchcam
```

### 最新开发版本

如果想使用尚未发布的最新功能，可以从源码安装：

```shell
pip install "torchcam @ git+https://github.com/frgfm/torch-cam.git"
```


## CAM 方法一览

本项目由仓库所有者开发和维护，其实现基于以下研究论文：

- [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150)：最初的 CAM 论文
- [Grad-CAM](https://arxiv.org/abs/1610.02391)：GradCAM 论文，将 CAM 推广到没有全局平均池化的模型。
- [Grad-CAM++](https://arxiv.org/abs/1710.11063)：对 GradCAM++ 的改进，可以更准确地衡量像素对激活的贡献。
- [Smooth Grad-CAM++](https://arxiv.org/abs/1908.01224)：将 SmoothGrad 机制与 GradCAM 结合。
- [Score-CAM](https://arxiv.org/abs/1910.01279)：通过分类得分对类激活进行加权，以提高可解释性。
- [SS-CAM](https://arxiv.org/abs/2006.14255)：将 SmoothGrad 机制与 Score-CAM 结合。
- [IS-CAM](https://arxiv.org/abs/2010.03023)：Score-CAM 的积分变体。
- [XGrad-CAM](https://arxiv.org/abs/2008.02312)：在敏感性和守恒性方面改进的 Grad-CAM。
- [Layer-CAM](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf)：Grad-CAM 的替代方案，利用梯度对激活的逐像素贡献。
- [Finer-CAM](https://arxiv.org/abs/2501.11309)：通过对比相似类别突出细粒度差异的 CAM 目标。
- [LeGrad](https://arxiv.org/abs/2404.03214)：利用视觉 Transformer 各层注意力概率的正梯度生成解释图。
- [RefineCAM](https://arxiv.org/abs/2605.14641)：融合多个网络层以生成高分辨率类激活图。

*不知道该用哪一种？请参阅[如何选择 CAM 方法](https://frgfm.github.io/torch-cam/getting-started/advanced-usage/#choosing-a-cam-method)。*

<p align="center">
    <a alt="袋鼠视频 CAM">
        <img src="https://github.com/frgfm/torch-cam/releases/download/v0.2.0/video_example_wallaby.gif" /></a>
</p>
<p align="center">
    <em>来源：<a href="https://www.youtube.com/watch?v=hZJN5BzKfxk">YouTube 视频</a>（激活图由 <a href="https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.LayerCAM">Layer-CAM</a> 和预训练的 <a href="https://pytorch.org/vision/stable/models.html#torchvision.models.resnet18">ResNet-18</a> 生成）</em>
</p>



## 更多内容

### 文档

完整的软件包文档可在[这里](https://frgfm.github.io/torch-cam/)查看，其中包含详细说明。

### 在线演示

项目提供了一个简洁的演示应用，供你体验支持的 CAM 方法。欢迎访问 [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/frgfm/torch-cam) 在线试用。

如果希望在本地运行演示，需要安装额外的依赖项 [Streamlit](https://streamlit.io/)：

```
pip install -e ".[demo]"
```

然后运行以下命令，即可在默认浏览器中打开应用：

```
streamlit run demo/app.py
```

![TorchCAM 演示](https://github.com/frgfm/torch-cam/releases/download/v0.2.0/torchcam_demo.png)

### 可视化脚本

项目提供了一个示例脚本，用于在同一张图片上比较多种 CAM 方法生成的热力图：

```shell
python scripts/cam_example.py --arch resnet18 --class-idx 232 --rows 2
```

![GradCAM 示例](https://github.com/frgfm/torch-cam/releases/download/v0.3.1/example.png)

*可以运行 `python scripts/cam_example.py --help` 查看脚本的所有参数。*

### 效果评测

CAM 方法旨在通过指出对模型输出影响最大的因素来提高模型的可解释性。理想情况下，CAM 应标出所有会影响分类分数的视觉线索。
这里使用两个指标：

- [置信度提升（Increase in Confidence）](https://frgfm.github.io/torch-cam/reference/metrics/#torchcam.metrics.ClassificationMetric)（越高越好）：将输入与 CAM 相乘（保留 CAM 值高处的原始像素，将 CAM 值低处置零）后再次进行前向传播，统计数据集中分类概率提高的次数。
- [平均下降（Average Drop）](https://frgfm.github.io/torch-cam/reference/metrics/#torchcam.metrics.ClassificationMetric)（越低越好）：将输入与 CAM 相乘（保留 CAM 值高处的原始像素，将 CAM 值低处置零）后再次进行前向传播，衡量分类概率下降的幅度。

| CAM 方法 | 架构 | 平均下降（↓） | 置信度提升（↑） |
| -------- | ---- | ------------- | ---------------- |
| [GradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAM) | resnet18 | 0.2686 | 0.2250 |
| [GradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAMpp) | resnet18 | 0.5271 | 0.1962 |
| [SmoothGradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.SmoothGradCAMpp) | resnet18 | 0.2088 | 0.2499 |
| [LayerCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.LayerCAM) | resnet18 | 0.1712 | 0.2819 |
| [GradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAM) | mobilenet_v3_large | 0.2678 | 0.3483 |
| [GradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAMpp) | mobilenet_v3_large | 0.3182 | 0.2535 |
| [SmoothGradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.SmoothGradCAMpp) | mobilenet_v3_large | 0.2681 | 0.2678 |
| [LayerCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.LayerCAM) | mobilenet_v3_large | 0.2526 | 0.2882 |

该评测在 [Imagenette](https://github.com/fastai/imagenette) 验证集上进行。Imagenette 是 ImageNet 的一个子集，输入尺寸为 (224, 224)。

可以在自己的硬件上运行以下命令，评测任意 CAM 方法：

```bash
python scripts/eval_perf.py ~/Downloads/imagenette LayerCAM --arch mobilenet_v3_large
```

*可以运行 `python scripts/eval_perf.py --help` 查看脚本的所有参数。*

### 延迟基准测试

想生成漂亮的激活图，却不知道它的延迟是否满足需求？

下表给出了所有 CAM 方法的额外延迟（不包括前向传播）：

| CAM 方法 | 架构 | GPU 平均值（标准差） | CPU 平均值（标准差） |
| -------- | ---- | -------------------- | -------------------- |
| [CAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.CAM) | resnet18           | 0.11ms (0.02ms)    | 0.14ms (0.03ms)      |
| [GradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAM) | resnet18           | 3.71ms (1.11ms)    | 40.66ms (1.82ms)     |
| [GradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAMpp) | resnet18           | 5.21ms (1.22ms)    | 41.61ms (3.24ms)     |
| [SmoothGradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.SmoothGradCAMpp) | resnet18           | 33.67ms (2.51ms)   | 239.27ms (7.85ms)    |
| [ScoreCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.ScoreCAM) | resnet18           | 304.74ms (11.54ms) | 6796.89ms (415.14ms) |
| [XGradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.XGradCAM) | resnet18           | 3.78ms (0.96ms)    | 40.63ms (2.03ms)     |
| [LayerCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.LayerCAM) | resnet18           | 3.65ms (1.04ms)    | 40.91ms (1.79ms)     |
| [CAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.CAM) | mobilenet_v3_large | 不适用*            | 不适用*              |
| [GradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAM) | mobilenet_v3_large | 8.61ms (1.04ms)    | 26.64ms (3.46ms)     |
| [GradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.GradCAMpp) | mobilenet_v3_large | 8.83ms (1.29ms)    | 25.50ms (3.10ms)     |
| [SmoothGradCAMpp](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.SmoothGradCAMpp) | mobilenet_v3_large | 77.38ms (3.83ms)   | 156.25ms (4.89ms)    |
| [ScoreCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.ScoreCAM) | mobilenet_v3_large | 35.19ms (2.11ms)   | 679.16ms (55.04ms)   |
| [XGradCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.XGradCAM) | mobilenet_v3_large | 8.41ms (0.98ms)    | 24.21ms (2.94ms)     |
| [LayerCAM](https://frgfm.github.io/torch-cam/reference/methods/#torchcam.methods.LayerCAM) | mobilenet_v3_large | 8.02ms (0.95ms)    | 25.14ms (3.17ms)     |

**基础 CAM 方法无法用于包含多个全连接层的架构。*

该基准测试使用 (224, 224) 输入，在一台笔记本电脑上迭代 100 次完成，以更贴近普通用户能够获得的性能。硬件配置为 [Intel(R) Core(TM) i7-10750H](https://ark.intel.com/content/www/us/en/ark/products/201837/intel-core-i710750h-processor-12m-cache-up-to-5-00-ghz.html) CPU 和 [NVIDIA GeForce RTX 2070 with Max-Q Design](https://www.nvidia.com/fr-fr/geforce/graphics-cards/rtx-2070/) GPU。

可以在自己的硬件上运行以下命令，对任意 CAM 方法进行延迟测试：

```bash
python scripts/eval_latency.py SmoothGradCAMpp
```

*可以运行 `python scripts/eval_latency.py --help` 查看脚本的所有参数。*

### 示例笔记本

想查看更多 TorchCAM 功能示例？
可以查看 [Jupyter 笔记本](notebooks)，获得更全面的了解。

## 引用

如果要引用本项目，可以使用以下 [BibTeX](http://www.bibtex.org/)：

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

## 参与贡献

想扩展 CAM 的功能，或提交某篇论文的实现？欢迎任何形式的贡献！

请阅读 [`CONTRIBUTING`](CONTRIBUTING.md) 中的简短指南，帮助项目不断成长。

## 许可证

本项目采用 Apache 2.0 许可证发布。详情请参阅 [`LICENSE`](LICENSE)。

[![FOSSA 状态](https://app.fossa.com/api/projects/git%2Bgithub.com%2Ffrgfm%2Ftorch-cam.svg?type=large&issueType=license)](https://app.fossa.com/projects/git%2Bgithub.com%2Ffrgfm%2Ftorch-cam?ref=badge_large&issueType=license)
