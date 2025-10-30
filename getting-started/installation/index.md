# Installation

## Virtual environment

Tip

You will need an environment manager, and I cannot recommend enough [uv](https://docs.astral.sh/uv/getting-started/installation/).

Create a virtual environment with your prefered Python version (3.11 or higher is required to use TorchCAM):

```bash
$ uv venv --python 3.11
```

```bash
$ uv pip install torchcam
```

```bash
$ uv pip install torchcam @ git+https://github.com/frgfm/torch-cam.git
```

## System installation

You'll need [Python](https://www.python.org/downloads/) 3.11 or higher, and a package installer like [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/).

```bash
$ uv pip install --system torchcam
```

```bash
$ uv pip install --system torchcam @ git+https://github.com/frgfm/torch-cam.git
```

```bash
$ pip install torchcam
```

```bash
$ pip install torchcam @ git+https://github.com/frgfm/torch-cam.git
```

Info

TorchCAM is built on top of [PyTorch](https://github.com/pytorch/pytorch) which is a complex dependency. Proper installation depends on your system and available hardware. You can refer to [installation guide of uv](https://docs.astral.sh/uv/guides/integration/pytorch) which is quite detailed.
