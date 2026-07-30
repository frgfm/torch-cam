# Installation

The online documentation follows the repository's `main` branch. Choose the install that matches the API you
want to use:

| Source | Install | What you get |
| --- | --- | --- |
| **PyPI (stable)** | `pip install torchcam` | Latest tagged [release](https://github.com/frgfm/torch-cam/releases/latest). |
| **Git (`main`)** | `pip install "torchcam @ git+https://github.com/frgfm/torch-cam.git"` | Unreleased changes matching this site. |

Check the installed version when an example and your environment behave differently:

```python
import importlib.metadata

print(importlib.metadata.version("torchcam"))
```

## Virtual environment

!!! tip
    You will need an environment manager, and I cannot recommend enough [uv](https://docs.astral.sh/uv/getting-started/installation/).

Create a virtual environment with your preferred Python version (3.11 or higher is required to use TorchCAM):
```bash
$ uv venv --python 3.11
```

=== "Stable"

    ```bash
    $ uv pip install torchcam
    ```

=== "Latest"

    ```bash
    $ uv pip install torchcam @ git+https://github.com/frgfm/torch-cam.git
    ```


## System installation

You'll need [Python](https://www.python.org/downloads/) 3.11 or higher, and a package installer like [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/).

=== "Stable"

    ```bash
    $ uv pip install --system torchcam
    ```

=== "Latest"

    ```bash
    $ uv pip install --system torchcam @ git+https://github.com/frgfm/torch-cam.git
    ```

=== "Stable (pip)"

    ```bash
    $ pip install torchcam
    ```

=== "Latest (pip)"

    ```bash
    $ pip install torchcam @ git+https://github.com/frgfm/torch-cam.git
    ```

!!! info
    TorchCAM is built on top of [PyTorch](https://github.com/pytorch/pytorch) which is a complex dependency. Proper installation depends on your system and available hardware. You can refer to [installation guide of uv](https://docs.astral.sh/uv/guides/integration/pytorch) which is quite detailed.

For common pitfalls (e.g. `no_grad` with Grad-CAM, hook cleanup, picking `target_layer`), see [Troubleshooting](troubleshooting.md).
