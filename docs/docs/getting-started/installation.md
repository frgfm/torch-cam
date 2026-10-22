# Installation

The online documentation follows the repository's `main` branch. Choose the install that matches the API you
want to use:

| Source | Install | What you get |
| --- | --- | --- |
| **PyPI (stable)** | `pip install torchcam` | Latest tagged [release](https://github.com/frgfm/torch-cam/releases/latest). |
| **Git (`main`)** | `pip install "torchcam @ git+https://github.com/frgfm/torch-cam.git"` | Unreleased changes matching this site. |

The development version requires PyTorch 2.13 or higher within the 2.x series and supports both NumPy 1.26.4
and NumPy 2 (`numpy>=1.26.4,<3`). This floor includes fixes for
[GHSA-53q9-r3pm-6pq6](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6),
[GHSA-63cw-57p8-fm3p](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p), and
[GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47). These version floors do not guarantee
that future advisories will not affect a resolved environment. If you install torchvision for the demo or examples,
use the release matching your PyTorch version; PyTorch 2.13 pairs with torchvision 0.28.

Matplotlib 3.8.4 or higher, below version 4, is also required. Earlier Matplotlib wheels can fail to import with NumPy 2.

Pillow 13 or higher is required for the core library and demo; its [release notes](https://pillow.readthedocs.io/en/latest/releasenotes/13.0.0.html) list the image-parsing fixes.
The checked-in [demo requirements](https://github.com/frgfm/torch-cam/blob/main/demo/requirements.txt) pin the Python 3.13 Linux CPU environment.
Refresh them with `make lock-demo` and re-audit before deployment.

On macOS, the PyTorch 2.13 wheels support Apple Silicon on macOS 14 or later. PyTorch does not publish matching
Intel Mac wheels at this version floor.

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
    $ uv pip install "torchcam @ git+https://github.com/frgfm/torch-cam.git"
    ```


## System installation

You'll need [Python](https://www.python.org/downloads/) 3.11 or higher, and a package installer like [uv](https://docs.astral.sh/uv/getting-started/installation/) or [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/).

=== "Stable"

    ```bash
    $ uv pip install --system torchcam
    ```

=== "Latest"

    ```bash
    $ uv pip install --system "torchcam @ git+https://github.com/frgfm/torch-cam.git"
    ```

=== "Stable (pip)"

    ```bash
    $ pip install torchcam
    ```

=== "Latest (pip)"

    ```bash
    $ pip install "torchcam @ git+https://github.com/frgfm/torch-cam.git"
    ```

!!! info
    TorchCAM is built on top of [PyTorch](https://github.com/pytorch/pytorch) which is a complex dependency. Proper installation depends on your system and available hardware. You can refer to [installation guide of uv](https://docs.astral.sh/uv/guides/integration/pytorch) which is quite detailed.

For common pitfalls (e.g. `no_grad` with Grad-CAM, hook cleanup, picking `target_layer`), see [Troubleshooting](troubleshooting.md).
