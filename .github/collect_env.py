# Copyright (C) 2020-2025, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

"""Outputs relevant system environment info for bug reports. Run it with `python collect_env.py`."""

from importlib.metadata import PackageNotFoundError, version

try:
    from torch.utils.collect_env import get_pretty_env_info
except ImportError:
    get_pretty_env_info = None

try:
    torchcam_version = version("torchcam")
except PackageNotFoundError:
    torchcam_version = "N/A"


def main():
    print("Collecting environment information...")
    print(f"TorchCAM version: {torchcam_version}\n")
    print(get_pretty_env_info() if get_pretty_env_info is not None else "PyTorch not found")


if __name__ == "__main__":
    main()
