# Copyright (C) 2020-2024, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.


import os
from pathlib import Path

from setuptools import setup

PKG_NAME = "torchcam"
VERSION = os.getenv("BUILD_VERSION", "0.4.1.dev0")


if __name__ == "__main__":
    print(f"Building wheel {PKG_NAME}-{VERSION}")

    # Dynamically set the __version__ attribute
    cwd = Path(__file__).parent.absolute()
    with cwd.joinpath("torchcam", "version.py").open("w", encoding="utf-8") as f:
        f.write(f"__version__ = '{VERSION}'\n")

    setup(name=PKG_NAME, version=VERSION)
