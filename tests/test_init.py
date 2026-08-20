import importlib.metadata
import subprocess  # noqa: S404
import sys

import pytest

import torchcam

_PUBLIC_MODULES = ("explain", "methods", "metrics", "utils")


def test_lazy_imports_in_fresh_process() -> None:
    script = f"""
import sys
import torchcam

names = {_PUBLIC_MODULES!r}
assert "torch" not in sys.modules
assert "matplotlib.pyplot" not in sys.modules
assert all(f"torchcam.{{name}}" not in sys.modules for name in names)
assert all(name not in vars(torchcam) for name in names)
assert torchcam.__version__

from torchcam import *
from torchcam import explain, methods, metrics, utils

modules = (explain, methods, metrics, utils)
assert modules == (torchcam.explain, torchcam.methods, torchcam.metrics, torchcam.utils)
assert all(module is getattr(torchcam, name) for name, module in zip(names, modules))
assert all(module is sys.modules[f"torchcam.{{name}}"] for name, module in zip(names, modules))
assert version is torchcam.version
assert "matplotlib.pyplot" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


@pytest.mark.parametrize("module_name", _PUBLIC_MODULES)
def test_lazy_module_is_cached(module_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(torchcam, module_name, raising=False)
    module = getattr(torchcam, module_name)
    assert module is getattr(torchcam, module_name)
    assert module is sys.modules[f"torchcam.{module_name}"]


def test_package_version() -> None:
    assert torchcam.__version__ == importlib.metadata.version("torchcam")
    assert torchcam.version is importlib.metadata.version


def test_unknown_attribute() -> None:
    with pytest.raises(AttributeError, match="module 'torchcam' has no attribute 'missing'"):
        _ = torchcam.missing  # type: ignore[attr-defined]
