import importlib.metadata
import subprocess  # noqa: S404
import sys

import pytest

import torchcam

_PUBLIC_MODULES = ("explain", "methods", "metrics", "utils")


def _run_in_fresh_process(statement: str) -> None:
    script = f"""
import sys
import torchcam

module_names = {(_PUBLIC_MODULES)!r}
assert "torch" not in sys.modules
assert "matplotlib.pyplot" not in sys.modules
assert all(f"torchcam.{{name}}" not in sys.modules for name in module_names)
assert all(name not in vars(torchcam) for name in module_names)
assert torchcam.__version__

{statement}
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


def test_bare_import_is_lazy() -> None:
    _run_in_fresh_process("")


@pytest.mark.parametrize("module_name", _PUBLIC_MODULES)
def test_lazy_module_in_fresh_process(module_name: str) -> None:
    _run_in_fresh_process(
        f"""
module = getattr(torchcam, {module_name!r})
assert module.__name__ == "torchcam.{module_name}"
assert module is getattr(torchcam, {module_name!r})
assert module is sys.modules["torchcam.{module_name}"]
"""
    )


def test_from_import_syntax() -> None:
    _run_in_fresh_process(
        """
from torchcam import explain, methods, metrics, utils

assert explain is sys.modules["torchcam.explain"]
assert methods is sys.modules["torchcam.methods"]
assert metrics is sys.modules["torchcam.metrics"]
assert utils is sys.modules["torchcam.utils"]
"""
    )


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
