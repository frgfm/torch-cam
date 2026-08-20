from importlib import import_module as _import_module
from importlib.metadata import version
from types import ModuleType as _ModuleType

__all__ = ("explain", "methods", "metrics", "utils", "version")  # noqa: F822

__version__ = version("torchcam")


def __getattr__(name: str) -> _ModuleType:
    if name in {"explain", "methods", "metrics", "utils"}:
        module = _import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
