from contextlib import suppress
from torchcam import methods, metrics, utils

with suppress(ImportError):
    from .version import __version__
