"""MedPMC Medical Figure Classification."""

from .classification import (
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_MODEL,
    classify_sources,
)

__all__ = [
    "DEFAULT_CHECKPOINT_FILENAME",
    "DEFAULT_MODEL",
    "classify_sources",
]

__version__ = "0.1.0"
