"""Image validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = None

SUPPORTED_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".webp", ".bmp"}
)


@dataclass(frozen=True)
class ImageInspection:
    width: int
    height: int
    format: str
    mode: str


def is_supported_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def inspect_image(path: str | Path) -> ImageInspection:
    path = Path(path)
    if not is_supported_image_path(path):
        raise ValueError(f"Unsupported image extension: {path.suffix or '<none>'}")

    try:
        with Image.open(path) as image:
            image.seek(0)
            width, height = image.size
            image_format = str(image.format or path.suffix.lstrip(".")).upper()
            mode = str(image.mode or "")
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(f"Unable to decode image {path}: {exc}") from exc

    if width < 1 or height < 1:
        raise ValueError(f"Invalid image dimensions for {path}: {width}x{height}")
    return ImageInspection(width=width, height=height, format=image_format, mode=mode)
