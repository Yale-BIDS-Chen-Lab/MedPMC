from pathlib import Path

from PIL import Image

from medpmc_multi_panel_figure_detection.images import inspect_image, is_supported_image_path


def test_inspect_image(tmp_path: Path):
    path = tmp_path / "figure.png"
    Image.new("RGB", (12, 8)).save(path)
    result = inspect_image(path)
    assert result.width == 12
    assert result.height == 8
    assert result.format == "PNG"


def test_supported_suffixes():
    assert is_supported_image_path("x.jpg")
    assert is_supported_image_path("x.TIFF")
    assert not is_supported_image_path("x.pdf")
