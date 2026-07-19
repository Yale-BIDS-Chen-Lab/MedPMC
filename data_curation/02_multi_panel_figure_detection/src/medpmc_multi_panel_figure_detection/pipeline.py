"""End-to-end Stage 2 pipeline."""

from __future__ import annotations

from pathlib import Path

from .aws import DEFAULT_BUCKET
from .manifest import prepare_images
from .model import DEFAULT_CHECKPOINT_FILENAME, DEFAULT_MODEL, detect_directory


def run_pipeline(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 16,
    max_figures: int | None = None,
    bucket: str = DEFAULT_BUCKET,
    region_name: str = "us-east-1",
    model_name: str = DEFAULT_MODEL,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
    batch_size: int = 128,
    loader_workers: int = 4,
    threshold: float = 0.5,
    positive_label: int = 1,
    image_size: int | None = None,
    device: str = "auto",
    amp: bool = False,
    force: bool = False,
):
    output_dir = Path(output_dir)
    prepare_images(
        input_dir,
        output_dir,
        workers=workers,
        max_figures=max_figures,
        bucket=bucket,
        region_name=region_name,
        force=force,
    )
    return detect_directory(
        output_dir / "manifests" / "figures",
        output_dir / "results",
        image_root=output_dir,
        model_name=model_name,
        checkpoint_filename=checkpoint_filename,
        batch_size=batch_size,
        loader_workers=loader_workers,
        threshold=threshold,
        positive_label=positive_label,
        image_size=image_size,
        device=device,
        amp=amp,
        force=force,
    )
