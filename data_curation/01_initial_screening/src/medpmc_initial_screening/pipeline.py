"""High-level Initial Screening workflows."""

from __future__ import annotations

from pathlib import Path

from .download import download_bulk_assets, download_pmcids
from .extract import extract_from_tar, extract_from_xml_dir
from .licenses import DEFAULT_ALLOWED_LICENSES
from .screen import DEFAULT_MODEL, DEFAULT_REFERENCE_MODE, screen_directory


def prepare_bulk(
    archive_url: str,
    filelist_url: str,
    output_dir: str | Path,
    *,
    allowed_licenses=DEFAULT_ALLOWED_LICENSES,
    shard_size: int = 10_000,
    max_articles: int | None = None,
    force: bool = False,
) -> tuple[Path, Path, Path]:
    """Download a PMC bulk package and extract figure-level text only."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    extracted_dir = output_dir / "intermediate" / "figure_text"

    archive_path, filelist_path = download_bulk_assets(
        archive_url,
        filelist_url,
        raw_dir,
        overwrite=force,
    )
    summary_path = extract_from_tar(
        archive_path,
        filelist_path,
        extracted_dir,
        allowed_licenses=allowed_licenses,
        shard_size=shard_size,
        max_articles=max_articles,
        force=force,
    )
    return archive_path, filelist_path, summary_path


def prepare_pmcids(
    pmcid_file: str | Path,
    output_dir: str | Path,
    *,
    allowed_licenses=DEFAULT_ALLOWED_LICENSES,
    workers: int = 8,
    shard_size: int = 10_000,
    max_articles: int | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    """Download selected PMC XML files and extract figure-level text only."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "raw"
    extracted_dir = output_dir / "intermediate" / "figure_text"

    manifest_path = download_pmcids(
        pmcid_file,
        raw_dir,
        allowed_licenses=allowed_licenses,
        workers=workers,
        overwrite=force,
    )
    summary_path = extract_from_xml_dir(
        raw_dir / "xml",
        extracted_dir,
        metadata_dir=raw_dir / "metadata",
        allowed_licenses=allowed_licenses,
        shard_size=shard_size,
        max_articles=max_articles,
        force=force,
    )
    return manifest_path, summary_path


def run_bulk(
    archive_url: str,
    filelist_url: str,
    output_dir: str | Path,
    *,
    allowed_licenses=DEFAULT_ALLOWED_LICENSES,
    shard_size: int = 10_000,
    max_articles: int | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    threshold: float = 0.5,
    max_length: int = 512,
    positive_label: int = 1,
    device: str = "auto",
    reference_mode: str = DEFAULT_REFERENCE_MODE,
    force: bool = False,
) -> None:
    """Run download, extraction, and screening for a PMC bulk package."""
    output_dir = Path(output_dir)
    prepare_bulk(
        archive_url,
        filelist_url,
        output_dir,
        allowed_licenses=allowed_licenses,
        shard_size=shard_size,
        max_articles=max_articles,
        force=force,
    )
    screen_directory(
        output_dir / "intermediate" / "figure_text",
        output_dir / "results",
        model_name=model_name,
        batch_size=batch_size,
        threshold=threshold,
        max_length=max_length,
        positive_label=positive_label,
        device=device,
        reference_mode=reference_mode,
        force=force,
    )


def run_pmcids(
    pmcid_file: str | Path,
    output_dir: str | Path,
    *,
    allowed_licenses=DEFAULT_ALLOWED_LICENSES,
    workers: int = 8,
    shard_size: int = 10_000,
    max_articles: int | None = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    threshold: float = 0.5,
    max_length: int = 512,
    positive_label: int = 1,
    device: str = "auto",
    reference_mode: str = DEFAULT_REFERENCE_MODE,
    force: bool = False,
) -> None:
    """Run download, extraction, and screening for selected PMC IDs."""
    output_dir = Path(output_dir)
    prepare_pmcids(
        pmcid_file,
        output_dir,
        allowed_licenses=allowed_licenses,
        workers=workers,
        shard_size=shard_size,
        max_articles=max_articles,
        force=force,
    )
    screen_directory(
        output_dir / "intermediate" / "figure_text",
        output_dir / "results",
        model_name=model_name,
        batch_size=batch_size,
        threshold=threshold,
        max_length=max_length,
        positive_label=positive_label,
        device=device,
        reference_mode=reference_mode,
        force=force,
    )
