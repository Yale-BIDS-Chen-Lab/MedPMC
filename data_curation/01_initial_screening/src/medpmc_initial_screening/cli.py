"""Command-line interface for MedPMC Initial Screening."""

from __future__ import annotations

import argparse
from pathlib import Path

from .download import download_bulk_assets, download_pmcids
from .extract import extract_from_tar, extract_from_xml_dir
from .licenses import DEFAULT_ALLOWED_LICENSES, parse_allowed_licenses
from .pipeline import prepare_bulk, prepare_pmcids, run_bulk, run_pmcids
from .screen import (
    DEFAULT_MODEL,
    DEFAULT_REFERENCE_MODE,
    REFERENCE_MODES,
    screen_directory,
)

DEFAULT_LICENSE_TEXT = ",".join(sorted(DEFAULT_ALLOWED_LICENSES))


def _add_license_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allowed-licenses",
        default=DEFAULT_LICENSE_TEXT,
        help=(
            "Comma-separated article-level license allowlist. "
            f"Default: {DEFAULT_LICENSE_TEXT}"
        ),
    )


def _add_screen_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, or mps",
    )
    parser.add_argument(
        "--reference-mode",
        choices=REFERENCE_MODES,
        default=DEFAULT_REFERENCE_MODE,
        help=(
            "Reference representation used by the classifier. "
            "Default: model-compatible (recursive paragraph extraction)."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medpmc-initial-screening",
        description="Initial Screening stage of the MedPMC curation pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_bulk_parser = subparsers.add_parser(
        "run-bulk",
        help="Download a bulk XML package, extract figure text, and screen it.",
    )
    run_bulk_parser.add_argument("--archive-url", required=True)
    run_bulk_parser.add_argument("--filelist-url", required=True)
    run_bulk_parser.add_argument("--output-dir", required=True)
    run_bulk_parser.add_argument("--shard-size", type=int, default=10_000)
    run_bulk_parser.add_argument("--max-articles", type=int)
    run_bulk_parser.add_argument("--force", action="store_true")
    _add_license_argument(run_bulk_parser)
    _add_screen_arguments(run_bulk_parser)

    run_ids_parser = subparsers.add_parser(
        "run-pmcids",
        help="Download selected PMC XML files from NLM S3 and screen them.",
    )
    run_ids_parser.add_argument("--pmcid-file", required=True)
    run_ids_parser.add_argument("--output-dir", required=True)
    run_ids_parser.add_argument("--workers", type=int, default=8)
    run_ids_parser.add_argument("--shard-size", type=int, default=10_000)
    run_ids_parser.add_argument("--max-articles", type=int)
    run_ids_parser.add_argument("--force", action="store_true")
    _add_license_argument(run_ids_parser)
    _add_screen_arguments(run_ids_parser)

    prepare_bulk_parser = subparsers.add_parser(
        "prepare-bulk",
        help="Download a bulk XML package and extract figure text without screening.",
    )
    prepare_bulk_parser.add_argument("--archive-url", required=True)
    prepare_bulk_parser.add_argument("--filelist-url", required=True)
    prepare_bulk_parser.add_argument("--output-dir", required=True)
    prepare_bulk_parser.add_argument("--shard-size", type=int, default=10_000)
    prepare_bulk_parser.add_argument("--max-articles", type=int)
    prepare_bulk_parser.add_argument("--force", action="store_true")
    _add_license_argument(prepare_bulk_parser)

    prepare_ids_parser = subparsers.add_parser(
        "prepare-pmcids",
        help="Download selected PMC XML files and extract figure text without screening.",
    )
    prepare_ids_parser.add_argument("--pmcid-file", required=True)
    prepare_ids_parser.add_argument("--output-dir", required=True)
    prepare_ids_parser.add_argument("--workers", type=int, default=8)
    prepare_ids_parser.add_argument("--shard-size", type=int, default=10_000)
    prepare_ids_parser.add_argument("--max-articles", type=int)
    prepare_ids_parser.add_argument("--force", action="store_true")
    _add_license_argument(prepare_ids_parser)

    download_bulk_parser = subparsers.add_parser(
        "download-bulk",
        help="Download a bulk XML archive and its file list only.",
    )
    download_bulk_parser.add_argument("--archive-url", required=True)
    download_bulk_parser.add_argument("--filelist-url", required=True)
    download_bulk_parser.add_argument("--output-dir", required=True)
    download_bulk_parser.add_argument("--force", action="store_true")

    download_ids_parser = subparsers.add_parser(
        "download-pmcids",
        help="Download XML and metadata for selected PMC IDs only.",
    )
    download_ids_parser.add_argument("--pmcid-file", required=True)
    download_ids_parser.add_argument("--output-dir", required=True)
    download_ids_parser.add_argument("--workers", type=int, default=8)
    download_ids_parser.add_argument("--force", action="store_true")
    _add_license_argument(download_ids_parser)

    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract figure captions and inline reference text from PMC XML.",
    )
    source = extract_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--xml-tar")
    source.add_argument("--xml-dir")
    extract_parser.add_argument(
        "--filelist",
        help="Required with --xml-tar for article-level license metadata.",
    )
    extract_parser.add_argument("--metadata-dir")
    extract_parser.add_argument("--output-dir", required=True)
    extract_parser.add_argument("--shard-size", type=int, default=10_000)
    extract_parser.add_argument("--max-articles", type=int)
    extract_parser.add_argument("--force", action="store_true")
    _add_license_argument(extract_parser)

    screen_parser = subparsers.add_parser(
        "screen",
        help="Run the released PubMedBERT Initial Screening model.",
    )
    screen_parser.add_argument("--input-dir", required=True)
    screen_parser.add_argument("--output-dir", required=True)
    screen_parser.add_argument("--force", action="store_true")
    _add_screen_arguments(screen_parser)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    allowed = parse_allowed_licenses(
        getattr(args, "allowed_licenses", DEFAULT_LICENSE_TEXT)
    )

    if args.command == "run-bulk":
        run_bulk(
            args.archive_url,
            args.filelist_url,
            args.output_dir,
            allowed_licenses=allowed,
            shard_size=args.shard_size,
            max_articles=args.max_articles,
            model_name=args.model,
            batch_size=args.batch_size,
            threshold=args.threshold,
            max_length=args.max_length,
            positive_label=args.positive_label,
            device=args.device,
            reference_mode=args.reference_mode,
            force=args.force,
        )
    elif args.command == "run-pmcids":
        run_pmcids(
            args.pmcid_file,
            args.output_dir,
            allowed_licenses=allowed,
            workers=args.workers,
            shard_size=args.shard_size,
            max_articles=args.max_articles,
            model_name=args.model,
            batch_size=args.batch_size,
            threshold=args.threshold,
            max_length=args.max_length,
            positive_label=args.positive_label,
            device=args.device,
            reference_mode=args.reference_mode,
            force=args.force,
        )
    elif args.command == "prepare-bulk":
        archive, filelist, summary = prepare_bulk(
            args.archive_url,
            args.filelist_url,
            args.output_dir,
            allowed_licenses=allowed,
            shard_size=args.shard_size,
            max_articles=args.max_articles,
            force=args.force,
        )
        print(f"Archive: {archive}")
        print(f"File list: {filelist}")
        print(f"Summary: {summary}")
    elif args.command == "prepare-pmcids":
        manifest, summary = prepare_pmcids(
            args.pmcid_file,
            args.output_dir,
            allowed_licenses=allowed,
            workers=args.workers,
            shard_size=args.shard_size,
            max_articles=args.max_articles,
            force=args.force,
        )
        print(f"Manifest: {manifest}")
        print(f"Summary: {summary}")
    elif args.command == "download-bulk":
        archive, filelist = download_bulk_assets(
            args.archive_url,
            args.filelist_url,
            args.output_dir,
            overwrite=args.force,
        )
        print(f"Archive: {archive}")
        print(f"File list: {filelist}")
    elif args.command == "download-pmcids":
        manifest = download_pmcids(
            args.pmcid_file,
            args.output_dir,
            allowed_licenses=allowed,
            workers=args.workers,
            overwrite=args.force,
        )
        print(f"Manifest: {manifest}")
    elif args.command == "extract":
        if args.xml_tar:
            if not args.filelist:
                parser.error("--filelist is required with --xml-tar")
            summary = extract_from_tar(
                args.xml_tar,
                args.filelist,
                args.output_dir,
                allowed_licenses=allowed,
                shard_size=args.shard_size,
                max_articles=args.max_articles,
                force=args.force,
            )
        else:
            summary = extract_from_xml_dir(
                args.xml_dir,
                args.output_dir,
                metadata_dir=args.metadata_dir,
                allowed_licenses=allowed,
                shard_size=args.shard_size,
                max_articles=args.max_articles,
                force=args.force,
            )
        print(f"Summary: {summary}")
    elif args.command == "screen":
        summary = screen_directory(
            args.input_dir,
            args.output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            threshold=args.threshold,
            max_length=args.max_length,
            positive_label=args.positive_label,
            device=args.device,
            reference_mode=args.reference_mode,
            force=args.force,
        )
        print(f"Summary: {summary}")
    else:
        raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
