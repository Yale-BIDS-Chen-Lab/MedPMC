"""CLI for MedPMC Multi-panel Figure Detection."""

from __future__ import annotations

import argparse

from .aws import DEFAULT_BUCKET
from .manifest import prepare_images
from .model import DEFAULT_CHECKPOINT_FILENAME, DEFAULT_MODEL, detect_directory
from .pipeline import run_pipeline


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", required=True, help="Stage 1 results/retained directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-figures", type=int)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--force", action="store_true")


def _add_detect_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--checkpoint-filename", default=DEFAULT_CHECKPOINT_FILENAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loader-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override the checkpoint input size; by default it is detected automatically.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--amp", action="store_true", help="Use CUDA float16 autocast")
    parser.add_argument("--force", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medpmc-multi-panel-figure-detection",
        description="Multi-panel Figure Detection stage of the MedPMC pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare-images",
        help="Resolve and download retained PMC figure images, then create manifests.",
    )
    _add_prepare_arguments(prepare_parser)

    detect_parser = subparsers.add_parser(
        "detect",
        help="Run the released ViT multi-panel detector over prepared images.",
    )
    _add_detect_arguments(detect_parser)

    run_parser = subparsers.add_parser(
        "run",
        help="Prepare retained images and run multi-panel detection end to end.",
    )
    _add_prepare_arguments(run_parser)
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument("--checkpoint-filename", default=DEFAULT_CHECKPOINT_FILENAME)
    run_parser.add_argument("--batch-size", type=int, default=256)
    run_parser.add_argument("--loader-workers", type=int, default=4)
    run_parser.add_argument("--threshold", type=float, default=0.5)
    run_parser.add_argument("--positive-label", type=int, default=1)
    run_parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override the checkpoint input size; by default it is detected automatically.",
    )
    run_parser.add_argument("--device", default="auto")
    run_parser.add_argument("--amp", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "prepare-images":
        summary = prepare_images(
            args.input_dir,
            args.output_dir,
            workers=args.workers,
            max_figures=args.max_figures,
            bucket=args.bucket,
            region_name=args.region,
            force=args.force,
        )
    elif args.command == "detect":
        summary = detect_directory(
            args.manifest_dir,
            args.output_dir,
            image_root=args.image_root,
            model_name=args.model,
            checkpoint_filename=args.checkpoint_filename,
            batch_size=args.batch_size,
            loader_workers=args.loader_workers,
            threshold=args.threshold,
            positive_label=args.positive_label,
            image_size=args.image_size,
            device=args.device,
            amp=args.amp,
            force=args.force,
        )
    elif args.command == "run":
        summary = run_pipeline(
            args.input_dir,
            args.output_dir,
            workers=args.workers,
            max_figures=args.max_figures,
            bucket=args.bucket,
            region_name=args.region,
            model_name=args.model,
            checkpoint_filename=args.checkpoint_filename,
            batch_size=args.batch_size,
            loader_workers=args.loader_workers,
            threshold=args.threshold,
            positive_label=args.positive_label,
            image_size=args.image_size,
            device=args.device,
            amp=args.amp,
            force=args.force,
        )
    else:
        raise AssertionError(f"Unhandled command: {args.command}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
