"""CLI for MedPMC Medical Figure Classification."""

from __future__ import annotations

import argparse

from .classification import DEFAULT_CHECKPOINT_FILENAME, DEFAULT_MODEL, classify_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medpmc-medical-figure-classification",
        description="Medical Figure Classification stage of the MedPMC pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run",
        help="Classify Stage 2 single-panel figures and Stage 4 aligned subfigures.",
    )
    run.add_argument("--singlepanel-dir")
    run.add_argument("--singlepanel-image-root")
    run.add_argument("--subfigure-dir")
    run.add_argument("--subfigure-image-root")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--checkpoint-filename", default=DEFAULT_CHECKPOINT_FILENAME)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--loader-workers", type=int, default=4)
    run.add_argument("--threshold", type=float, default=0.5)
    run.add_argument("--medical-label", type=int, default=1)
    run.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Must match the checkpoint input size; omit to resolve it from timm.",
    )
    run.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    run.add_argument("--amp", action="store_true", help="Use CUDA float16 autocast")
    run.add_argument("--max-images", type=int)
    run.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "run":
        raise AssertionError(f"Unhandled command: {args.command}")
    summary = classify_sources(
        args.output_dir,
        singlepanel_dir=args.singlepanel_dir,
        singlepanel_image_root=args.singlepanel_image_root,
        subfigure_dir=args.subfigure_dir,
        subfigure_image_root=args.subfigure_image_root,
        model_name=args.model,
        checkpoint_filename=args.checkpoint_filename,
        batch_size=args.batch_size,
        loader_workers=args.loader_workers,
        threshold=args.threshold,
        medical_label=args.medical_label,
        image_size=args.image_size,
        device=args.device,
        amp=args.amp,
        max_images=args.max_images,
        force=args.force,
    )
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
