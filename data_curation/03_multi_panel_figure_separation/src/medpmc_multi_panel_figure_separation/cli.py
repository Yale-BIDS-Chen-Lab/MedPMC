"""Command-line interface for Multi-panel Figure Separation."""

from __future__ import annotations

import argparse

from .separation import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIDENCE,
    DEFAULT_MODEL,
    separate_directory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medpmc-multi-panel-figure-separation",
        description="Separate Stage 2 multi-panel figures into panel crops with YOLOv10.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    separate = subparsers.add_parser("separate", help="Detect and crop panels")
    separate.add_argument("--classified-dir", required=True)
    separate.add_argument("--output-dir", required=True)
    separate.add_argument(
        "--image-root",
        default=None,
        help="Root used to resolve Stage 2 local_image_path; inferred from classified-dir when omitted",
    )
    separate.add_argument("--model", default=DEFAULT_MODEL)
    separate.add_argument("--checkpoint-filename", default=DEFAULT_CHECKPOINT)
    separate.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    separate.add_argument("--batch-size", type=int, default=1)
    separate.add_argument("--device", default="auto")
    separate.add_argument("--max-figures", type=int, default=None)
    separate.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "separate":
        summary = separate_directory(
            args.classified_dir,
            args.output_dir,
            image_root=args.image_root,
            model_name=args.model,
            checkpoint_filename=args.checkpoint_filename,
            confidence=args.conf,
            batch_size=args.batch_size,
            device=args.device,
            max_figures=args.max_figures,
            force=args.force,
        )
        print(f"Summary: {summary}")
        return 0
    raise RuntimeError(f"Unknown command: {args.command}")
