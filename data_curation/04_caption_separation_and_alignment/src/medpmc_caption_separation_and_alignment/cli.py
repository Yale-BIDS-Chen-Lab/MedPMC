"""Command-line interface for Caption Separation and Alignment."""

from __future__ import annotations

import argparse

from .alignment import (
    DEFAULT_IMAGE_LOADER_WORKERS,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_MAX_IMAGES_PER_BATCH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PARENTS_PER_SHARD,
    DEFAULT_RETRY_MAX_NEW_TOKENS,
    DEFAULT_SESSION_LEN,
    DEFAULT_VISION_MAX_BATCH_SIZE,
    align_directory,
)
from .recovery import merge_retry_runs, retry_directory


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parent-dir", required=True)
    parser.add_argument("--panel-dir", required=True)
    parser.add_argument("--parent-image-root", required=True)
    parser.add_argument("--panel-image-root", required=True)
    parser.add_argument("--output-dir", required=True)


def _add_inference_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["lmdeploy"], default="lmdeploy")
    parser.add_argument("--engine", choices=["turbomind", "pytorch"], default="turbomind")
    parser.add_argument(
        "--chat-template",
        default="auto",
        help="LMDeploy chat template; auto delegates detection to the model/runtime",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["auto", "explicit_tokens", "ordered_images"],
        default="auto",
    )
    parser.add_argument("--session-len", type=int, default=DEFAULT_SESSION_LEN)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=default_max_new_tokens,
        help=(
            "Maximum generated tokens for this pass. This is a user-controlled "
            "limit, not a guarantee that every caption fits."
        ),
    )
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--repetition-ngram-size", type=int, default=20)
    parser.add_argument("--repetition-ngram-threshold", type=int, default=3)
    parser.add_argument("--tp", type=int, default=1, help="Tensor-parallel GPU count")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=DEFAULT_INFERENCE_BATCH_SIZE,
    )
    parser.add_argument(
        "--max-images-per-batch",
        type=int,
        default=DEFAULT_MAX_IMAGES_PER_BATCH,
    )
    parser.add_argument("--engine-max-batch-size", type=int, default=None)
    parser.add_argument(
        "--vision-max-batch-size",
        type=int,
        default=DEFAULT_VISION_MAX_BATCH_SIZE,
    )
    parser.add_argument(
        "--cache-max-entry-count",
        type=float,
        default=0.8,
        help="LMDeploy KV-cache memory fraction; lower this for memory-heavy retry passes",
    )
    parser.add_argument(
        "--image-loader-workers",
        type=int,
        default=DEFAULT_IMAGE_LOADER_WORKERS,
    )
    parser.add_argument(
        "--batch-by-panel-count",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--parents-per-shard", type=int, default=DEFAULT_PARENTS_PER_SHARD)
    parser.add_argument("--min-panel-count", type=int, default=1)
    parser.add_argument("--max-panel-count", type=int, default=None)
    parser.add_argument("--max-parents", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")


def _alignment_kwargs(args: argparse.Namespace) -> dict:
    return {
        "parent_image_root": args.parent_image_root,
        "panel_image_root": args.panel_image_root,
        "model_name": args.model,
        "backend_name": args.backend,
        "engine": args.engine,
        "chat_template": args.chat_template,
        "prompt_style": args.prompt_style,
        "session_len": args.session_len,
        "max_new_tokens": args.max_new_tokens,
        "tensor_parallel": args.tp,
        "inference_batch_size": args.inference_batch_size,
        "max_images_per_batch": args.max_images_per_batch,
        "engine_max_batch_size": args.engine_max_batch_size,
        "vision_max_batch_size": args.vision_max_batch_size,
        "cache_max_entry_count": args.cache_max_entry_count,
        "image_loader_workers": args.image_loader_workers,
        "batch_by_panel_count": args.batch_by_panel_count,
        "trust_remote_code": args.trust_remote_code,
        "repetition_penalty": args.repetition_penalty,
        "repetition_ngram_size": args.repetition_ngram_size,
        "repetition_ngram_threshold": args.repetition_ngram_threshold,
        "parents_per_shard": args.parents_per_shard,
        "min_panel_count": args.min_panel_count,
        "max_panel_count": args.max_panel_count,
        "max_parents": args.max_parents,
        "resume": args.resume,
        "force": args.force,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medpmc-caption-separation-and-alignment",
        description="Separate compound captions and align them with ordered Stage 3 subfigures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        aliases=["align"],
        help="Run a first-pass caption separation and alignment job",
    )
    run.set_defaults(action="run")
    _add_data_arguments(run)
    _add_inference_arguments(run)
    run.add_argument(
        "--retry-truncated",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optionally retry truncated predictions within the same run. "
            "Use the separate retry command for independent output and resource settings."
        ),
    )
    run.add_argument(
        "--retry-max-new-tokens",
        type=int,
        default=DEFAULT_RETRY_MAX_NEW_TOKENS,
    )

    retry = subparsers.add_parser(
        "retry",
        help="Re-run selected failed parents from a previous run in a new process",
    )
    retry.set_defaults(action="retry")
    retry.add_argument("--source-run-dir", required=True)
    retry.add_argument(
        "--status",
        action="append",
        default=None,
        help=(
            "Source status to retry; may be repeated. Default: generation_truncated"
        ),
    )
    _add_data_arguments(retry)
    _add_inference_arguments(
        retry,
        default_max_new_tokens=DEFAULT_RETRY_MAX_NEW_TOKENS,
    )

    merge = subparsers.add_parser(
        "merge",
        help="Merge aligned retry results into a first-pass run",
    )
    merge.set_defaults(action="merge")
    merge.add_argument("--base-run-dir", required=True)
    merge.add_argument(
        "--retry-run-dir",
        action="append",
        required=True,
        help="Retry directory; repeat in increasing retry-budget order",
    )
    merge.add_argument("--output-dir", required=True)
    merge.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    action = getattr(args, "action", None)

    if action == "run":
        kwargs = _alignment_kwargs(args)
        kwargs.update(
            retry_truncated=args.retry_truncated,
            retry_max_new_tokens=args.retry_max_new_tokens,
        )
        summary = align_directory(
            args.parent_dir,
            args.panel_dir,
            args.output_dir,
            **kwargs,
        )
    elif action == "retry":
        summary = retry_directory(
            args.source_run_dir,
            args.parent_dir,
            args.panel_dir,
            args.output_dir,
            statuses=args.status or ["generation_truncated"],
            **_alignment_kwargs(args),
        )
    elif action == "merge":
        summary = merge_retry_runs(
            args.base_run_dir,
            args.retry_run_dir,
            args.output_dir,
            force=args.force,
        )
    else:  # pragma: no cover
        raise RuntimeError(f"Unknown command: {args.command}")

    print(f"Summary: {summary}")
    return 0
