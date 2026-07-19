"""Caption separation and alignment with pluggable batched VLM backends."""

from __future__ import annotations

import inspect
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from .backends import (
    BatchInferenceBackend,
    InferenceRequest,
    InferenceResult,
    LMDeployBackend,
)
from .batching import plan_request_batches
from .prompting import (
    OUTPUT_DELIMITER,
    PROMPT_PREFIX,
    PROMPT_VERSION,
    build_ordered_images_prompt,
    build_prompt,
    clean_prediction,
    parse_subcaptions,
)
from .storage import read_dataset, write_json, write_jsonl, write_table

DEFAULT_MODEL = "Yale-BIDS-Chen/medpmc-caption-separation-internvl-2.5-4b-mpo"
DEFAULT_SESSION_LEN = 32768
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_RETRY_MAX_NEW_TOKENS = 2048
DEFAULT_PARENTS_PER_SHARD = 100
DEFAULT_INFERENCE_BATCH_SIZE = 4
DEFAULT_MAX_IMAGES_PER_BATCH = 40
DEFAULT_VISION_MAX_BATCH_SIZE = 16
DEFAULT_IMAGE_LOADER_WORKERS = 8
SUBFIGURE_ORDER_SOURCE = "historical_order_index"

# LMDeploy should infer the chat template from the model repository by default.
# Explicit template names are passed only when the user deliberately requests one.
# The released MedPMC InternVL repository uses Hugging Face custom config code.
# Do not globally trust arbitrary repositories; auto-enable only for checkpoints
# explicitly maintained by this pipeline.
DEFAULT_TRUST_REMOTE_CODE_MODELS = {DEFAULT_MODEL}


def resolve_chat_template(model_name: str, requested: str | None) -> str | None:
    """Return an explicit LMDeploy template only when requested.

    ``auto`` deliberately returns ``None`` so LMDeploy can infer the correct
    template from the model repository. This is required for LMDeploy 0.14,
    where forcing the unregistered ``internvl2_5`` name falls back to a generic
    template and can cause non-terminating/repetitive generation.
    """
    del model_name  # retained in the signature for API stability
    value = str(requested or "auto").strip()
    if value.lower() in {"", "auto", "none"}:
        return None
    return value





def resolve_trust_remote_code(model_name: str, requested: bool | None) -> bool:
    """Resolve remote-code trust without enabling it for arbitrary models."""
    if requested is None:
        return model_name in DEFAULT_TRUST_REMOTE_CODE_MODELS
    return bool(requested)

def resolve_prompt_style(model_name: str, requested: str | None) -> str:
    value = str(requested or "auto").strip().lower()
    if value == "auto":
        return "explicit_tokens" if model_name == DEFAULT_MODEL else "ordered_images"
    if value not in {"explicit_tokens", "ordered_images"}:
        raise ValueError("prompt_style must be auto, explicit_tokens, or ordered_images")
    return value

def _require_columns(names: list[str], required: set[str], label: str) -> None:
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"Missing required {label} columns: {missing}")


def _resolve_path(value: Any, root: Path) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def _parent_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("pmcid", pa.string()),
            ("article_version", pa.string()),
            ("resolved_article_version", pa.string()),
            ("pmid", pa.string()),
            ("figure_id", pa.string()),
            ("figure_label", pa.string()),
            ("parent_image_id", pa.string()),
            ("parent_local_image_path", pa.string()),
            ("main_caption", pa.string()),
            ("expected_subcaption_count", pa.int64()),
            ("predicted_subcaption_count", pa.int64()),
            ("raw_prediction", pa.string()),
            ("caption_alignment_status", pa.string()),
            ("caption_alignment_error", pa.string()),
            ("caption_model", pa.string()),
            ("caption_backend", pa.string()),
            ("caption_engine", pa.string()),
            ("caption_chat_template", pa.string()),
            ("caption_prompt_version", pa.string()),
            ("caption_prompt_style", pa.string()),
            ("caption_session_len", pa.int64()),
            ("caption_max_new_tokens", pa.int64()),
            ("caption_generation_attempts", pa.int64()),
            ("caption_retried_after_truncation", pa.bool_()),
            ("subfigure_order_source", pa.string()),
            ("inference_batch_size", pa.int64()),
            ("inference_batch_image_count", pa.int64()),
            ("inference_batch_seconds", pa.float64()),
            ("caption_finish_reason", pa.string()),
            ("caption_input_token_len", pa.int64()),
            ("caption_generate_token_len", pa.int64()),
            ("caption_initial_finish_reason", pa.string()),
            ("caption_initial_generate_token_len", pa.int64()),
            ("caption_run_kind", pa.string()),
            ("caption_source_run_dir", pa.string()),
        ]
    )


def _subfigure_schema(input_schema):
    import pyarrow as pa

    additions = [
        pa.field("main_caption", pa.string()),
        pa.field("subcaption", pa.string()),
        pa.field("caption_alignment_status", pa.string()),
        pa.field("caption_alignment_error", pa.string()),
        pa.field("caption_model", pa.string()),
        pa.field("caption_backend", pa.string()),
        pa.field("caption_engine", pa.string()),
        pa.field("caption_chat_template", pa.string()),
        pa.field("caption_prompt_version", pa.string()),
        pa.field("caption_prompt_style", pa.string()),
        pa.field("caption_session_len", pa.int64()),
        pa.field("caption_max_new_tokens", pa.int64()),
        pa.field("caption_generation_attempts", pa.int64()),
        pa.field("caption_retried_after_truncation", pa.bool_()),
        pa.field("subfigure_order_source", pa.string()),
        pa.field("caption_run_kind", pa.string()),
        pa.field("caption_source_run_dir", pa.string()),
    ]
    existing = set(input_schema.names)
    return pa.schema(list(input_schema) + [field for field in additions if field.name not in existing])


def _text(record: dict[str, Any], key: str) -> str:
    return str(record.get(key) or "")


def _group_panels(panel_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in panel_rows:
        if str(row.get("crop_status") or "") != "ready":
            continue
        grouped[str(row.get("parent_image_id") or "")].append(row)
    for parent_id, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                int(
                    row.get("historical_order_index")
                    if row.get("historical_order_index") is not None
                    else row.get("subfigure_index") or 0
                ),
                int(row.get("detector_index") or 0),
            )
        )
        indices = [int(row.get("subfigure_index") or 0) for row in rows]
        if indices != list(range(len(rows))):
            raise ValueError(f"Non-contiguous subfigure_index for {parent_id}: {indices[:20]}")
    return dict(grouped)


def _status_for_prediction(
    parsed: list[str],
    expected: int,
    *,
    finish_reason: str = "",
) -> tuple[str, str]:
    if str(finish_reason or "").lower() == "length":
        return "generation_truncated", "Generation reached the token limit"
    if not parsed:
        return "empty_prediction", "Model returned no subcaptions"
    if len(parsed) != expected:
        return "count_mismatch", f"Expected {expected} subcaptions but parsed {len(parsed)}"
    if any(not value for value in parsed):
        return "empty_subcaption", "One or more subcaptions are empty"
    return "aligned", ""


def _backend_label(backend: BatchInferenceBackend | None) -> tuple[str, str]:
    if backend is None:
        return "callable", "single_request"
    return str(getattr(backend, "name", "custom")), str(getattr(backend, "engine", "custom"))


def _parent_output_record(
    parent: dict[str, Any],
    *,
    main_caption: str,
    expected: int,
    parsed: list[str],
    raw_prediction: str,
    status: str,
    error: str,
    model_name: str,
    backend_name: str,
    engine_name: str,
    chat_template: str | None,
    prompt_style: str,
    session_len: int,
    max_new_tokens: int,
    batch_metadata: dict[str, Any] | None = None,
    run_kind: str = "first_pass",
    source_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    metadata = batch_metadata or {}
    return {
        "pmcid": _text(parent, "pmcid"),
        "article_version": _text(parent, "article_version"),
        "resolved_article_version": _text(parent, "resolved_article_version"),
        "pmid": _text(parent, "pmid"),
        "figure_id": _text(parent, "figure_id"),
        "figure_label": _text(parent, "figure_label"),
        "parent_image_id": _text(parent, "parent_image_id"),
        "parent_local_image_path": _text(parent, "parent_local_image_path"),
        "main_caption": main_caption,
        "expected_subcaption_count": int(expected),
        "predicted_subcaption_count": int(len(parsed)),
        "raw_prediction": str(raw_prediction or ""),
        "caption_alignment_status": status,
        "caption_alignment_error": error,
        "caption_model": model_name,
        "caption_backend": backend_name,
        "caption_engine": engine_name,
        "caption_chat_template": str(chat_template or "auto"),
        "caption_prompt_version": PROMPT_VERSION,
        "caption_prompt_style": prompt_style,
        "caption_session_len": int(session_len),
        "caption_max_new_tokens": int(metadata.get("max_new_tokens") or max_new_tokens),
        "caption_generation_attempts": int(metadata.get("generation_attempts") or 1),
        "caption_retried_after_truncation": bool(metadata.get("retried_after_truncation") or False),
        "subfigure_order_source": SUBFIGURE_ORDER_SOURCE,
        "inference_batch_size": int(metadata.get("batch_size") or 0),
        "inference_batch_image_count": int(metadata.get("batch_image_count") or 0),
        "inference_batch_seconds": float(metadata.get("batch_seconds") or 0.0),
        "caption_finish_reason": str(metadata.get("finish_reason") or ""),
        "caption_input_token_len": int(metadata.get("input_token_len") or 0),
        "caption_generate_token_len": int(metadata.get("generate_token_len") or 0),
        "caption_initial_finish_reason": str(metadata.get("initial_finish_reason") or ""),
        "caption_initial_generate_token_len": int(metadata.get("initial_generate_token_len") or 0),
        "caption_run_kind": str(run_kind),
        "caption_source_run_dir": str(source_run_dir or ""),
    }


def _subfigure_output_records(
    panel_rows: list[dict[str, Any]],
    *,
    main_caption: str,
    parsed: list[str],
    status: str,
    error: str,
    model_name: str,
    backend_name: str,
    engine_name: str,
    chat_template: str | None,
    prompt_style: str,
    session_len: int,
    max_new_tokens: int,
    batch_metadata: dict[str, Any] | None = None,
    run_kind: str = "first_pass",
    source_run_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    metadata = batch_metadata or {}
    exact = len(parsed) == len(panel_rows)
    outputs: list[dict[str, Any]] = []
    for index, panel in enumerate(panel_rows):
        outputs.append(
            {
                **panel,
                "main_caption": main_caption,
                "subcaption": parsed[index] if exact else "",
                "caption_alignment_status": status,
                "caption_alignment_error": error,
                "caption_model": model_name,
                "caption_backend": backend_name,
                "caption_engine": engine_name,
                "caption_chat_template": str(chat_template or "auto"),
                "caption_prompt_version": PROMPT_VERSION,
                "caption_prompt_style": prompt_style,
                "caption_session_len": int(session_len),
                "caption_max_new_tokens": int(metadata.get("max_new_tokens") or max_new_tokens),
                "caption_generation_attempts": int(metadata.get("generation_attempts") or 1),
                "caption_retried_after_truncation": bool(metadata.get("retried_after_truncation") or False),
                "subfigure_order_source": SUBFIGURE_ORDER_SOURCE,
                "caption_run_kind": str(run_kind),
                "caption_source_run_dir": str(source_run_dir or ""),
            }
        )
    return outputs


def _backend_generate_batch(
    backend: BatchInferenceBackend,
    requests: list[InferenceRequest],
    *,
    max_new_tokens: int | None = None,
) -> list[InferenceResult]:
    """Call a backend while preserving compatibility with older custom adapters."""
    method = backend.generate_batch
    parameters = inspect.signature(method).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if max_new_tokens is not None and (
        accepts_kwargs or "max_new_tokens" in parameters
    ):
        return method(requests, max_new_tokens=max_new_tokens)
    return method(requests)


def _run_batch_with_isolation(
    backend: BatchInferenceBackend,
    requests: list[InferenceRequest],
    *,
    max_new_tokens: int | None = None,
) -> list[InferenceResult]:
    """Recursively split failed batches so one bad sample does not lose a shard."""
    try:
        results = _backend_generate_batch(
            backend, requests, max_new_tokens=max_new_tokens
        )
        by_id = {result.request_id: result for result in results}
        missing = [request.request_id for request in requests if request.request_id not in by_id]
        if missing:
            raise RuntimeError(f"Backend omitted request IDs: {missing[:5]}")
        return [by_id[request.request_id] for request in requests]
    except Exception as exc:
        if len(requests) == 1:
            return [InferenceResult(request_id=requests[0].request_id, error=str(exc))]
        midpoint = len(requests) // 2
        return _run_batch_with_isolation(
            backend, requests[:midpoint], max_new_tokens=max_new_tokens
        ) + _run_batch_with_isolation(
            backend, requests[midpoint:], max_new_tokens=max_new_tokens
        )


class _CallableBackend:
    """Compatibility adapter for tests and custom single-request callables."""

    name = "callable"
    engine = "single_request"

    def __init__(self, runner: Callable[[str, list[Path]], str]) -> None:
        self.runner = runner

    def generate_batch(
        self,
        requests: list[InferenceRequest],
        *,
        max_new_tokens: int | None = None,
    ) -> list[InferenceResult]:
        outputs = []
        for request in requests:
            started = time.perf_counter()
            try:
                text = self.runner(request.prompt, list(request.image_paths))
                outputs.append(
                    InferenceResult(
                        request_id=request.request_id,
                        text=str(text or ""),
                        metadata={
                            "batch_size": 1,
                            "batch_image_count": request.image_count,
                            "batch_seconds": time.perf_counter() - started,
                            "max_new_tokens": int(max_new_tokens or 0),
                            "generation_attempts": 1,
                        },
                    )
                )
            except Exception as exc:
                outputs.append(InferenceResult(request_id=request.request_id, error=str(exc)))
        return outputs

    def close(self) -> None:
        return None


def align_directory(
    parent_dir: str | Path,
    panel_dir: str | Path,
    output_dir: str | Path,
    *,
    parent_image_root: str | Path,
    panel_image_root: str | Path,
    model_name: str = DEFAULT_MODEL,
    backend_name: str = "lmdeploy",
    engine: str = "turbomind",
    chat_template: str | None = "auto",
    prompt_style: str = "auto",
    session_len: int = DEFAULT_SESSION_LEN,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    retry_truncated: bool = False,
    retry_max_new_tokens: int = DEFAULT_RETRY_MAX_NEW_TOKENS,
    tensor_parallel: int = 1,
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    max_images_per_batch: int | None = DEFAULT_MAX_IMAGES_PER_BATCH,
    engine_max_batch_size: int | None = None,
    vision_max_batch_size: int | None = DEFAULT_VISION_MAX_BATCH_SIZE,
    cache_max_entry_count: float = 0.8,
    image_loader_workers: int = DEFAULT_IMAGE_LOADER_WORKERS,
    batch_by_panel_count: bool = True,
    trust_remote_code: bool | None = None,
    repetition_penalty: float = 1.0,
    repetition_ngram_size: int = 20,
    repetition_ngram_threshold: int = 3,
    parents_per_shard: int = DEFAULT_PARENTS_PER_SHARD,
    min_panel_count: int = 1,
    max_panel_count: int | None = None,
    max_parents: int | None = None,
    resume: bool = False,
    force: bool = False,
    include_parent_ids: set[str] | None = None,
    run_kind: str = "first_pass",
    source_run_dir: str | Path | None = None,
    backend: BatchInferenceBackend | None = None,
    runner: Callable[[str, list[Path]], str] | None = None,
) -> Path:
    """Separate compound captions and align one subcaption to each ordered panel."""
    pipeline_started = time.perf_counter()
    if session_len < 1 or max_new_tokens < 1 or retry_max_new_tokens < 1:
        raise ValueError("session_len and generation token limits must be positive")
    if retry_truncated and retry_max_new_tokens <= max_new_tokens:
        raise ValueError("retry_max_new_tokens must exceed max_new_tokens when retries are enabled")
    if repetition_penalty <= 0 or repetition_ngram_size < 0 or repetition_ngram_threshold < 0:
        raise ValueError("repetition settings must be non-negative and penalty must be positive")
    if tensor_parallel < 1 or parents_per_shard < 1 or inference_batch_size < 1:
        raise ValueError("parallelism, batch size, and shard size must be positive")
    if max_images_per_batch is not None and max_images_per_batch < 1:
        raise ValueError("max_images_per_batch must be positive")
    if min_panel_count < 1:
        raise ValueError("min_panel_count must be positive")
    if max_panel_count is not None and max_panel_count < min_panel_count:
        raise ValueError("max_panel_count must be at least min_panel_count")
    if max_parents is not None and max_parents < 1:
        raise ValueError("max_parents must be positive")
    if resume and force:
        raise ValueError("--resume and --force cannot be used together")
    if backend is not None and runner is not None:
        raise ValueError("Provide either backend or runner, not both")
    if run_kind not in {"first_pass", "retry"}:
        raise ValueError("run_kind must be 'first_pass' or 'retry'")

    parent_dir = Path(parent_dir)
    panel_dir = Path(panel_dir)
    output_dir = Path(output_dir)
    parent_image_root = Path(parent_image_root)
    panel_image_root = Path(panel_image_root)

    parent_table = read_dataset(parent_dir)
    panel_table = read_dataset(panel_dir)
    _require_columns(
        parent_table.schema.names,
        {"parent_image_id", "parent_local_image_path", "separation_status"},
        "Stage 3 parent",
    )
    _require_columns(
        panel_table.schema.names,
        {
            "parent_image_id",
            "subfigure_index",
            "historical_order_index",
            "local_panel_path",
            "crop_status",
            "caption",
        },
        "Stage 3 panel",
    )

    parent_rows = parent_table.to_pylist()
    panel_rows = panel_table.to_pylist()
    grouped_panels = _group_panels(panel_rows)
    eligible = [
        row
        for row in parent_rows
        if str(row.get("separation_status") or "") == "separated"
        and str(row.get("parent_image_id") or "") in grouped_panels
        and len(grouped_panels[str(row.get("parent_image_id") or "")]) >= min_panel_count
        and (
            max_panel_count is None
            or len(grouped_panels[str(row.get("parent_image_id") or "")]) <= max_panel_count
        )
    ]
    if include_parent_ids is not None:
        requested_ids = {str(value) for value in include_parent_ids}
        eligible = [
            row
            for row in eligible
            if str(row.get("parent_image_id") or "") in requested_ids
        ]
    eligible.sort(key=lambda row: str(row.get("parent_image_id") or ""))
    if max_parents is not None:
        eligible = eligible[:max_parents]

    parent_output_dir = output_dir / "manifests" / "parents"
    subfigure_output_dir = output_dir / "manifests" / "subfigures"
    raw_output_dir = output_dir / "raw_predictions"
    summary_path = output_dir / "caption_alignment_summary.json"
    parent_output_dir.mkdir(parents=True, exist_ok=True)
    subfigure_output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    chunks = [eligible[index : index + parents_per_shard] for index in range(0, len(eligible), parents_per_shard)]
    pending = []
    completed_shards: list[tuple[Path, Path]] = []
    for shard_index, chunk in enumerate(chunks):
        parent_path = parent_output_dir / f"part-{shard_index:06d}.parquet"
        subfigure_path = subfigure_output_dir / f"part-{shard_index:06d}.parquet"
        raw_path = raw_output_dir / f"part-{shard_index:06d}.jsonl"
        exists = [parent_path.exists(), subfigure_path.exists(), raw_path.exists()]
        if any(exists) and not all(exists):
            raise RuntimeError(f"Incomplete output shard {shard_index}; remove it or rerun with --force")
        if all(exists):
            if force:
                for path in (parent_path, subfigure_path, raw_path):
                    path.unlink()
            elif resume:
                completed_shards.append((parent_path, subfigure_path))
                continue
            else:
                raise FileExistsError(f"Output shard already exists: {parent_path}. Use --resume or --force.")
        pending.append((shard_index, chunk, parent_path, subfigure_path, raw_path))

    resolved_chat_template = resolve_chat_template(model_name, chat_template)
    resolved_prompt_style = resolve_prompt_style(model_name, prompt_style)
    resolved_trust_remote_code = resolve_trust_remote_code(
        model_name, trust_remote_code
    )
    owns_backend = False
    backend_initialization_started = time.perf_counter()
    if pending and backend is None:
        if runner is not None:
            backend = _CallableBackend(runner)
            owns_backend = True
        elif backend_name == "lmdeploy":
            backend = LMDeployBackend(
                model_name,
                engine=engine,
                session_len=session_len,
                max_new_tokens=max_new_tokens,
                tensor_parallel=tensor_parallel,
                chat_template=resolved_chat_template,
                engine_max_batch_size=engine_max_batch_size,
                vision_max_batch_size=vision_max_batch_size,
                cache_max_entry_count=cache_max_entry_count,
                image_loader_workers=image_loader_workers,
                trust_remote_code=resolved_trust_remote_code,
                repetition_penalty=repetition_penalty,
                repetition_ngram_size=repetition_ngram_size,
                repetition_ngram_threshold=repetition_ngram_threshold,
            )
            owns_backend = True
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
    backend_initialization_seconds = time.perf_counter() - backend_initialization_started

    actual_backend_name, actual_engine_name = _backend_label(backend)
    status_counts: Counter[str] = Counter()
    existing_processed = 0
    existing_aligned = 0
    existing_aligned_subfigures = 0
    if completed_shards:
        import pyarrow.parquet as pq

        for completed_parent_path, completed_subfigure_path in completed_shards:
            completed_parents = pq.read_table(completed_parent_path).to_pylist()
            completed_subfigures = pq.read_table(completed_subfigure_path).to_pylist()
            existing_processed += len(completed_parents)
            for row in completed_parents:
                status = str(row.get("caption_alignment_status") or "")
                status_counts[status] += 1
                if status == "aligned":
                    existing_aligned += 1
            existing_aligned_subfigures += sum(
                str(row.get("caption_alignment_status") or "") == "aligned"
                for row in completed_subfigures
            )

    cuda_available = False
    cuda_device_count = 0
    cuda_device_names: list[str] = []
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        cuda_device_names = [
            str(torch.cuda.get_device_name(index))
            for index in range(cuda_device_count)
        ]
    except Exception:
        pass

    summary: dict[str, Any] = {
        "model": model_name,
        "trust_remote_code": resolved_trust_remote_code,
        "backend": actual_backend_name,
        "engine": actual_engine_name,
        "chat_template": resolved_chat_template or "auto",
        "prompt_style": resolved_prompt_style,
        "session_len": int(session_len),
        "max_new_tokens": int(max_new_tokens),
        "retry_truncated": bool(retry_truncated),
        "retry_max_new_tokens": int(retry_max_new_tokens),
        "tensor_parallel": int(tensor_parallel),
        "inference_batch_size": int(inference_batch_size),
        "max_images_per_batch": int(max_images_per_batch) if max_images_per_batch is not None else None,
        "engine_max_batch_size": int(engine_max_batch_size) if engine_max_batch_size is not None else None,
        "vision_max_batch_size": int(vision_max_batch_size) if vision_max_batch_size is not None else None,
        "cache_max_entry_count": float(cache_max_entry_count),
        "image_loader_workers": int(image_loader_workers),
        "batch_by_panel_count": bool(batch_by_panel_count),
        "repetition_penalty": float(repetition_penalty),
        "repetition_ngram_size": int(repetition_ngram_size),
        "repetition_ngram_threshold": int(repetition_ngram_threshold),
        "repetition_ngram_effective": actual_engine_name == "pytorch",
        "retention_rule": "exact_count_nonempty_not_truncated",
        "run_kind": str(run_kind),
        "source_run_dir": str(source_run_dir or ""),
        "selected_parent_id_count": len(include_parent_ids) if include_parent_ids is not None else None,
        "prompt_version": PROMPT_VERSION,
        "output_delimiter": OUTPUT_DELIMITER,
        "subfigure_order_source": SUBFIGURE_ORDER_SOURCE,
        "parent_dir": str(parent_dir),
        "panel_dir": str(panel_dir),
        "parent_image_root": str(parent_image_root),
        "panel_image_root": str(panel_image_root),
        "input_parent_rows": int(parent_table.num_rows),
        "input_panel_rows": int(panel_table.num_rows),
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_device_names": cuda_device_names,
        "backend_initialization_seconds": backend_initialization_seconds,
        "eligible_parent_rows": len(eligible),
        "min_panel_count": int(min_panel_count),
        "max_panel_count": int(max_panel_count) if max_panel_count is not None else None,
        "processed_parent_rows": existing_processed,
        "aligned_parent_rows": existing_aligned,
        "aligned_subfigure_rows": existing_aligned_subfigures,
        "failed_or_mismatched_parent_rows": existing_processed - existing_aligned,
        "excluded_parent_rows": existing_processed - existing_aligned,
        "status_counts": dict(sorted(status_counts.items())),
        "shards_total": len(chunks),
        "shards_processed_this_run": 0,
        "shards_skipped_by_resume": len(chunks) - len(pending),
        "inference_batches": 0,
        "inference_parent_rows": 0,
        "inference_requests": 0,
        "inference_images": 0,
        "inference_seconds": 0.0,
        "image_load_seconds": 0.0,
        "generation_seconds": 0.0,
        "retry_batches": 0,
        "retry_requests": 0,
        "retry_recovered_parent_rows": 0,
        "parents_per_inference_second": None,
        "requests_per_inference_second": None,
        "images_per_inference_second": None,
        "end_to_end_seconds": 0.0,
    }

    pending_parent_count = sum(len(chunk) for _, chunk, *_ in pending)
    progress = tqdm(
        total=pending_parent_count,
        desc="Caption separation and alignment",
        unit="parent",
    )
    try:
        for shard_index, chunk, parent_path, subfigure_path, raw_path in pending:
            prepared: dict[str, dict[str, Any]] = {}
            requests: list[InferenceRequest] = []

            for parent in chunk:
                parent_id = _text(parent, "parent_image_id")
                panels = grouped_panels[parent_id]
                main_caption = _text(panels[0], "caption")
                parent_image = _resolve_path(parent.get("parent_local_image_path"), parent_image_root)
                panel_images = [_resolve_path(panel.get("local_panel_path"), panel_image_root) for panel in panels]
                image_paths = [parent_image, *panel_images]
                missing = [str(path) for path in image_paths if not path.exists()]
                prepared[parent_id] = {
                    "parent": parent,
                    "panels": panels,
                    "main_caption": main_caption,
                    "status": "",
                    "error": "",
                    "result": None,
                }
                if missing:
                    prepared[parent_id]["status"] = "missing_image"
                    prepared[parent_id]["error"] = "Missing image(s): " + "; ".join(missing[:5])
                elif not main_caption.strip():
                    prepared[parent_id]["status"] = "missing_main_caption"
                    prepared[parent_id]["error"] = "Main caption is empty"
                else:
                    requests.append(
                        InferenceRequest(
                            request_id=parent_id,
                            prompt=(
                                build_prompt(main_caption, len(panels))
                                if resolved_prompt_style == "explicit_tokens"
                                else build_ordered_images_prompt(main_caption, len(panels))
                            ),
                            image_paths=tuple(image_paths),
                        )
                    )

            request_by_id = {request.request_id: request for request in requests}
            immediate_invalid = len(chunk) - len(requests)
            if immediate_invalid:
                progress.update(immediate_invalid)

            request_batches = plan_request_batches(
                requests,
                max_requests=inference_batch_size,
                max_images=max_images_per_batch,
                group_by_image_count=batch_by_panel_count,
            )
            assert backend is not None or not request_batches
            for request_batch in request_batches:
                batch_started = time.perf_counter()
                results = _run_batch_with_isolation(backend, request_batch)  # type: ignore[arg-type]
                elapsed = time.perf_counter() - batch_started
                summary["inference_batches"] += 1
                summary["inference_parent_rows"] += len(request_batch)
                summary["inference_requests"] += len(request_batch)
                summary["inference_images"] += sum(request.image_count for request in request_batch)
                summary["inference_seconds"] += elapsed
                if results:
                    summary["image_load_seconds"] += float(
                        results[0].metadata.get("image_load_seconds") or 0.0
                    )
                    summary["generation_seconds"] += float(
                        results[0].metadata.get("generation_seconds") or 0.0
                    )
                for result in results:
                    result.metadata.setdefault("generation_attempts", 1)
                    prepared[result.request_id]["result"] = result
                progress.update(len(request_batch))
                progress.set_postfix(
                    batches=summary["inference_batches"],
                    retries=summary["retry_requests"],
                    images_per_s=(
                        f'{summary["inference_images"] / summary["inference_seconds"]:.2f}'
                        if summary["inference_seconds"] > 0
                        else "0.00"
                    ),
                )

            if retry_truncated:
                retry_requests = [
                    request_by_id[parent_id]
                    for parent_id, item in prepared.items()
                    if parent_id in request_by_id
                    and item["result"] is not None
                    and not item["result"].error
                    and str(
                        item["result"].metadata.get("finish_reason") or ""
                    ).lower() == "length"
                ]
                retry_batches = plan_request_batches(
                    retry_requests,
                    max_requests=inference_batch_size,
                    max_images=max_images_per_batch,
                    group_by_image_count=batch_by_panel_count,
                )
                for retry_batch in retry_batches:
                    batch_started = time.perf_counter()
                    retry_results = _run_batch_with_isolation(
                        backend,
                        retry_batch,
                        max_new_tokens=retry_max_new_tokens,
                    )  # type: ignore[arg-type]
                    elapsed = time.perf_counter() - batch_started
                    summary["inference_batches"] += 1
                    summary["retry_batches"] += 1
                    summary["inference_requests"] += len(retry_batch)
                    summary["retry_requests"] += len(retry_batch)
                    summary["inference_images"] += sum(
                        request.image_count for request in retry_batch
                    )
                    summary["inference_seconds"] += elapsed
                    if retry_results:
                        summary["image_load_seconds"] += float(
                            retry_results[0].metadata.get("image_load_seconds") or 0.0
                        )
                        summary["generation_seconds"] += float(
                            retry_results[0].metadata.get("generation_seconds") or 0.0
                        )
                    for retry_result in retry_results:
                        initial_result = prepared[retry_result.request_id]["result"]
                        retry_result.metadata.update(
                            generation_attempts=2,
                            retried_after_truncation=True,
                            initial_finish_reason=str(
                                initial_result.metadata.get("finish_reason") or ""
                            ),
                            initial_generate_token_len=int(
                                initial_result.metadata.get("generate_token_len") or 0
                            ),
                        )
                        prepared[retry_result.request_id]["result"] = retry_result
                    progress.set_postfix(
                        batches=summary["inference_batches"],
                        retries=summary["retry_requests"],
                        images_per_s=(
                            f'{summary["inference_images"] / summary["inference_seconds"]:.2f}'
                            if summary["inference_seconds"] > 0
                            else "0.00"
                        ),
                    )

            parent_outputs: list[dict[str, Any]] = []
            subfigure_outputs: list[dict[str, Any]] = []
            raw_outputs: list[dict[str, Any]] = []

            for parent in chunk:
                parent_id = _text(parent, "parent_image_id")
                item = prepared[parent_id]
                panels = item["panels"]
                main_caption = item["main_caption"]
                result: InferenceResult | None = item["result"]
                raw_prediction = ""
                parsed: list[str] = []
                batch_metadata: dict[str, Any] = {}

                if item["status"]:
                    status = item["status"]
                    error = item["error"]
                elif result is None:
                    status = "inference_failed"
                    error = "No backend result was returned"
                elif result.error:
                    status = "inference_failed"
                    error = result.error
                    batch_metadata = result.metadata
                else:
                    raw_prediction = result.text
                    batch_metadata = result.metadata
                    parsed = parse_subcaptions(raw_prediction)
                    status, error = _status_for_prediction(
                        parsed,
                        len(panels),
                        finish_reason=str(batch_metadata.get("finish_reason") or ""),
                    )

                status_counts[status] += 1
                if status == "aligned" and bool(batch_metadata.get("retried_after_truncation")):
                    summary["retry_recovered_parent_rows"] += 1
                parent_outputs.append(
                    _parent_output_record(
                        parent,
                        main_caption=main_caption,
                        expected=len(panels),
                        parsed=parsed,
                        raw_prediction=raw_prediction,
                        status=status,
                        error=error,
                        model_name=model_name,
                        backend_name=actual_backend_name,
                        engine_name=actual_engine_name,
                        chat_template=resolved_chat_template,
                        prompt_style=resolved_prompt_style,
                        session_len=session_len,
                        max_new_tokens=max_new_tokens,
                        batch_metadata=batch_metadata,
                        run_kind=run_kind,
                        source_run_dir=source_run_dir,
                    )
                )
                if status == "aligned":
                    subfigure_outputs.extend(
                        _subfigure_output_records(
                            panels,
                            main_caption=main_caption,
                            parsed=parsed,
                            status=status,
                            error=error,
                            model_name=model_name,
                            backend_name=actual_backend_name,
                            engine_name=actual_engine_name,
                            chat_template=resolved_chat_template,
                            prompt_style=resolved_prompt_style,
                            session_len=session_len,
                            max_new_tokens=max_new_tokens,
                            batch_metadata=batch_metadata,
                            run_kind=run_kind,
                            source_run_dir=source_run_dir,
                        )
                    )
                raw_outputs.append(
                    {
                        "parent_image_id": parent_id,
                        "expected_subcaption_count": len(panels),
                        "predicted_subcaption_count": len(parsed),
                        "caption_alignment_status": status,
                        "caption_alignment_error": error,
                        "raw_prediction": raw_prediction,
                        "parsed_subcaptions": parsed,
                        "retained_in_final_output": status == "aligned",
                        "backend_metadata": batch_metadata,
                    }
                )

            write_table(parent_path, parent_outputs, _parent_schema())
            write_table(subfigure_path, subfigure_outputs, _subfigure_schema(panel_table.schema))
            write_jsonl(raw_path, raw_outputs)

            summary["processed_parent_rows"] += len(parent_outputs)
            aligned_parent_count = sum(
                row["caption_alignment_status"] == "aligned"
                for row in parent_outputs
            )
            summary["aligned_parent_rows"] += aligned_parent_count
            summary["aligned_subfigure_rows"] += sum(
                1
                for row in subfigure_outputs
                if row["caption_alignment_status"] == "aligned"
            )
            excluded_count = len(parent_outputs) - aligned_parent_count
            summary["failed_or_mismatched_parent_rows"] += excluded_count
            summary["excluded_parent_rows"] += excluded_count
            summary["status_counts"] = dict(sorted(status_counts.items()))
            summary["shards_processed_this_run"] += 1
            if summary["inference_seconds"] > 0:
                summary["parents_per_inference_second"] = summary["inference_parent_rows"] / summary["inference_seconds"]
                summary["requests_per_inference_second"] = summary["inference_requests"] / summary["inference_seconds"]
                summary["images_per_inference_second"] = summary["inference_images"] / summary["inference_seconds"]
            summary["end_to_end_seconds"] = time.perf_counter() - pipeline_started
            write_json(summary_path, summary)
            progress.set_postfix(
                aligned=summary["aligned_parent_rows"],
                excluded=summary["excluded_parent_rows"],
                retries=summary["retry_requests"],
                images_per_s=(
                    f'{summary["inference_images"] / summary["inference_seconds"]:.2f}'
                    if summary["inference_seconds"] > 0
                    else "0.00"
                ),
            )
    finally:
        progress.close()
        if owns_backend and backend is not None:
            backend.close()

    summary["end_to_end_seconds"] = time.perf_counter() - pipeline_started
    write_json(summary_path, summary)
    _write_retry_candidates(parent_output_dir, output_dir / "retry_candidates.jsonl")
    return summary_path


def _write_retry_candidates(parent_output_dir: Path, output_path: Path) -> Path:
    """Write a compact manifest of outputs that stopped at the token limit."""
    try:
        rows = read_dataset(parent_output_dir).to_pylist()
    except FileNotFoundError:
        rows = []
    candidates = [
        {
            "parent_image_id": str(row.get("parent_image_id") or ""),
            "caption_alignment_status": str(row.get("caption_alignment_status") or ""),
            "expected_subcaption_count": int(row.get("expected_subcaption_count") or 0),
            "predicted_subcaption_count": int(row.get("predicted_subcaption_count") or 0),
            "caption_max_new_tokens": int(row.get("caption_max_new_tokens") or 0),
            "caption_finish_reason": str(row.get("caption_finish_reason") or ""),
            "caption_generate_token_len": int(row.get("caption_generate_token_len") or 0),
        }
        for row in rows
        if str(row.get("caption_alignment_status") or "") == "generation_truncated"
    ]
    return write_jsonl(output_path, candidates)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SESSION_LEN",
    "DEFAULT_RETRY_MAX_NEW_TOKENS",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PARENTS_PER_SHARD",
    "DEFAULT_INFERENCE_BATCH_SIZE",
    "DEFAULT_MAX_IMAGES_PER_BATCH",
    "PROMPT_PREFIX",
    "build_prompt",
    "clean_prediction",
    "parse_subcaptions",
    "resolve_chat_template",
    "resolve_prompt_style",
    "align_directory",
]
