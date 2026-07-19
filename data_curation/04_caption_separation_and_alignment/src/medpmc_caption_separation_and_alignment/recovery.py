"""Separate retry passes and deterministic merging for Stage 4 outputs."""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .alignment import align_directory
from .storage import read_dataset, write_json, write_jsonl, write_table


def parent_ids_with_status(
    run_dir: str | Path,
    statuses: Iterable[str] = ("generation_truncated",),
) -> set[str]:
    """Return parent IDs whose latest result has one of the requested statuses."""
    run_dir = Path(run_dir)
    requested = {str(value).strip() for value in statuses if str(value).strip()}
    if not requested:
        raise ValueError("At least one retry status is required")
    table = read_dataset(run_dir / "manifests" / "parents")
    return {
        str(row.get("parent_image_id") or "")
        for row in table.to_pylist()
        if str(row.get("caption_alignment_status") or "") in requested
    }


def retry_directory(
    source_run_dir: str | Path,
    parent_dir: str | Path,
    panel_dir: str | Path,
    output_dir: str | Path,
    *,
    statuses: Iterable[str] = ("generation_truncated",),
    **align_kwargs: Any,
) -> Path:
    """Re-run only selected failed parents in a fresh process/output directory.

    ``max_new_tokens`` is supplied by the caller and is intentionally not capped
    at 2048. Users may chain retries (for example 2048, then 4096) by using the
    previous retry directory as the next ``source_run_dir``.
    """
    source_run_dir = Path(source_run_dir)
    requested_statuses = [str(value).strip() for value in statuses if str(value).strip()]
    selected = parent_ids_with_status(source_run_dir, requested_statuses)
    summary_path = align_directory(
        parent_dir,
        panel_dir,
        output_dir,
        include_parent_ids=selected,
        run_kind="retry",
        source_run_dir=source_run_dir,
        retry_truncated=False,
        **align_kwargs,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["retry_source_statuses"] = requested_statuses
    summary["retry_source_candidate_rows"] = len(selected)
    write_json(summary_path, summary)
    return summary_path


def _read_jsonl_directory(directory: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not directory.exists():
        return records
    for path in sorted(directory.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                parent_id = str(row.get("parent_image_id") or "")
                if parent_id:
                    records[parent_id] = row
    return records


def _unified_schema(tables: list[Any], additions: list[Any]):
    import pyarrow as pa

    schemas = [table.schema for table in tables]
    if additions:
        schemas.append(pa.schema(additions))
    return pa.unify_schemas(schemas)


def merge_retry_runs(
    base_run_dir: str | Path,
    retry_run_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Merge successful retry predictions into a first-pass run.

    Retry directories are applied in the order provided. Only ``aligned`` retry
    rows replace the previous selected result. Failed retry attempts remain in
    their own run directories and are summarized in ``retry_history.jsonl``.
    """
    import pyarrow as pa

    base_run_dir = Path(base_run_dir)
    retry_run_dirs = [Path(value) for value in retry_run_dirs]
    output_dir = Path(output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        if not force:
            raise FileExistsError(
                f"Merge output already exists: {output_dir}. Use --force to replace it."
            )
        shutil.rmtree(output_dir)

    base_parent_table = read_dataset(base_run_dir / "manifests" / "parents")
    base_subfigure_table = read_dataset(base_run_dir / "manifests" / "subfigures")
    retry_parent_tables = [
        read_dataset(path / "manifests" / "parents") for path in retry_run_dirs
    ]
    retry_subfigure_tables = [
        read_dataset(path / "manifests" / "subfigures") for path in retry_run_dirs
    ]

    base_parent_rows = base_parent_table.to_pylist()
    parent_by_id = {
        str(row.get("parent_image_id") or ""): dict(row) for row in base_parent_rows
    }
    base_status_by_id = {
        parent_id: str(row.get("caption_alignment_status") or "")
        for parent_id, row in parent_by_id.items()
    }

    subfigures_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in base_subfigure_table.to_pylist():
        subfigures_by_parent[str(row.get("parent_image_id") or "")].append(dict(row))

    raw_by_id = _read_jsonl_directory(base_run_dir / "raw_predictions")
    selected_source = {parent_id: str(base_run_dir) for parent_id in parent_by_id}
    selected_source_kind = {parent_id: "first_pass" for parent_id in parent_by_id}
    latest_attempt_status = dict(base_status_by_id)
    latest_attempt_source = {parent_id: str(base_run_dir) for parent_id in parent_by_id}
    retry_attempts: Counter[str] = Counter()
    retry_transitions: Counter[str] = Counter()
    retry_history: list[dict[str, Any]] = []
    replacements_per_run: dict[str, int] = {}

    for retry_dir, parent_table, subfigure_table in zip(
        retry_run_dirs, retry_parent_tables, retry_subfigure_tables, strict=True
    ):
        retry_subfigures: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in subfigure_table.to_pylist():
            retry_subfigures[str(row.get("parent_image_id") or "")].append(dict(row))
        retry_raw = _read_jsonl_directory(retry_dir / "raw_predictions")
        replacements = 0
        for row in parent_table.to_pylist():
            row = dict(row)
            parent_id = str(row.get("parent_image_id") or "")
            if parent_id not in parent_by_id:
                raise ValueError(
                    f"Retry parent {parent_id} does not exist in base run {base_run_dir}"
                )
            current_status = str(row.get("caption_alignment_status") or "")
            previous_status = latest_attempt_status.get(parent_id, "")
            transition = f"{previous_status}->{current_status}"
            retry_attempts[parent_id] += 1
            retry_transitions[transition] += 1
            latest_attempt_status[parent_id] = current_status
            latest_attempt_source[parent_id] = str(retry_dir)
            retry_history.append(
                {
                    "parent_image_id": parent_id,
                    "retry_run_dir": str(retry_dir),
                    "previous_status": previous_status,
                    "caption_alignment_status": current_status,
                    "selected_for_final_output": current_status == "aligned",
                    "caption_max_new_tokens": int(
                        row.get("caption_max_new_tokens") or 0
                    ),
                    "caption_finish_reason": str(
                        row.get("caption_finish_reason") or ""
                    ),
                    "caption_generate_token_len": int(
                        row.get("caption_generate_token_len") or 0
                    ),
                }
            )
            if current_status != "aligned":
                continue
            parent_by_id[parent_id] = row
            subfigures_by_parent[parent_id] = retry_subfigures.get(parent_id, [])
            if parent_id in retry_raw:
                raw_by_id[parent_id] = retry_raw[parent_id]
            selected_source[parent_id] = str(retry_dir)
            selected_source_kind[parent_id] = "retry"
            replacements += 1
        replacements_per_run[str(retry_dir)] = replacements

    parent_outputs: list[dict[str, Any]] = []
    for parent_id in sorted(parent_by_id):
        row = dict(parent_by_id[parent_id])
        row.update(
            caption_final_prediction_source=selected_source_kind[parent_id],
            caption_final_source_run_dir=selected_source[parent_id],
            caption_base_status=base_status_by_id[parent_id],
            caption_retry_attempt_count=int(retry_attempts[parent_id]),
            caption_latest_attempt_status=latest_attempt_status[parent_id],
            caption_latest_attempt_source_run_dir=latest_attempt_source[parent_id],
        )
        parent_outputs.append(row)

    subfigure_outputs: list[dict[str, Any]] = []
    for parent_id in sorted(subfigures_by_parent):
        for row in sorted(
            subfigures_by_parent[parent_id],
            key=lambda value: int(value.get("subfigure_index") or 0),
        ):
            output = dict(row)
            output.update(
                caption_final_prediction_source=selected_source_kind.get(
                    parent_id, "first_pass"
                ),
                caption_final_source_run_dir=selected_source.get(
                    parent_id, str(base_run_dir)
                ),
            )
            subfigure_outputs.append(output)

    parent_schema = _unified_schema(
        [base_parent_table, *retry_parent_tables],
        [
            pa.field("caption_final_prediction_source", pa.string()),
            pa.field("caption_final_source_run_dir", pa.string()),
            pa.field("caption_base_status", pa.string()),
            pa.field("caption_retry_attempt_count", pa.int64()),
            pa.field("caption_latest_attempt_status", pa.string()),
            pa.field("caption_latest_attempt_source_run_dir", pa.string()),
        ],
    )
    subfigure_schema = _unified_schema(
        [base_subfigure_table, *retry_subfigure_tables],
        [
            pa.field("caption_final_prediction_source", pa.string()),
            pa.field("caption_final_source_run_dir", pa.string()),
        ],
    )

    write_table(
        output_dir / "manifests" / "parents" / "part-000000.parquet",
        parent_outputs,
        parent_schema,
    )
    write_table(
        output_dir / "manifests" / "subfigures" / "part-000000.parquet",
        subfigure_outputs,
        subfigure_schema,
    )
    selected_raw = []
    for parent_id in sorted(parent_by_id):
        row = dict(raw_by_id.get(parent_id, {}))
        row.setdefault("parent_image_id", parent_id)
        row["final_prediction_source"] = selected_source_kind[parent_id]
        row["final_source_run_dir"] = selected_source[parent_id]
        selected_raw.append(row)
    write_jsonl(output_dir / "raw_predictions" / "part-000000.jsonl", selected_raw)
    write_jsonl(output_dir / "retry_history.jsonl", retry_history)

    status_counts = Counter(
        str(row.get("caption_alignment_status") or "") for row in parent_outputs
    )
    latest_status_counts = Counter(latest_attempt_status.values())
    selected_source_counts = Counter(selected_source_kind.values())
    summary = {
        "base_run_dir": str(base_run_dir),
        "retry_run_dirs": [str(path) for path in retry_run_dirs],
        "parent_rows": len(parent_outputs),
        "aligned_parent_rows": int(status_counts.get("aligned", 0)),
        "aligned_subfigure_rows": len(subfigure_outputs),
        "status_counts": dict(sorted(status_counts.items())),
        "selected_status_counts": dict(sorted(status_counts.items())),
        "latest_attempt_status_counts": dict(sorted(latest_status_counts.items())),
        "retry_transition_counts": dict(sorted(retry_transitions.items())),
        "selected_prediction_source_counts": dict(sorted(selected_source_counts.items())),
        "replacements_per_retry_run": replacements_per_run,
        "total_retry_attempt_rows": len(retry_history),
        "parents_recovered_by_retry": sum(
            base_status_by_id[parent_id] != "aligned"
            and str(parent_by_id[parent_id].get("caption_alignment_status") or "")
            == "aligned"
            for parent_id in parent_by_id
        ),
        "selection_rule": "apply retry runs in order; replace only with aligned predictions",
    }
    return write_json(output_dir / "caption_alignment_merge_summary.json", summary)


__all__ = [
    "parent_ids_with_status",
    "retry_directory",
    "merge_retry_runs",
]
