"""Parquet and JSON helpers for Stage 5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def find_parquet_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    files = sorted(directory.glob("part-*.parquet"))
    if not files:
        files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under: {directory}")
    return files


def output_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("source_type", pa.string()),
            ("pmcid", pa.string()),
            ("article_version", pa.string()),
            ("resolved_article_version", pa.string()),
            ("pmid", pa.string()),
            ("article_title", pa.string()),
            ("journal_title", pa.string()),
            ("license", pa.string()),
            ("figure_id", pa.string()),
            ("figure_label", pa.string()),
            ("image_id", pa.string()),
            ("parent_image_id", pa.string()),
            ("subfigure_index", pa.int64()),
            ("local_image_path", pa.string()),
            ("selected_image_href", pa.string()),
            ("caption", pa.string()),
            ("main_caption", pa.string()),
            ("reference_texts", pa.list_(pa.string())),
            ("upstream_status", pa.string()),
            ("upstream_model", pa.string()),
            ("upstream_score", pa.float64()),
            ("medical_score", pa.float32()),
            ("medical_label", pa.int8()),
            ("medical_class_name", pa.string()),
            ("is_medical", pa.bool_()),
            ("medical_classification_status", pa.string()),
            ("medical_classification_error", pa.string()),
            ("medical_classification_model", pa.string()),
            ("medical_classification_architecture", pa.string()),
            ("medical_classification_threshold", pa.float64()),
            ("medical_classification_positive_label", pa.int8()),
            ("medical_classification_image_size", pa.int32()),
        ]
    )


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def write_table(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=output_schema())
    pq.write_table(table, path, compression="zstd")
    return path


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
