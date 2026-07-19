"""Schemas and filesystem helpers for Stage 3 outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def panel_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("source_type", pa.string()),
            ("pmcid", pa.string()),
            ("article_version", pa.string()),
            ("resolved_article_version", pa.string()),
            ("pmid", pa.string()),
            ("figure_id", pa.string()),
            ("figure_label", pa.string()),
            ("caption", pa.string()),
            ("reference_texts", pa.list_(pa.string())),
            ("selected_image_href", pa.string()),
            ("parent_image_id", pa.string()),
            ("parent_local_image_path", pa.string()),
            ("parent_image_width", pa.int64()),
            ("parent_image_height", pa.int64()),
            ("multipanel_score", pa.float64()),
            ("multipanel_label", pa.int64()),
            ("is_multipanel", pa.bool_()),
            ("detector_index", pa.int64()),
            ("spatial_index", pa.int64()),
            ("historical_order_index", pa.int64()),
            ("ordering_method", pa.string()),
            ("ordering_y_threshold", pa.float64()),
            ("subfigure_index", pa.int64()),
            ("image_id", pa.string()),
            ("panel_image_id", pa.string()),
            ("identifier_convention", pa.string()),
            ("panel_class_id", pa.int64()),
            ("panel_class_name", pa.string()),
            ("panel_confidence", pa.float64()),
            ("box_x1", pa.float64()),
            ("box_y1", pa.float64()),
            ("box_x2", pa.float64()),
            ("box_y2", pa.float64()),
            ("box_x1_normalized", pa.float64()),
            ("box_y1_normalized", pa.float64()),
            ("box_x2_normalized", pa.float64()),
            ("box_y2_normalized", pa.float64()),
            ("local_panel_path", pa.string()),
            ("panel_width", pa.int64()),
            ("panel_height", pa.int64()),
            ("panel_format", pa.string()),
            ("panel_md5", pa.string()),
            ("crop_method", pa.string()),
            ("crop_gain", pa.float64()),
            ("crop_pad", pa.int64()),
            ("crop_square", pa.bool_()),
            ("crop_status", pa.string()),
            ("crop_error", pa.string()),
            ("separation_model", pa.string()),
            ("separation_checkpoint", pa.string()),
            ("separation_confidence_threshold", pa.float64()),
            ("separation_device", pa.string()),
        ]
    )


def parent_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("pmcid", pa.string()),
            ("article_version", pa.string()),
            ("resolved_article_version", pa.string()),
            ("pmid", pa.string()),
            ("figure_id", pa.string()),
            ("figure_label", pa.string()),
            ("selected_image_href", pa.string()),
            ("parent_image_id", pa.string()),
            ("parent_local_image_path", pa.string()),
            ("multipanel_score", pa.float64()),
            ("panel_count", pa.int64()),
            ("separation_status", pa.string()),
            ("separation_error", pa.string()),
            ("separation_model", pa.string()),
            ("separation_checkpoint", pa.string()),
            ("separation_confidence_threshold", pa.float64()),
            ("separation_device", pa.string()),
            ("crop_method", pa.string()),
        ]
    )


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_table(path: str | Path, records: list[dict[str, Any]], schema) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path, compression="zstd")
    return path


def md5_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
