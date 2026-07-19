"""Parquet schemas and small storage helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

STAGE1_FIELDS = [
    ("pmcid", pa.string()),
    ("article_version", pa.string()),
    ("pmid", pa.string()),
    ("article_title", pa.string()),
    ("journal_title", pa.string()),
    ("license", pa.string()),
    ("figure_id", pa.string()),
    ("figure_label", pa.string()),
    ("caption", pa.string()),
    ("reference_texts", pa.list_(pa.string())),
    ("reference_texts_screening", pa.list_(pa.string())),
    ("image_hrefs", pa.list_(pa.string())),
    ("source_xml", pa.string()),
    ("screening_score", pa.float32()),
    ("screening_label", pa.int8()),
    ("retained", pa.bool_()),
    ("screening_model", pa.string()),
    ("screening_reference_mode", pa.string()),
    ("screening_threshold", pa.float64()),
    ("screening_max_length", pa.int32()),
    ("screening_positive_label", pa.int8()),
]

IMAGE_FIELDS = [
    ("resolved_article_version", pa.string()),
    ("article_resolution_method", pa.string()),
    ("article_candidate_versions", pa.int16()),
    ("article_metadata_key", pa.string()),
    ("article_is_retracted", pa.bool_()),
    ("article_metadata_license", pa.string()),
    ("selected_image_href", pa.string()),
    ("image_selection_method", pa.string()),
    ("image_candidate_count", pa.int16()),
    ("image_href_match_type", pa.string()),
    ("image_s3_url", pa.string()),
    ("image_s3_key", pa.string()),
    ("image_expected_md5", pa.string()),
    ("local_image_path", pa.string()),
    ("image_status", pa.string()),
    ("image_transfer", pa.string()),
    ("image_error", pa.string()),
    ("image_width", pa.int32()),
    ("image_height", pa.int32()),
    ("image_format", pa.string()),
    ("image_mode", pa.string()),
    ("image_md5", pa.string()),
]

DETECTION_FIELDS = [
    ("multipanel_score", pa.float32()),
    ("multipanel_label", pa.int8()),
    ("is_multipanel", pa.bool_()),
    ("detection_status", pa.string()),
    ("detection_error", pa.string()),
    ("detection_model", pa.string()),
    ("detection_architecture", pa.string()),
    ("detection_threshold", pa.float64()),
    ("detection_image_size", pa.int32()),
    ("detection_positive_label", pa.int8()),
]

IMAGE_MANIFEST_SCHEMA = pa.schema(STAGE1_FIELDS + IMAGE_FIELDS)
DETECTION_SCHEMA = pa.schema(STAGE1_FIELDS + IMAGE_FIELDS + DETECTION_FIELDS)


def normalize_stage1_record(record: dict[str, Any]) -> dict[str, Any]:
    """Fill optional fields when reading older but compatible Stage 1 outputs."""
    defaults: dict[str, Any] = {
        "pmcid": "",
        "article_version": "",
        "pmid": "",
        "article_title": "",
        "journal_title": "",
        "license": "",
        "figure_id": "",
        "figure_label": "",
        "caption": "",
        "reference_texts": [],
        "reference_texts_screening": None,
        "image_hrefs": [],
        "source_xml": "",
        "screening_score": None,
        "screening_label": None,
        "retained": True,
        "screening_model": "",
        "screening_reference_mode": "",
        "screening_threshold": None,
        "screening_max_length": None,
        "screening_positive_label": None,
    }
    return {**defaults, **record}


def write_table(path: str | Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="zstd")


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
