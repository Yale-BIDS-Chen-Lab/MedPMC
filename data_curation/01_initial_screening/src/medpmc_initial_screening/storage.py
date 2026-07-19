"""Sharded Parquet utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

EXTRACTION_SCHEMA = pa.schema(
    [
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
    ]
)

SCREENING_SCHEMA = pa.schema(
    list(EXTRACTION_SCHEMA)
    + [
        ("screening_score", pa.float32()),
        ("screening_label", pa.int8()),
        ("retained", pa.bool_()),
        ("screening_model", pa.string()),
        ("screening_reference_mode", pa.string()),
        ("screening_threshold", pa.float64()),
        ("screening_max_length", pa.int32()),
        ("screening_positive_label", pa.int8()),
    ]
)


class ParquetShardWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        schema: pa.Schema,
        shard_size: int = 10_000,
        prefix: str = "part",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schema = schema
        self.shard_size = max(1, shard_size)
        self.prefix = prefix
        self.buffer: list[dict[str, Any]] = []
        existing = sorted(self.output_dir.glob(f"{prefix}-*.parquet"))
        self.shard_index = len(existing)
        self.rows_written = 0

    def write(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def write_many(self, records: list[dict[str, Any]]) -> None:
        for record in records:
            self.write(record)

    def flush(self) -> None:
        if not self.buffer:
            return
        table = pa.Table.from_pylist(self.buffer, schema=self.schema)
        path = self.output_dir / f"{self.prefix}-{self.shard_index:06d}.parquet"
        pq.write_table(table, path, compression="zstd")
        self.rows_written += len(self.buffer)
        self.shard_index += 1
        self.buffer.clear()

    def close(self) -> None:
        self.flush()


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
