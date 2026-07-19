"""Filesystem and Parquet helpers for Stage 4."""

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


def read_dataset(directory: str | Path):
    import pyarrow.dataset as ds

    return ds.dataset(find_parquet_files(directory), format="parquet").to_table()


def write_json(path: str | Path, value: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_table(path: str | Path, records: list[dict[str, Any]], schema) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, path, compression="zstd")
    return path
