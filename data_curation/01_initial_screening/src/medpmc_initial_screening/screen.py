"""Run the released MedPMC Initial Screening classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from .jats import format_model_input
from .storage import SCREENING_SCHEMA, write_json

DEFAULT_MODEL = "Yale-BIDS-Chen/medpmc-screening-pubmedbert-caption-reference"
DEFAULT_REFERENCE_MODE = "model-compatible"
REFERENCE_MODES = ("model-compatible", "clean")


def _reference_texts_for_screening(
    record: dict[str, Any],
    reference_mode: str,
) -> list[str]:
    clean_references = record.get("reference_texts") or []
    if reference_mode == "clean":
        return clean_references

    # The nullable override is stored only when recursive paragraph extraction
    # differs from the cleaned downstream representation.
    screening_override = record.get("reference_texts_screening")
    if screening_override is None:
        return clean_references
    return list(screening_override)


def _device_name(requested: str) -> str:
    import torch

    requested = requested.strip().lower()
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but PyTorch cannot initialize CUDA. "
                f"torch={torch.__version__}, torch CUDA build={torch.version.cuda}. "
                "Check the NVIDIA driver/PyTorch CUDA compatibility or use --device cpu."
            )
        return requested
    if requested == "mps":
        if not (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is not available.")
        return requested
    if requested == "cpu":
        return requested
    if requested != "auto":
        raise ValueError(
            "Unsupported device. Use auto, cpu, cuda, cuda:N, or mps."
        )

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _predict_scores(
    texts: list[str],
    tokenizer,
    model,
    device: str,
    *,
    batch_size: int,
    max_length: int,
    positive_label: int,
    progress_desc: str,
) -> list[float]:
    import torch

    scores: list[float] = []
    starts = range(0, len(texts), batch_size)
    for start in tqdm(
        starts,
        desc=progress_desc,
        leave=False,
        unit="batch",
    ):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)
        scores.extend(probabilities[:, positive_label].detach().cpu().tolist())
    return scores


def _validate_existing_shard(
    screened_path: Path,
    *,
    model_name: str,
    reference_mode: str,
    threshold: float,
    max_length: int,
    positive_label: int,
) -> tuple[int, int]:
    required = {
        "retained",
        "screening_model",
        "screening_reference_mode",
        "screening_threshold",
        "screening_max_length",
        "screening_positive_label",
    }
    schema = pq.read_schema(screened_path)
    missing = required.difference(schema.names)
    if missing:
        raise RuntimeError(
            f"Existing screening shard {screened_path} predates the current "
            f"provenance fields ({', '.join(sorted(missing))}). Re-run with "
            "--force or use a new output directory."
        )

    table = pq.read_table(screened_path, columns=sorted(required))
    rows = table.to_pylist()
    for row in rows:
        actual = {
            "model": row.get("screening_model"),
            "reference_mode": row.get("screening_reference_mode"),
            "threshold": float(row.get("screening_threshold")),
            "max_length": int(row.get("screening_max_length")),
            "positive_label": int(row.get("screening_positive_label")),
        }
        expected = {
            "model": model_name,
            "reference_mode": reference_mode,
            "threshold": float(threshold),
            "max_length": int(max_length),
            "positive_label": int(positive_label),
        }
        if actual != expected:
            raise RuntimeError(
                f"Existing screening shard {screened_path} was generated with "
                f"a different configuration: {actual}. Expected: {expected}. "
                "Re-run with --force or use a new output directory."
            )

    retained = sum(bool(row["retained"]) for row in rows)
    return len(rows), retained


def screen_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    threshold: float = 0.5,
    max_length: int = 512,
    positive_label: int = 1,
    device: str = "auto",
    reference_mode: str = DEFAULT_REFERENCE_MODE,
    force: bool = False,
) -> Path:
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Install with: "
            "pip install -e '.[inference]'"
        ) from exc

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    reference_mode = reference_mode.strip().lower()
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(
            f"Unsupported reference mode: {reference_mode}. "
            f"Choose one of: {', '.join(REFERENCE_MODES)}"
        )

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    screened_dir = output_dir / "screened"
    retained_dir = output_dir / "retained"
    screened_dir.mkdir(parents=True, exist_ok=True)
    retained_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "screening_summary.json"

    if summary_path.exists() and not force:
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_summary = {
            "model": model_name,
            "threshold": float(threshold),
            "max_length": int(max_length),
            "positive_label": int(positive_label),
            "reference_mode": reference_mode,
        }
        actual_summary = {
            key: existing_summary.get(key) for key in expected_summary
        }
        if actual_summary == expected_summary:
            return summary_path
        raise RuntimeError(
            "The existing screening_summary.json was created with a different "
            f"configuration: {actual_summary}. Expected: {expected_summary}. "
            "Re-run with --force or use a new output directory."
        )

    requested_device = device.strip().lower()
    selected_device = _device_name(requested_device)
    print(
        f"Screening configuration: device={selected_device}, "
        f"batch_size={batch_size}, max_length={max_length}, "
        f"reference_mode={reference_mode}"
    )
    if requested_device == "auto" and selected_device == "cpu":
        print(
            "Warning: --device auto selected CPU. For a GPU job, use --device cuda "
            "so CUDA initialization failures stop immediately instead of falling back."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(selected_device)
    model.eval()

    summary: dict[str, Any] = {
        "model": model_name,
        "device": selected_device,
        "batch_size": batch_size,
        "max_length": max_length,
        "threshold": threshold,
        "positive_label": positive_label,
        "reference_mode": reference_mode,
        "reference_field": (
            "reference_texts_screening (fallback: reference_texts)"
            if reference_mode == "model-compatible"
            else "reference_texts"
        ),
        "model_compatible_fallback_rows": 0,
        "input_rows": 0,
        "retained_rows": 0,
        "shards_processed": 0,
    }

    input_paths = sorted(input_dir.glob("part-*.parquet"))
    if not input_paths:
        raise FileNotFoundError(f"No part-*.parquet files found under {input_dir}")

    shard_progress = tqdm(input_paths, desc="Screening figure-text shards", unit="shard")
    for shard_index, input_path in enumerate(shard_progress, start=1):
        screened_path = screened_dir / input_path.name
        retained_path = retained_dir / input_path.name
        if screened_path.exists() and retained_path.exists() and not force:
            row_count, retained_count = _validate_existing_shard(
                screened_path,
                model_name=model_name,
                reference_mode=reference_mode,
                threshold=threshold,
                max_length=max_length,
                positive_label=positive_label,
            )
            summary["input_rows"] += row_count
            summary["retained_rows"] += retained_count
            summary["shards_processed"] += 1
            shard_progress.set_postfix(
                rows=summary["input_rows"],
                retained=summary["retained_rows"],
            )
            continue

        table = pq.read_table(input_path)
        has_screening_reference_field = (
            "reference_texts_screening" in table.schema.names
        )
        records = table.to_pylist()
        if reference_mode == "model-compatible" and not has_screening_reference_field:
            summary["model_compatible_fallback_rows"] += len(records)

        texts = [
            format_model_input(
                record.get("caption", ""),
                _reference_texts_for_screening(record, reference_mode),
            )
            for record in records
        ]
        scores = _predict_scores(
            texts,
            tokenizer,
            model,
            selected_device,
            batch_size=batch_size,
            max_length=max_length,
            positive_label=positive_label,
            progress_desc=f"Inference {shard_index}/{len(input_paths)}",
        )

        output_records: list[dict[str, Any]] = []
        retained_records: list[dict[str, Any]] = []
        for record, score in zip(records, scores, strict=True):
            retained = float(score) >= threshold
            output = {
                **record,
                "reference_texts_screening": record.get(
                    "reference_texts_screening"
                ),
                "screening_score": float(score),
                "screening_label": int(retained),
                "retained": retained,
                "screening_model": model_name,
                "screening_reference_mode": reference_mode,
                "screening_threshold": float(threshold),
                "screening_max_length": int(max_length),
                "screening_positive_label": int(positive_label),
            }
            output_records.append(output)
            if retained:
                retained_records.append(output)

        screened_table = pa.Table.from_pylist(output_records, schema=SCREENING_SCHEMA)
        pq.write_table(screened_table, screened_path, compression="zstd")

        retained_table = pa.Table.from_pylist(retained_records, schema=SCREENING_SCHEMA)
        pq.write_table(retained_table, retained_path, compression="zstd")

        summary["input_rows"] += len(output_records)
        summary["retained_rows"] += len(retained_records)
        summary["shards_processed"] += 1
        shard_progress.set_postfix(
            rows=summary["input_rows"],
            retained=summary["retained_rows"],
        )

    if summary["model_compatible_fallback_rows"]:
        print(
            "Warning: model-compatible screening was requested, but some input "
            "Parquet shards predate reference_texts_screening. Those rows used "
            "the cleaned reference_texts field. Re-run extraction for exact "
            "released-model preprocessing."
        )

    write_json(summary_path, summary)
    return summary_path
