"""Manifest-backed Medical Figure Classification pipeline."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .model import (
    CLASS_NAMES,
    DEFAULT_CHECKPOINT_FILENAME,
    DEFAULT_MODEL,
    build_preprocess,
    jsonable_data_config,
    load_classifier,
    predict_probabilities,
    resolve_device,
    resolve_timm_data_config,
)
from .records import (
    candidate_from_singlepanel,
    candidate_from_subfigure,
    predicted_label_from_score,
    resolve_candidate_path,
    singlepanel_is_eligible,
    subfigure_is_eligible,
)
from .storage import find_parquet_files, read_json, read_rows, write_json, write_table


class CandidateDataset:
    def __init__(
        self,
        candidates: list[dict[str, Any]],
        *,
        singlepanel_image_root: Path | None,
        subfigure_image_root: Path | None,
        preprocess,
    ) -> None:
        self.candidates = candidates
        self.singlepanel_image_root = singlepanel_image_root
        self.subfigure_image_root = subfigure_image_root
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int):
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        candidate = self.candidates[index]
        path = resolve_candidate_path(
            candidate,
            singlepanel_image_root=self.singlepanel_image_root,
            subfigure_image_root=self.subfigure_image_root,
        )
        try:
            with Image.open(path) as image:
                image.seek(0)
                tensor = self.preprocess(image.convert("RGB"))
            return index, tensor, ""
        except Exception as exc:
            return index, None, str(exc)


def collate_candidates(batch):
    import torch

    indices: list[int] = []
    tensors = []
    errors: list[tuple[int, str]] = []
    for index, tensor, error in batch:
        if tensor is None:
            errors.append((index, error))
        else:
            indices.append(index)
            tensors.append(tensor)
    return indices, torch.stack(tensors) if tensors else None, errors


def _candidate_inputs(
    singlepanel_dir: Path | None,
    subfigure_dir: Path | None,
) -> list[tuple[str, Path]]:
    inputs: list[tuple[str, Path]] = []
    if singlepanel_dir is not None:
        inputs.extend(("single_panel", path) for path in find_parquet_files(singlepanel_dir))
    if subfigure_dir is not None:
        inputs.extend(("subfigure", path) for path in find_parquet_files(subfigure_dir))
    if not inputs:
        raise ValueError("At least one of --singlepanel-dir or --subfigure-dir is required")
    return sorted(inputs, key=lambda item: (item[0], str(item[1])))


def _prepare_candidates(source_type: str, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if source_type == "single_panel":
        eligible = [row for row in rows if singlepanel_is_eligible(row)]
        return [candidate_from_singlepanel(row) for row in eligible], len(rows) - len(eligible)
    if source_type == "subfigure":
        eligible = [row for row in rows if subfigure_is_eligible(row)]
        return [candidate_from_subfigure(row) for row in eligible], len(rows) - len(eligible)
    raise ValueError(f"Unknown source_type: {source_type}")


def _base_output(candidate: dict[str, Any], *, model_name: str, architecture: str, threshold: float, medical_label: int, image_size: int) -> dict[str, Any]:
    return {
        **candidate,
        "medical_score": None,
        "medical_label": None,
        "medical_class_name": "",
        "is_medical": None,
        "medical_classification_status": "",
        "medical_classification_error": "",
        "medical_classification_model": model_name,
        "medical_classification_architecture": architecture,
        "medical_classification_threshold": float(threshold),
        "medical_classification_positive_label": int(medical_label),
        "medical_classification_image_size": int(image_size),
    }


def _config_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    existing = read_json(path)
    return all(existing.get(key) == value for key, value in expected.items())


def classify_sources(
    output_dir: str | Path,
    *,
    singlepanel_dir: str | Path | None = None,
    singlepanel_image_root: str | Path | None = None,
    subfigure_dir: str | Path | None = None,
    subfigure_image_root: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
    batch_size: int = 256,
    loader_workers: int = 4,
    threshold: float = 0.5,
    medical_label: int = 1,
    image_size: int | None = None,
    device: str = "auto",
    amp: bool = False,
    max_images: int | None = None,
    force: bool = False,
) -> Path:
    import torch
    from torch.utils.data import DataLoader

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if loader_workers < 0:
        raise ValueError("loader_workers must be non-negative")
    if medical_label not in {0, 1}:
        raise ValueError("medical_label must be 0 or 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive")

    output_dir = Path(output_dir)
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    singlepanel_dir = Path(singlepanel_dir) if singlepanel_dir else None
    subfigure_dir = Path(subfigure_dir) if subfigure_dir else None
    singlepanel_image_root = Path(singlepanel_image_root) if singlepanel_image_root else None
    subfigure_image_root = Path(subfigure_image_root) if subfigure_image_root else None
    if singlepanel_dir is not None and singlepanel_image_root is None:
        raise ValueError("--singlepanel-image-root is required with --singlepanel-dir")
    if subfigure_dir is not None and subfigure_image_root is None:
        raise ValueError("--subfigure-image-root is required with --subfigure-dir")

    selected_device = resolve_device(device)
    if selected_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    init_start = time.perf_counter()
    model, architecture, checkpoint_path, checkpoint = load_classifier(
        model_name, checkpoint_filename, selected_device
    )
    data_config = resolve_timm_data_config(model, image_size)
    preprocess = build_preprocess(data_config)
    selected_image_size = int(data_config["input_size"][-1])
    init_seconds = time.perf_counter() - init_start

    inputs = _candidate_inputs(singlepanel_dir, subfigure_dir)
    config_path = output_dir / "classification_config.json"
    expected_config = {
        "model": model_name,
        "checkpoint_filename": checkpoint_filename,
        "architecture": architecture,
        "singlepanel_dir": str(singlepanel_dir or ""),
        "singlepanel_image_root": str(singlepanel_image_root or ""),
        "subfigure_dir": str(subfigure_dir or ""),
        "subfigure_image_root": str(subfigure_image_root or ""),
        "batch_size": batch_size,
        "loader_workers": loader_workers,
        "threshold": threshold,
        "medical_label": medical_label,
        "image_size": selected_image_size,
        "device": selected_device,
        "amp": bool(amp),
        "max_images": max_images,
        "timm_data_config": jsonable_data_config(data_config),
    }
    if config_path.exists() and not force and not _config_matches(config_path, expected_config):
        raise RuntimeError(
            f"Existing output configuration differs: {config_path}. Use --force or another output directory."
        )
    write_json(config_path, expected_config)

    classified_dir = output_dir / "manifests" / "classified"
    medical_dir = output_dir / "manifests" / "medical"
    nonmedical_dir = output_dir / "manifests" / "non_medical"
    for directory in (classified_dir, medical_dir, nonmedical_dir):
        directory.mkdir(parents=True, exist_ok=True)

    import timm

    def json_scalar(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    summary: dict[str, Any] = {
        **expected_config,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": json_scalar(checkpoint.get("epoch")),
        "checkpoint_metric": json_scalar(checkpoint.get("metric")),
        "checkpoint_version": json_scalar(checkpoint.get("version")),
        "torch_version": str(torch.__version__),
        "timm_version": str(timm.__version__),
        "checkpoint_initialization_seconds": init_seconds,
        "input_shards": len(inputs),
        "processed_shards": 0,
        "skipped_existing_shards": 0,
        "input_rows": 0,
        "eligible_rows": 0,
        "skipped_upstream_rows": 0,
        "classified_rows": 0,
        "medical_rows": 0,
        "non_medical_rows": 0,
        "decode_failed_rows": 0,
        "source_counts": {},
        "inference_seconds": 0.0,
        "end_to_end_seconds": 0.0,
        "class_map": {"0": CLASS_NAMES[0], "1": CLASS_NAMES[1]},
        "retention_rule": f"medical_score > {threshold} for class index {medical_label}",
    }
    source_counts: Counter[str] = Counter()
    start_time = time.perf_counter()
    remaining = max_images

    progress = tqdm(inputs, desc="Medical Figure Classification", unit="shard")
    for part_index, (source_type, input_path) in enumerate(progress):
        if remaining is not None and remaining <= 0:
            break
        part_name = f"part-{part_index:06d}.parquet"
        classified_path = classified_dir / part_name
        medical_path = medical_dir / part_name
        nonmedical_path = nonmedical_dir / part_name

        if classified_path.exists() and not force and max_images is None:
            existing = read_rows(classified_path)
            summary["input_rows"] += len(existing)
            summary["eligible_rows"] += len(existing)
            summary["classified_rows"] += sum(row.get("medical_classification_status") == "classified" for row in existing)
            summary["medical_rows"] += sum(row.get("is_medical") is True for row in existing)
            summary["non_medical_rows"] += sum(row.get("is_medical") is False for row in existing)
            summary["decode_failed_rows"] += sum(row.get("medical_classification_status") == "decode_failed" for row in existing)
            source_counts.update(str(row.get("source_type") or "") for row in existing)
            summary["skipped_existing_shards"] += 1
            continue

        rows = read_rows(input_path)
        candidates, skipped = _prepare_candidates(source_type, rows)
        if remaining is not None:
            candidates = candidates[:remaining]
            remaining -= len(candidates)
        outputs = [
            _base_output(
                candidate,
                model_name=model_name,
                architecture=architecture,
                threshold=threshold,
                medical_label=medical_label,
                image_size=selected_image_size,
            )
            for candidate in candidates
        ]

        dataset = CandidateDataset(
            candidates,
            singlepanel_image_root=singlepanel_image_root,
            subfigure_image_root=subfigure_image_root,
            preprocess=preprocess,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=selected_device.startswith("cuda"),
            persistent_workers=loader_workers > 0,
            collate_fn=collate_candidates,
        )
        inference_start = time.perf_counter()
        for valid_indices, tensors, errors in tqdm(
            loader,
            desc=f"Classifying {source_type} {input_path.name}",
            leave=False,
            unit="batch",
        ):
            for index, error in errors:
                outputs[index]["medical_classification_status"] = "decode_failed"
                outputs[index]["medical_classification_error"] = error
            if tensors is None:
                continue
            probabilities = predict_probabilities(
                model, tensors, device=selected_device, amp=amp
            )
            scores = probabilities[:, medical_label].tolist()
            for index, score in zip(valid_indices, scores, strict=True):
                label = predicted_label_from_score(
                    float(score), threshold=threshold, medical_label=medical_label
                )
                outputs[index].update(
                    medical_score=float(score),
                    medical_label=int(label),
                    medical_class_name=CLASS_NAMES[int(label)],
                    is_medical=bool(label == medical_label),
                    medical_classification_status="classified",
                )
        summary["inference_seconds"] += time.perf_counter() - inference_start

        medical_rows = [row for row in outputs if row.get("is_medical") is True]
        nonmedical_rows = [row for row in outputs if row.get("is_medical") is False]
        write_table(classified_path, outputs)
        write_table(medical_path, medical_rows)
        write_table(nonmedical_path, nonmedical_rows)

        summary["input_rows"] += len(rows)
        summary["eligible_rows"] += len(candidates)
        summary["skipped_upstream_rows"] += skipped
        summary["classified_rows"] += len(medical_rows) + len(nonmedical_rows)
        summary["medical_rows"] += len(medical_rows)
        summary["non_medical_rows"] += len(nonmedical_rows)
        summary["decode_failed_rows"] += sum(
            row.get("medical_classification_status") == "decode_failed" for row in outputs
        )
        source_counts.update(source_type for _ in candidates)
        summary["processed_shards"] += 1
        summary["source_counts"] = dict(sorted(source_counts.items()))
        summary["end_to_end_seconds"] = time.perf_counter() - start_time
        write_json(output_dir / "medical_figure_classification_summary.json", summary)
        progress.set_postfix(medical=summary["medical_rows"], classified=summary["classified_rows"])

    summary["source_counts"] = dict(sorted(source_counts.items()))
    summary["end_to_end_seconds"] = time.perf_counter() - start_time
    classified = int(summary["classified_rows"])
    summary["images_per_second"] = (
        classified / float(summary["inference_seconds"])
        if summary["inference_seconds"]
        else 0.0
    )
    return write_json(output_dir / "medical_figure_classification_summary.json", summary)
