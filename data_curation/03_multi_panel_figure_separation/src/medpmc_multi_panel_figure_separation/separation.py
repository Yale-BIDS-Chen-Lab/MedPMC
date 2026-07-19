"""YOLOv10-based separation of multi-panel biomedical figures."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import os
from typing import Any
from urllib.parse import unquote

from PIL import Image
from tqdm import tqdm

from .storage import md5_file, panel_schema, parent_schema, write_json, write_table

DEFAULT_MODEL = "Yale-BIDS-Chen/medpmc-multi-fig-separation-yolov10"
DEFAULT_CHECKPOINT = "model.pt"
DEFAULT_CONFIDENCE = 0.5
CROP_METHOD = "ultralytics_save_one_box"
CROP_GAIN = 1.02
CROP_PAD = 10
CROP_SQUARE = False
IDENTIFIER_CONVENTION = "medpmc_final_subfigure_historical_row_major_v1"
SUBFIGURE_INDEX_SOURCE = "historical_order_index"
ORDERING_METHOD = "normalized_top_left_row_major"
ORDERING_Y_THRESHOLD = 0.05
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".gif")


def find_parquet_files(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    files = sorted(directory.glob("part-*.parquet"))
    if not files:
        files = sorted(directory.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under: {directory}")
    return files


def normalize_graphic_id(value: Any) -> str:
    text = unquote(str(value or "")).split("?", 1)[0].split("#", 1)[0]
    text = PurePosixPath(text).name
    lowered = text.casefold()
    for extension in IMAGE_EXTENSIONS:
        if lowered.endswith(extension):
            text = text[: -len(extension)]
            break
    return text


def semantic_image_id(parent_id: str, subfigure_index: int) -> str:
    return f"{parent_id}_{int(subfigure_index)}"


def parent_image_id(record: dict[str, Any]) -> str:
    pmcid = str(record.get("pmcid") or "")
    graphic = normalize_graphic_id(
        record.get("selected_image_href")
        or record.get("local_image_path")
        or record.get("figure_id")
    )
    return f"{pmcid}_{graphic}" if pmcid and graphic else f"{pmcid}_{record.get('figure_id') or 'figure'}"


def _resolve_device(requested: str) -> str | int:
    value = str(requested).strip().lower()
    if value in {"cuda", "cuda:0"}:
        return 0
    if value.startswith("cuda:"):
        return int(value.split(":", 1)[1])
    if value == "auto":
        try:
            import torch
            return 0 if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if value.isdigit():
        return int(value)
    return requested



@contextmanager
def trusted_full_checkpoint_loading():
    """Temporarily allow trusted full-model checkpoints on PyTorch >= 2.6.

    YOLOv10 checkpoints contain a serialized DetectionModel rather than a plain
    tensor-only state dict. PyTorch 2.6+ defaults torch.load to
    weights_only=True, so full-object loading is enabled only while loading the
    trusted checkpoint.
    """
    variable = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    previous = os.environ.get(variable)
    os.environ[variable] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous

def load_separator(model_name: str, checkpoint_filename: str):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface-hub is missing. Install with: pip install -e '.[inference]'"
        ) from exc

    try:
        from ultralytics import YOLOv10
    except ImportError as exc:
        raise RuntimeError(
            "The YOLOv10 implementation is missing. Install the official repository first:\n"
            "  python -m pip install 'git+https://github.com/THU-MIG/yolov10.git'"
        ) from exc

    checkpoint_path = Path(
        hf_hub_download(repo_id=model_name, filename=checkpoint_filename)
    )
    with trusted_full_checkpoint_loading():
        model = YOLOv10(str(checkpoint_path))
    return model, checkpoint_path


def _required_columns(names: list[str]) -> None:
    required = {
        "pmcid",
        "local_image_path",
        "detection_status",
        "is_multipanel",
        "selected_image_href",
    }
    missing = sorted(required - set(names))
    if missing:
        raise ValueError(f"Missing required Stage 2 columns: {missing}")


def _select_rows(table, max_figures: int | None) -> list[dict[str, Any]]:
    _required_columns(table.schema.names)
    rows = [
        row
        for row in table.to_pylist()
        if str(row.get("detection_status") or "") == "classified"
        and bool(row.get("is_multipanel"))
    ]
    if max_figures is not None:
        rows = rows[:max_figures]
    return rows


def _resolve_image_path(record: dict[str, Any], image_root: Path) -> Path:
    path = Path(str(record.get("local_image_path") or ""))
    return path if path.is_absolute() else image_root / path


def historical_row_major_order(
    boxes: list[dict[str, Any]],
    *,
    image_width: int | float,
    image_height: int | float,
    y_threshold: float = ORDERING_Y_THRESHOLD,
) -> dict[int, int]:
    """Compute the canonical top-left row-major panel ordering.

    Each box is converted to normalized top-left coordinates, sorted by
    ``y_top``, grouped with adjacent boxes whose ``y_top`` difference is at
    most the configured threshold, and sorted within each row by ``x_left``.
    Stable sorting preserves detector order for ties.
    """
    if not boxes:
        return {}
    width = max(float(image_width), 1.0)
    height = max(float(image_height), 1.0)
    indexed: list[tuple[int, dict[str, Any], float, float]] = []
    for box in boxes:
        detector_index = int(box["detector_index"])
        x_left = float(box["x1"]) / width
        y_top = float(box["y1"]) / height
        indexed.append((detector_index, box, x_left, y_top))

    # Sort by y_top; Python sorting is stable for equal values.
    indexed.sort(key=lambda item: item[3])

    rows: list[list[tuple[int, dict[str, Any], float, float]]] = []
    current_row: list[tuple[int, dict[str, Any], float, float]] = []
    for item in indexed:
        if current_row and abs(item[3] - current_row[-1][3]) > float(y_threshold):
            rows.append(sorted(current_row, key=lambda candidate: candidate[2]))
            current_row = []
        current_row.append(item)
    if current_row:
        rows.append(sorted(current_row, key=lambda candidate: candidate[2]))

    flattened = [item for row in rows for item in row]
    return {detector_index: index for index, (detector_index, _, _, _) in enumerate(flattened)}


def spatial_order(
    boxes: list[dict[str, Any]],
    *,
    image_width: int | float | None = None,
    image_height: int | float | None = None,
    y_threshold: float = ORDERING_Y_THRESHOLD,
) -> dict[int, int]:
    """Alias for the canonical row-major ordering."""
    if not boxes:
        return {}
    if image_width is None:
        image_width = max(float(box["x2"]) for box in boxes)
    if image_height is None:
        image_height = max(float(box["y2"]) for box in boxes)
    return historical_row_major_order(
        boxes,
        image_width=image_width,
        image_height=image_height,
        y_threshold=y_threshold,
    )


def _box_records(result) -> list[dict[str, Any]]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().tolist()
    confs = result.boxes.conf.detach().cpu().tolist()
    classes = result.boxes.cls.detach().cpu().tolist()
    names = result.names or {}
    records: list[dict[str, Any]] = []
    for index, (coordinates, confidence, class_id) in enumerate(zip(xyxy, confs, classes)):
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        records.append(
            {
                "detector_index": index,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": float(confidence),
                "class_id": int(class_id),
                "class_name": str(names.get(int(class_id), int(class_id))),
            }
        )
    return records


def _save_crop(result, box: dict[str, Any], path: Path) -> None:
    try:
        from ultralytics.utils.plotting import save_one_box
    except ImportError as exc:
        raise RuntimeError("Could not import ultralytics.utils.plotting.save_one_box") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``Results.save_crop`` uses the same defaults and passes BGR=True because
    # ``result.orig_img`` is a BGR NumPy array.
    import torch

    save_one_box(
        torch.tensor([box["x1"], box["y1"], box["x2"], box["y2"]]),
        result.orig_img.copy(),
        file=path,
        gain=CROP_GAIN,
        pad=CROP_PAD,
        square=CROP_SQUARE,
        BGR=True,
        save=True,
    )


def _panel_record_base(record: dict[str, Any]) -> dict[str, Any]:
    def text(key: str) -> str:
        return str(record.get(key) or "")

    return {
        "pmcid": text("pmcid"),
        "article_version": text("article_version"),
        "resolved_article_version": text("resolved_article_version"),
        "pmid": text("pmid"),
        "figure_id": text("figure_id"),
        "figure_label": text("figure_label"),
        "caption": text("caption"),
        "reference_texts": [str(value) for value in (record.get("reference_texts") or [])],
        "selected_image_href": text("selected_image_href"),
        "parent_image_id": parent_image_id(record),
        "parent_local_image_path": text("local_image_path"),
        "parent_image_width": int(record.get("image_width") or 0),
        "parent_image_height": int(record.get("image_height") or 0),
        "multipanel_score": float(record.get("multipanel_score") or 0.0),
        "multipanel_label": int(record.get("multipanel_label") or 0),
        "is_multipanel": bool(record.get("is_multipanel")),
    }


def _parent_record(
    record: dict[str, Any],
    *,
    panel_count: int,
    status: str,
    error: str,
    model_name: str,
    checkpoint_filename: str,
    confidence: float,
    device: str,
) -> dict[str, Any]:
    return {
        "pmcid": str(record.get("pmcid") or ""),
        "article_version": str(record.get("article_version") or ""),
        "resolved_article_version": str(record.get("resolved_article_version") or ""),
        "pmid": str(record.get("pmid") or ""),
        "figure_id": str(record.get("figure_id") or ""),
        "figure_label": str(record.get("figure_label") or ""),
        "selected_image_href": str(record.get("selected_image_href") or ""),
        "parent_image_id": parent_image_id(record),
        "parent_local_image_path": str(record.get("local_image_path") or ""),
        "multipanel_score": float(record.get("multipanel_score") or 0.0),
        "panel_count": int(panel_count),
        "separation_status": status,
        "separation_error": error,
        "separation_model": model_name,
        "separation_checkpoint": checkpoint_filename,
        "separation_confidence_threshold": float(confidence),
        "separation_device": str(device),
        "crop_method": CROP_METHOD,
    }


def separate_directory(
    classified_dir: str | Path,
    output_dir: str | Path,
    *,
    image_root: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
    checkpoint_filename: str = DEFAULT_CHECKPOINT,
    confidence: float = DEFAULT_CONFIDENCE,
    batch_size: int = 1,
    device: str = "auto",
    max_figures: int | None = None,
    force: bool = False,
) -> Path:
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("confidence must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if max_figures is not None and max_figures < 1:
        raise ValueError("max_figures must be positive")

    import pyarrow.parquet as pq

    classified_dir = Path(classified_dir)
    output_dir = Path(output_dir)
    image_root = Path(image_root) if image_root else classified_dir.parent.parent
    panels_dir = output_dir / "panels"
    parent_manifest_dir = output_dir / "manifests" / "parents"
    panel_manifest_dir = output_dir / "manifests" / "panels"
    summary_path = output_dir / "separation_summary.json"
    panels_dir.mkdir(parents=True, exist_ok=True)
    parent_manifest_dir.mkdir(parents=True, exist_ok=True)
    panel_manifest_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint_path = load_separator(model_name, checkpoint_filename)
    resolved_device = _resolve_device(device)
    display_device = str(resolved_device)
    input_paths = find_parquet_files(classified_dir)

    summary: dict[str, Any] = {
        "model": model_name,
        "checkpoint_filename": checkpoint_filename,
        "checkpoint_path": str(checkpoint_path),
        "confidence": float(confidence),
        "batch_size": int(batch_size),
        "device": display_device,
        "crop_method": CROP_METHOD,
        "crop_gain": CROP_GAIN,
        "crop_pad": CROP_PAD,
        "crop_square": CROP_SQUARE,
        "identifier_convention": IDENTIFIER_CONVENTION,
        "subfigure_index_source": SUBFIGURE_INDEX_SOURCE,
        "ordering_method": ORDERING_METHOD,
        "ordering_y_threshold": ORDERING_Y_THRESHOLD,
        "webdataset_key_assignment": "deferred_to_packaging",
        "classified_dir": str(classified_dir),
        "image_root": str(image_root),
        "input_classified_rows": 0,
        "multipanel_input_rows": 0,
        "processed_parent_rows": 0,
        "parents_with_panels": 0,
        "parents_without_panels": 0,
        "failed_parent_rows": 0,
        "panel_rows": 0,
        "crop_status_counts": {},
        "shards_processed": 0,
    }
    crop_statuses: Counter[str] = Counter()
    remaining = max_figures

    progress = tqdm(input_paths, desc="Separating multi-panel figures", unit="shard")
    for shard_index, input_path in enumerate(progress):
        table = pq.read_table(input_path)
        summary["input_classified_rows"] += table.num_rows
        shard_limit = remaining
        rows = _select_rows(table, shard_limit)
        if remaining is not None:
            remaining -= len(rows)
        summary["multipanel_input_rows"] += len(rows)
        if not rows:
            if remaining == 0:
                break
            continue

        parent_output_path = parent_manifest_dir / f"part-{shard_index:06d}.parquet"
        panel_output_path = panel_manifest_dir / f"part-{shard_index:06d}.parquet"
        if (parent_output_path.exists() or panel_output_path.exists()) and not force:
            raise FileExistsError(
                f"Output shard already exists: {parent_output_path}. Use --force to replace it."
            )

        sources: list[str] = []
        source_rows: list[dict[str, Any]] = []
        parent_records: list[dict[str, Any]] = []
        panel_records: list[dict[str, Any]] = []
        for record in rows:
            image_path = _resolve_image_path(record, image_root)
            if not image_path.exists():
                parent_records.append(
                    _parent_record(
                        record,
                        panel_count=0,
                        status="missing_parent_image",
                        error=f"Image not found: {image_path}",
                        model_name=model_name,
                        checkpoint_filename=checkpoint_filename,
                        confidence=confidence,
                        device=display_device,
                    )
                )
                summary["failed_parent_rows"] += 1
                continue
            sources.append(str(image_path))
            source_rows.append(record)

        try:
            results = model.predict(
                source=sources,
                save=False,
                save_crop=False,
                save_txt=False,
                batch=batch_size,
                conf=confidence,
                device=resolved_device,
                verbose=False,
            ) if sources else []
        except Exception as exc:
            for record in source_rows:
                parent_records.append(
                    _parent_record(
                        record,
                        panel_count=0,
                        status="inference_failed",
                        error=str(exc),
                        model_name=model_name,
                        checkpoint_filename=checkpoint_filename,
                        confidence=confidence,
                        device=display_device,
                    )
                )
            summary["failed_parent_rows"] += len(source_rows)
            results = []
            source_rows = []

        if len(results) != len(source_rows):
            raise RuntimeError(
                f"YOLO returned {len(results)} results for {len(source_rows)} images"
            )

        for record, result in zip(source_rows, results):
            boxes = _box_records(result)
            base = _panel_record_base(record)
            parent_id = base["parent_image_id"]
            parent_width = int(base["parent_image_width"] or result.orig_shape[1])
            parent_height = int(base["parent_image_height"] or result.orig_shape[0])
            order = historical_row_major_order(
                boxes,
                image_width=parent_width,
                image_height=parent_height,
                y_threshold=ORDERING_Y_THRESHOLD,
            )
            successful_panels = 0
            for box in boxes:
                detector_index = int(box["detector_index"])
                historical_order_index = int(order[detector_index])
                # ``spatial_index`` mirrors the canonical subfigure index.
                spatial_index = historical_order_index
                subfigure_index = historical_order_index
                image_id = semantic_image_id(parent_id, subfigure_index)
                relative_path = Path("panels") / parent_id / f"panel-{subfigure_index:04d}.jpg"
                output_path = output_dir / relative_path
                status = "ready"
                error = ""
                width = height = 0
                image_format = ""
                digest = ""
                try:
                    if output_path.exists():
                        if force:
                            output_path.unlink()
                        else:
                            raise FileExistsError(f"Panel crop already exists: {output_path}")
                    _save_crop(result, box, output_path)
                    # save_one_box always writes JPEG and may append a suffix only if
                    # a collision exists. Unique names plus explicit force avoid that.
                    with Image.open(output_path) as panel:
                        panel.seek(0)
                        width, height = panel.size
                        image_format = str(panel.format or "")
                    digest = md5_file(output_path)
                    successful_panels += 1
                except Exception as exc:
                    status = "crop_failed"
                    error = str(exc)
                crop_statuses[status] += 1
                panel_records.append(
                    {
                        **base,
                        "parent_image_width": parent_width,
                        "parent_image_height": parent_height,
                        "source_type": "subfigure",
                        "detector_index": detector_index,
                        "spatial_index": spatial_index,
                        "historical_order_index": historical_order_index,
                        "ordering_method": ORDERING_METHOD,
                        "ordering_y_threshold": ORDERING_Y_THRESHOLD,
                        "subfigure_index": subfigure_index,
                        "image_id": image_id,
                        "panel_image_id": image_id,
                        "identifier_convention": IDENTIFIER_CONVENTION,
                        "panel_class_id": int(box["class_id"]),
                        "panel_class_name": str(box["class_name"]),
                        "panel_confidence": float(box["confidence"]),
                        "box_x1": float(box["x1"]),
                        "box_y1": float(box["y1"]),
                        "box_x2": float(box["x2"]),
                        "box_y2": float(box["y2"]),
                        "box_x1_normalized": float(box["x1"]) / max(parent_width, 1),
                        "box_y1_normalized": float(box["y1"]) / max(parent_height, 1),
                        "box_x2_normalized": float(box["x2"]) / max(parent_width, 1),
                        "box_y2_normalized": float(box["y2"]) / max(parent_height, 1),
                        "local_panel_path": relative_path.as_posix() if status == "ready" else "",
                        "panel_width": int(width),
                        "panel_height": int(height),
                        "panel_format": image_format,
                        "panel_md5": digest,
                        "crop_method": CROP_METHOD,
                        "crop_gain": CROP_GAIN,
                        "crop_pad": CROP_PAD,
                        "crop_square": CROP_SQUARE,
                        "crop_status": status,
                        "crop_error": error,
                        "separation_model": model_name,
                        "separation_checkpoint": checkpoint_filename,
                        "separation_confidence_threshold": float(confidence),
                        "separation_device": display_device,
                    }
                )

            if not boxes:
                parent_status = "no_panels_detected"
                summary["parents_without_panels"] += 1
            elif successful_panels == len(boxes):
                parent_status = "separated"
                summary["parents_with_panels"] += 1
            else:
                parent_status = "partial_crop_failure"
                summary["failed_parent_rows"] += 1
            parent_records.append(
                _parent_record(
                    record,
                    panel_count=len(boxes),
                    status=parent_status,
                    error="" if parent_status in {"separated", "no_panels_detected"} else "One or more crops failed",
                    model_name=model_name,
                    checkpoint_filename=checkpoint_filename,
                    confidence=confidence,
                    device=display_device,
                )
            )

        write_table(parent_output_path, parent_records, parent_schema())
        write_table(panel_output_path, panel_records, panel_schema())
        summary["processed_parent_rows"] += len(parent_records)
        summary["panel_rows"] += len(panel_records)
        summary["shards_processed"] += 1
        summary["crop_status_counts"] = dict(sorted(crop_statuses.items()))
        write_json(summary_path, summary)
        progress.set_postfix(
            parents=summary["processed_parent_rows"],
            panels=summary["panel_rows"],
            failed=summary["failed_parent_rows"],
        )
        if remaining == 0:
            break

    write_json(summary_path, summary)
    return summary_path
