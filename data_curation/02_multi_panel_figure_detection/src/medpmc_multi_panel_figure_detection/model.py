"""ViT multi-panel detector inference."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm

from .storage import DETECTION_SCHEMA, write_json, write_table

DEFAULT_MODEL = "Yale-BIDS-Chen/medpmc-multi-fig-detection-vit"
DEFAULT_CHECKPOINT_FILENAME = "model.pth.tar"


def _device_name(requested: str) -> str:
    import torch

    value = requested.strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA was requested with --device {requested}, but "
            "torch.cuda.is_available() is False"
        )
    return requested


def _torch_load(path: str | Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_detector(model_name: str, checkpoint_filename: str, device: str):
    try:
        import timm
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Inference dependencies are missing. Install with: "
            "pip install -e '.[inference]'"
        ) from exc

    checkpoint_path = hf_hub_download(
        repo_id=model_name,
        filename=checkpoint_filename,
    )
    checkpoint = _torch_load(checkpoint_path)
    if (
        not isinstance(checkpoint, dict)
        or "arch" not in checkpoint
        or "state_dict" not in checkpoint
    ):
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} must contain 'arch' and 'state_dict'"
        )

    architecture = str(checkpoint["arch"])
    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["state_dict"].items()
    }
    model = timm.create_model(
        architecture,
        pretrained=False,
        num_classes=2,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, architecture, checkpoint_path


def _checkpoint_image_size(model) -> int:
    """Return the fixed square input size required by the loaded model."""
    patch_embed = getattr(model, "patch_embed", None)
    value = getattr(patch_embed, "img_size", None)
    if value is None:
        config = getattr(model, "pretrained_cfg", None) or {}
        input_size = config.get("input_size")
        if input_size and len(input_size) >= 3:
            value = input_size[-2:]

    if isinstance(value, int):
        height = width = int(value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
    else:
        raise RuntimeError(
            "Could not determine the detector input size from the loaded checkpoint. "
            "Pass --image-size explicitly."
        )

    if height != width:
        raise RuntimeError(
            f"The loaded detector expects a non-square input ({height}x{width}), "
            "which this pipeline does not currently support."
        )
    return height


def _resolve_image_size(model, requested: int | None) -> int:
    checkpoint_size = _checkpoint_image_size(model)
    if requested is None:
        return checkpoint_size
    if requested < 1:
        raise ValueError("image_size must be at least 1")
    if int(requested) != checkpoint_size:
        raise ValueError(
            f"--image-size {requested} does not match the loaded checkpoint, "
            f"which expects {checkpoint_size}x{checkpoint_size} inputs. "
            "Omit --image-size to detect it automatically."
        )
    return checkpoint_size


def _resolve_image_path(record: dict[str, Any], image_root: Path) -> Path:
    path = Path(str(record.get("local_image_path") or ""))
    return path if path.is_absolute() else image_root / path


def _resolve_timm_data_config(model, requested_image_size: int | None) -> dict[str, Any]:
    """Resolve evaluation preprocessing from the loaded timm model.

    The returned configuration includes the model-derived input size, crop,
    interpolation, mean, and standard deviation used by the manifest-backed
    dataset.
    """
    from timm.data import resolve_data_config

    checkpoint_size = _checkpoint_image_size(model)
    overrides: dict[str, Any] = {}
    if requested_image_size is not None:
        if requested_image_size < 1:
            raise ValueError("image_size must be at least 1")
        if int(requested_image_size) != checkpoint_size:
            raise ValueError(
                f"--image-size {requested_image_size} does not match the loaded "
                f"checkpoint, which expects {checkpoint_size}x{checkpoint_size} "
                "inputs. Omit --image-size to use the model configuration."
            )
        overrides["input_size"] = (3, checkpoint_size, checkpoint_size)

    config = dict(resolve_data_config(overrides, model=model))
    input_size = tuple(int(value) for value in config.get("input_size", ()))
    if len(input_size) != 3:
        raise RuntimeError(
            f"timm returned an invalid input_size in data config: {input_size!r}"
        )
    if input_size[-2:] != (checkpoint_size, checkpoint_size):
        raise RuntimeError(
            "The timm data configuration and loaded checkpoint disagree: "
            f"data config expects {input_size[-2]}x{input_size[-1]}, while the "
            f"checkpoint patch embedding expects {checkpoint_size}x{checkpoint_size}."
        )
    config["input_size"] = input_size
    return config


def _build_preprocess(data_config: dict[str, Any]):
    from timm.data import create_transform

    return create_transform(
        **data_config,
        is_training=False,
    )


def _jsonable_data_config(data_config: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key in (
        "input_size",
        "interpolation",
        "mean",
        "std",
        "crop_pct",
        "crop_mode",
    ):
        value = data_config.get(key)
        if isinstance(value, tuple):
            value = list(value)
        output[key] = value
    return output


class _PreparedImageDataset:
    def __init__(
        self,
        items: list[tuple[int, dict[str, Any]]],
        *,
        image_root: Path,
        preprocess,
    ) -> None:
        self.items = items
        self.image_root = image_root
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        from PIL import Image

        Image.MAX_IMAGE_PIXELS = None
        output_index, record = self.items[index]
        image_path = _resolve_image_path(record, self.image_root)
        try:
            with Image.open(image_path) as image:
                image.seek(0)
                tensor = self.preprocess(image.convert("RGB"))
            return output_index, tensor, ""
        except Exception as exc:
            return output_index, None, str(exc)


def _collate_images(batch):
    import torch

    valid_indices: list[int] = []
    tensors = []
    errors: list[tuple[int, str]] = []
    for output_index, tensor, error in batch:
        if tensor is None:
            errors.append((output_index, error))
        else:
            valid_indices.append(output_index)
            tensors.append(tensor)
    stacked = torch.stack(tensors) if tensors else None
    return valid_indices, stacked, errors


def _predict_batch(
    model,
    inputs,
    *,
    device: str,
    positive_label: int,
    amp: bool,
):
    import torch

    inputs = inputs.to(
        device,
        non_blocking=device.startswith("cuda"),
    )
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp and device.startswith("cuda")
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities[:, positive_label].detach().cpu().tolist()


def _label_from_positive_score(
    score: float,
    *,
    threshold: float,
    positive_label: int,
) -> int:
    # For a two-class softmax at the default threshold, class 1 is selected
    # when p(class 1) > 0.5; an exact tie resolves to class 0.
    is_positive = float(score) > float(threshold)
    return int(positive_label if is_positive else 1 - positive_label)


def _existing_summary_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    existing = json.loads(path.read_text(encoding="utf-8"))
    return all(existing.get(key) == value for key, value in expected.items())


def _manifest_selection_methods(input_paths: list[Path]) -> list[str]:
    methods: set[str] = set()
    for path in input_paths:
        schema_names = set(pq.read_schema(path).names)
        if "image_selection_method" not in schema_names:
            methods.add("unrecorded")
            continue
        table = pq.read_table(path, columns=["image_selection_method"])
        methods.update(
            str(row.get("image_selection_method") or "unrecorded")
            for row in table.to_pylist()
        )
    return sorted(methods)


def detect_directory(
    manifest_dir: str | Path,
    output_dir: str | Path,
    *,
    image_root: str | Path | None = None,
    model_name: str = DEFAULT_MODEL,
    checkpoint_filename: str = DEFAULT_CHECKPOINT_FILENAME,
    batch_size: int = 256,
    loader_workers: int = 4,
    threshold: float = 0.5,
    positive_label: int = 1,
    image_size: int | None = None,
    device: str = "auto",
    amp: bool = False,
    force: bool = False,
) -> Path:
    import torch
    from torch.utils.data import DataLoader

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if loader_workers < 0:
        raise ValueError("loader_workers must be non-negative")
    if positive_label not in {0, 1}:
        raise ValueError("positive_label must be 0 or 1")

    manifest_dir = Path(manifest_dir)
    output_dir = Path(output_dir)
    inferred_root = (
        manifest_dir.parent.parent
        if manifest_dir.name == "figures"
        else manifest_dir.parent
    )
    image_root = Path(image_root) if image_root is not None else inferred_root
    selected_device = _device_name(device)
    if selected_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    classified_dir = output_dir / "classified"
    multipanel_dir = output_dir / "multipanel"
    singlepanel_dir = output_dir / "singlepanel"
    for directory in (classified_dir, multipanel_dir, singlepanel_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "detection_summary.json"

    input_paths = sorted(manifest_dir.glob("part-*.parquet"))
    if not input_paths:
        raise FileNotFoundError(
            f"No part-*.parquet files found under {manifest_dir}"
        )
    image_selection_methods = _manifest_selection_methods(input_paths)

    model, architecture, checkpoint_path = load_detector(
        model_name,
        checkpoint_filename,
        selected_device,
    )
    data_config = _resolve_timm_data_config(model, image_size)
    selected_image_size = int(data_config["input_size"][-1])
    preprocess = _build_preprocess(data_config)

    expected = {
        "model": model_name,
        "checkpoint_filename": checkpoint_filename,
        "batch_size": batch_size,
        "loader_workers": loader_workers,
        "threshold": threshold,
        "positive_label": positive_label,
        "image_size": selected_image_size,
        "device": selected_device,
        "amp": bool(amp),
        "timm_data_config": _jsonable_data_config(data_config),
        "image_selection_methods": image_selection_methods,
    }
    if not force and _existing_summary_matches(summary_path, expected):
        return summary_path

    print(
        f"Detection configuration: device={selected_device}, "
        f"batch_size={batch_size}, loader_workers={loader_workers}, "
        f"image_size={selected_image_size} (timm/checkpoint), amp={amp}"
    )

    summary: dict[str, Any] = {
        **expected,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
        "image_root": str(image_root),
        "input_rows": 0,
        "ready_rows": 0,
        "classified_rows": 0,
        "multipanel_rows": 0,
        "singlepanel_rows": 0,
        "failed_rows": 0,
        "shards_processed": 0,
    }

    shard_progress = tqdm(
        input_paths,
        desc="Detecting multi-panel figures",
        unit="shard",
    )
    for input_path in shard_progress:
        classified_path = classified_dir / input_path.name
        multipanel_path = multipanel_dir / input_path.name
        singlepanel_path = singlepanel_dir / input_path.name
        records = pq.read_table(input_path).to_pylist()

        output_rows: list[dict[str, Any]] = []
        ready_items: list[tuple[int, dict[str, Any]]] = []
        for record in records:
            output = {
                **record,
                "multipanel_score": None,
                "multipanel_label": None,
                "is_multipanel": None,
                "detection_status": "",
                "detection_error": "",
                "detection_model": model_name,
                "detection_architecture": architecture,
                "detection_threshold": float(threshold),
                "detection_image_size": int(selected_image_size),
                "detection_positive_label": int(positive_label),
            }
            output_rows.append(output)
            if record.get("image_status") != "ready":
                output["detection_status"] = "skipped_image_not_ready"
                output["detection_error"] = str(record.get("image_error") or "")
            else:
                ready_items.append((len(output_rows) - 1, record))

        dataset = _PreparedImageDataset(
            ready_items,
            image_root=image_root,
            preprocess=preprocess,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=loader_workers,
            pin_memory=selected_device.startswith("cuda"),
            persistent_workers=loader_workers > 0,
            collate_fn=_collate_images,
        )
        for valid_indices, inputs, errors in tqdm(
            loader,
            desc=f"Inference {input_path.name}",
            leave=False,
            unit="batch",
        ):
            for output_index, error in errors:
                output_rows[output_index]["detection_status"] = "decode_failed"
                output_rows[output_index]["detection_error"] = error
            if inputs is None:
                continue
            scores = _predict_batch(
                model,
                inputs,
                device=selected_device,
                positive_label=positive_label,
                amp=amp,
            )
            for output_index, score in zip(valid_indices, scores, strict=True):
                label = _label_from_positive_score(
                    score,
                    threshold=threshold,
                    positive_label=positive_label,
                )
                output_rows[output_index].update(
                    {
                        "multipanel_score": float(score),
                        "multipanel_label": int(label),
                        "is_multipanel": bool(label == 1),
                        "detection_status": "classified",
                    }
                )

        multipanel_rows = [
            row for row in output_rows if row.get("is_multipanel") is True
        ]
        singlepanel_rows = [
            row for row in output_rows if row.get("is_multipanel") is False
        ]
        write_table(classified_path, output_rows, DETECTION_SCHEMA)
        write_table(multipanel_path, multipanel_rows, DETECTION_SCHEMA)
        write_table(singlepanel_path, singlepanel_rows, DETECTION_SCHEMA)

        summary["input_rows"] += len(records)
        summary["ready_rows"] += len(ready_items)
        summary["classified_rows"] += len(multipanel_rows) + len(singlepanel_rows)
        summary["multipanel_rows"] += len(multipanel_rows)
        summary["singlepanel_rows"] += len(singlepanel_rows)
        summary["failed_rows"] += sum(
            row.get("detection_status") != "classified" for row in output_rows
        )
        summary["shards_processed"] += 1
        shard_progress.set_postfix(
            classified=summary["classified_rows"],
            multipanel=summary["multipanel_rows"],
        )

    write_json(summary_path, summary)
    return summary_path
