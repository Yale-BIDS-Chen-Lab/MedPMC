"""Pure record normalization helpers for Stage 5."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

IMAGE_EXTENSIONS = (
    ".jpeg", ".jpg", ".png", ".tiff", ".tif", ".webp", ".gif", ".bmp"
)


def _text(record: dict[str, Any], key: str) -> str:
    return str(record.get(key) or "")


def normalize_graphic_id(value: Any) -> str:
    text = unquote(str(value or "")).split("?", 1)[0].split("#", 1)[0]
    text = PurePosixPath(text).name
    lowered = text.casefold()
    for extension in IMAGE_EXTENSIONS:
        if lowered.endswith(extension):
            text = text[: -len(extension)]
            break
    return text


def singlepanel_image_id(record: dict[str, Any]) -> str:
    pmcid = _text(record, "pmcid")
    graphic = normalize_graphic_id(
        record.get("selected_image_href")
        or record.get("local_image_path")
        or record.get("figure_id")
    )
    suffix = graphic or _text(record, "figure_id") or "figure"
    return f"{pmcid}_{suffix}" if pmcid else suffix


def singlepanel_is_eligible(record: dict[str, Any]) -> bool:
    return (
        str(record.get("image_status") or "") == "ready"
        and str(record.get("detection_status") or "") == "classified"
        and record.get("is_multipanel") is False
        and bool(str(record.get("local_image_path") or ""))
    )


def subfigure_is_eligible(record: dict[str, Any]) -> bool:
    return (
        str(record.get("crop_status") or "") == "ready"
        and str(record.get("caption_alignment_status") or "") == "aligned"
        and bool(str(record.get("local_panel_path") or ""))
        and bool(str(record.get("subcaption") or "").strip())
    )


def candidate_from_singlepanel(record: dict[str, Any]) -> dict[str, Any]:
    image_id = singlepanel_image_id(record)
    caption = _text(record, "caption")
    return {
        "source_type": "single_panel",
        "pmcid": _text(record, "pmcid"),
        "article_version": _text(record, "article_version"),
        "resolved_article_version": _text(record, "resolved_article_version"),
        "pmid": _text(record, "pmid"),
        "article_title": _text(record, "article_title"),
        "journal_title": _text(record, "journal_title"),
        "license": _text(record, "license"),
        "figure_id": _text(record, "figure_id"),
        "figure_label": _text(record, "figure_label"),
        "image_id": image_id,
        "parent_image_id": image_id,
        "subfigure_index": None,
        "local_image_path": _text(record, "local_image_path"),
        "selected_image_href": _text(record, "selected_image_href"),
        "caption": caption,
        "main_caption": caption,
        "reference_texts": list(record.get("reference_texts") or []),
        "upstream_status": "singlepanel_classified",
        "upstream_model": _text(record, "detection_model"),
        "upstream_score": (
            float(record["multipanel_score"])
            if record.get("multipanel_score") is not None
            else None
        ),
    }


def candidate_from_subfigure(record: dict[str, Any]) -> dict[str, Any]:
    image_id = _text(record, "image_id") or _text(record, "panel_image_id")
    parent_id = _text(record, "parent_image_id")
    index = record.get("subfigure_index")
    return {
        "source_type": "subfigure",
        "pmcid": _text(record, "pmcid"),
        "article_version": _text(record, "article_version"),
        "resolved_article_version": _text(record, "resolved_article_version"),
        "pmid": _text(record, "pmid"),
        "article_title": _text(record, "article_title"),
        "journal_title": _text(record, "journal_title"),
        "license": _text(record, "license"),
        "figure_id": _text(record, "figure_id"),
        "figure_label": _text(record, "figure_label"),
        "image_id": image_id,
        "parent_image_id": parent_id,
        "subfigure_index": int(index) if index is not None else None,
        "local_image_path": _text(record, "local_panel_path"),
        "selected_image_href": _text(record, "selected_image_href"),
        "caption": _text(record, "subcaption"),
        "main_caption": _text(record, "main_caption") or _text(record, "caption"),
        "reference_texts": list(record.get("reference_texts") or []),
        "upstream_status": "caption_aligned",
        "upstream_model": _text(record, "caption_model"),
        "upstream_score": (
            float(record["panel_confidence"])
            if record.get("panel_confidence") is not None
            else None
        ),
    }


def resolve_candidate_path(
    candidate: dict[str, Any],
    *,
    singlepanel_image_root: Path | None,
    subfigure_image_root: Path | None,
) -> Path:
    path = Path(str(candidate.get("local_image_path") or ""))
    if path.is_absolute():
        return path
    source_type = str(candidate.get("source_type") or "")
    root = singlepanel_image_root if source_type == "single_panel" else subfigure_image_root
    if root is None:
        raise ValueError(f"No image root was provided for source_type={source_type!r}")
    return root / path


def predicted_label_from_score(
    score: float,
    *,
    threshold: float = 0.5,
    medical_label: int = 1,
) -> int:
    if medical_label not in {0, 1}:
        raise ValueError("medical_label must be 0 or 1")
    return medical_label if float(score) > float(threshold) else 1 - medical_label
