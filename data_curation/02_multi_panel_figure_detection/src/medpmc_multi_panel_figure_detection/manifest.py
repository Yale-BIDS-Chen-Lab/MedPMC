"""Prepare figure images and image-level manifests from Stage 1 retained rows."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tqdm import tqdm

from .aws import (
    DEFAULT_BUCKET,
    create_unsigned_s3_client,
    download_s3_object,
    match_all_media_urls,
    match_media_url,
    media_basename,
    parse_s3_url,
    resolve_article_metadata,
)
from .images import inspect_image, is_supported_image_path
from .storage import IMAGE_MANIFEST_SCHEMA, normalize_stage1_record, write_json, write_table


def _select_primary_image_match(
    hrefs: list[str],
    media_urls: list[str],
):
    """Select the first JATS graphic in document order.

    Returns ``(primary_href, primary_match, matched_candidate_count)``. Later
    graphics are counted for provenance but never substituted for the primary
    graphic.
    """
    normalized_hrefs = [str(value) for value in hrefs if str(value or "").strip()]
    if not normalized_hrefs:
        return "", None, 0
    primary_href = normalized_hrefs[0]
    all_matches = match_all_media_urls(normalized_hrefs, media_urls)
    return primary_href, match_media_url([primary_href], media_urls), len(all_matches)


def _empty_image_fields() -> dict[str, Any]:
    return {
        "resolved_article_version": "",
        "article_resolution_method": "",
        "article_candidate_versions": 0,
        "article_metadata_key": "",
        "article_is_retracted": None,
        "article_metadata_license": "",
        "selected_image_href": "",
        "image_selection_method": "",
        "image_candidate_count": 0,
        "image_href_match_type": "",
        "image_s3_url": "",
        "image_s3_key": "",
        "image_expected_md5": "",
        "local_image_path": "",
        "image_status": "",
        "image_transfer": "",
        "image_error": "",
        "image_width": None,
        "image_height": None,
        "image_format": "",
        "image_mode": "",
        "image_md5": "",
    }


def _error_record(record: dict[str, Any], status: str, error: str, **extra) -> dict[str, Any]:
    return {
        **normalize_stage1_record(record),
        **_empty_image_fields(),
        **extra,
        "image_status": status,
        "image_error": error,
    }


def _prepare_article(
    records: list[dict[str, Any]],
    *,
    output_dir: Path,
    bucket: str,
    region_name: str,
    force: bool,
    client,
) -> list[dict[str, Any]]:
    first = records[0]
    pmcid = str(first.get("pmcid") or "")
    explicit_versions = [
        str(record.get("article_version") or "")
        for record in records
        if str(record.get("article_version") or "").strip()
    ]
    article_version = explicit_versions[0] if explicit_versions else ""

    try:
        resolved = resolve_article_metadata(
            client,
            bucket=bucket,
            pmcid=pmcid,
            article_version=article_version,
            records=records,
        )
    except Exception as exc:
        return [
            _error_record(record, "metadata_error", str(exc))
            for record in records
        ]

    if not resolved.metadata:
        return [
            _error_record(
                record,
                "metadata_not_found",
                f"PMC metadata not found for {pmcid}",
                article_resolution_method=resolved.resolution_method,
            )
            for record in records
        ]

    metadata = resolved.metadata
    media_urls = metadata.get("media_urls") or []
    article_fields = {
        "resolved_article_version": resolved.versioned_pmcid,
        "article_resolution_method": resolved.resolution_method,
        "article_candidate_versions": resolved.candidate_versions,
        "article_metadata_key": resolved.metadata_key,
        "article_is_retracted": bool(metadata.get("is_retracted", False)),
        "article_metadata_license": str(metadata.get("license_code") or ""),
    }

    prepared: list[dict[str, Any]] = []
    for record in records:
        base = {**normalize_stage1_record(record), **_empty_image_fields(), **article_fields}
        hrefs = [str(value) for value in (record.get("image_hrefs") or []) if str(value)]
        if not hrefs:
            prepared.append(
                {**base, "image_status": "missing_image_href", "image_error": "No image href extracted from JATS"}
            )
            continue

        # Select the first JATS graphic in XML document order as the primary
        # figure image rather than substituting an alternative rendering.
        primary_href, match, candidate_count = _select_primary_image_match(
            hrefs, media_urls
        )
        selection_method = "first_graphic_document_order"

        if match is None:
            prepared.append(
                {
                    **base,
                    "selected_image_href": primary_href,
                    "image_selection_method": selection_method,
                    "image_candidate_count": candidate_count,
                    "image_status": "missing_from_metadata",
                    "image_error": (
                        "The first JATS graphic did not match a PMC media URL; "
                        "later graphic candidates were not substituted"
                    ),
                }
            )
            continue

        try:
            ref = parse_s3_url(match.media_url)
            filename = media_basename(match.media_url)
            relative_path = Path("images") / resolved.versioned_pmcid / filename
            destination = output_dir / relative_path
            if not is_supported_image_path(destination):
                raise ValueError(f"Unsupported image format: {filename}")
            transfer, actual_md5 = download_s3_object(
                client, ref, destination, force=force
            )
            inspection = inspect_image(destination)
        except Exception as exc:
            prepared.append(
                {
                    **base,
                    "selected_image_href": primary_href,
                    "image_selection_method": selection_method,
                    "image_candidate_count": candidate_count,
                    "image_href_match_type": match.match_type,
                    "image_s3_url": match.media_url,
                    "image_status": "download_or_decode_failed",
                    "image_error": str(exc),
                }
            )
            continue

        prepared.append(
            {
                **base,
                "selected_image_href": primary_href,
                "image_selection_method": selection_method,
                "image_candidate_count": candidate_count,
                "image_href_match_type": match.match_type,
                "image_s3_url": match.media_url,
                "image_s3_key": ref.key,
                "image_expected_md5": ref.expected_md5,
                "local_image_path": relative_path.as_posix(),
                "image_status": "ready",
                "image_transfer": transfer,
                "image_error": "",
                "image_width": inspection.width,
                "image_height": inspection.height,
                "image_format": inspection.format,
                "image_mode": inspection.mode,
                "image_md5": actual_md5,
            }
        )
    return prepared


def _group_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("pmcid") or "")].append(record)
    return list(grouped.values())


def prepare_images(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 16,
    max_figures: int | None = None,
    bucket: str = DEFAULT_BUCKET,
    region_name: str = "us-east-1",
    force: bool = False,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if max_figures is not None and max_figures < 1:
        raise ValueError("max_figures must be positive")

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    manifest_dir = output_dir / "manifests" / "figures"
    summary_path = output_dir / "image_preparation_summary.json"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(input_dir.glob("part-*.parquet"))
    if not input_paths:
        raise FileNotFoundError(f"No part-*.parquet files found under {input_dir}")

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "bucket": bucket,
        "region": region_name,
        "workers": workers,
        "max_figures": max_figures,
        "image_selection_method": "first_graphic_document_order",
        "input_rows": 0,
        "output_rows": 0,
        "shards_processed": 0,
        "status_counts": {},
        "transfer_counts": {},
    }
    status_counts: Counter[str] = Counter()
    transfer_counts: Counter[str] = Counter()
    remaining = max_figures

    shard_progress = tqdm(input_paths, desc="Preparing retained figure images", unit="shard")
    for input_path in shard_progress:
        if remaining is not None and remaining <= 0:
            break
        output_path = manifest_dir / input_path.name
        records = [normalize_stage1_record(row) for row in pq.read_table(input_path).to_pylist()]
        records = [record for record in records if bool(record.get("retained", True))]
        if output_path.exists() and not force and max_figures is None:
            schema_names = set(pq.read_schema(output_path).names)
            required_resume_columns = {
                "image_status",
                "image_transfer",
                "image_selection_method",
            }
            if required_resume_columns.issubset(schema_names):
                table = pq.read_table(
                    output_path,
                    columns=[
                        "image_status",
                        "image_transfer",
                        "image_selection_method",
                    ],
                )
                rows = table.to_pylist()
                compatible = all(
                    str(row.get("image_selection_method") or "")
                    == "first_graphic_document_order"
                    for row in rows
                )
                if len(rows) == len(records) and compatible:
                    summary["input_rows"] += len(rows)
                    summary["output_rows"] += len(rows)
                    summary["shards_processed"] += 1
                    status_counts.update(
                        str(row.get("image_status") or "") for row in rows
                    )
                    transfer_counts.update(
                        str(row.get("image_transfer") or "")
                        for row in rows
                        if row.get("image_transfer")
                    )
                    continue

        if remaining is not None:
            records = records[:remaining]
            remaining -= len(records)
        groups = _group_records(records)

        output_rows: list[dict[str, Any]] = []
        client = create_unsigned_s3_client(
            region_name=region_name, max_pool_connections=max(32, workers * 2)
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _prepare_article,
                    group,
                    output_dir=output_dir,
                    bucket=bucket,
                    region_name=region_name,
                    force=force,
                    client=client,
                )
                for group in groups
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Articles in {input_path.name}",
                leave=False,
                unit="article",
            ):
                output_rows.extend(future.result())

        output_rows.sort(key=lambda row: (str(row.get("pmcid")), str(row.get("figure_id"))))
        write_table(output_path, output_rows, IMAGE_MANIFEST_SCHEMA)
        summary["input_rows"] += len(records)
        summary["output_rows"] += len(output_rows)
        summary["shards_processed"] += 1
        status_counts.update(str(row.get("image_status") or "") for row in output_rows)
        transfer_counts.update(str(row.get("image_transfer") or "") for row in output_rows if row.get("image_transfer"))
        shard_progress.set_postfix(ready=status_counts.get("ready", 0), rows=summary["output_rows"])

    summary["status_counts"] = dict(sorted(status_counts.items()))
    summary["transfer_counts"] = dict(sorted(transfer_counts.items()))
    write_json(summary_path, summary)
    return summary_path
