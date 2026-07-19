"""Extract figure captions and inline reference text from PMC XML."""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .filelist import load_bulk_metadata, normalize_pmcid
from .jats import parse_article_xml
from .licenses import DEFAULT_ALLOWED_LICENSES, license_is_allowed, normalize_license
from .storage import EXTRACTION_SCHEMA, ParquetShardWriter, write_json

_PMCID_RE = re.compile(r"\bPMC\d+\b", flags=re.IGNORECASE)


def _pmcid_from_name(name: str) -> str | None:
    match = _PMCID_RE.search(name)
    return match.group(0).upper() if match else None


def _metadata_from_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    article_version = f"{value.get('pmcid', '')}.{value.get('version', '')}".strip(".")
    return {
        **value,
        "article_version": article_version,
        "license": normalize_license(value.get("license_code")),
    }


def extract_from_tar(
    xml_tar: str | Path,
    filelist: str | Path,
    output_dir: str | Path,
    *,
    allowed_licenses=DEFAULT_ALLOWED_LICENSES,
    shard_size: int = 10_000,
    max_articles: int | None = None,
    force: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    summary_path = output_dir / "extraction_summary.json"
    if summary_path.exists() and not force:
        return summary_path
    if force and output_dir.exists():
        for path in output_dir.glob("part-*.parquet"):
            path.unlink()

    bulk_metadata = load_bulk_metadata(filelist)
    writer = ParquetShardWriter(
        output_dir,
        schema=EXTRACTION_SCHEMA,
        shard_size=shard_size,
    )

    summary = {
        "source": str(xml_tar),
        "articles_seen": 0,
        "articles_parsed": 0,
        "articles_skipped_license": 0,
        "articles_skipped_retracted": 0,
        "articles_failed": 0,
        "figures_extracted": 0,
    }

    with tarfile.open(xml_tar, mode="r|gz") as archive:
        for member in tqdm(archive, desc="Streaming PMC XML archive"):
            if not member.isfile() or not member.name.lower().endswith(".xml"):
                continue
            if max_articles is not None and summary["articles_seen"] >= max_articles:
                break

            summary["articles_seen"] += 1
            pmcid = _pmcid_from_name(member.name)
            metadata = bulk_metadata.get(pmcid or "", {})
            license_code = normalize_license(metadata.get("license"))

            if metadata.get("retracted"):
                summary["articles_skipped_retracted"] += 1
                continue
            if license_code and not license_is_allowed(license_code, allowed_licenses):
                summary["articles_skipped_license"] += 1
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                summary["articles_failed"] += 1
                continue

            try:
                xml_bytes = extracted.read()
                records = parse_article_xml(
                    xml_bytes,
                    source_xml=member.name,
                    metadata={
                        **metadata,
                        "pmcid": pmcid or metadata.get("pmcid"),
                        "license": license_code,
                    },
                )
                effective_license = license_code
                if records and not effective_license:
                    effective_license = normalize_license(records[0].get("license"))
                if not license_is_allowed(effective_license, allowed_licenses):
                    summary["articles_skipped_license"] += 1
                    continue

                writer.write_many(records)
                summary["articles_parsed"] += 1
                summary["figures_extracted"] += len(records)
            except Exception:
                summary["articles_failed"] += 1

    writer.close()
    write_json(summary_path, summary)
    return summary_path


def extract_from_xml_dir(
    xml_dir: str | Path,
    output_dir: str | Path,
    *,
    metadata_dir: str | Path | None = None,
    allowed_licenses=DEFAULT_ALLOWED_LICENSES,
    shard_size: int = 10_000,
    max_articles: int | None = None,
    force: bool = False,
) -> Path:
    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)
    metadata_dir = Path(metadata_dir) if metadata_dir else None
    summary_path = output_dir / "extraction_summary.json"

    if summary_path.exists() and not force:
        return summary_path
    if force and output_dir.exists():
        for path in output_dir.glob("part-*.parquet"):
            path.unlink()

    writer = ParquetShardWriter(
        output_dir,
        schema=EXTRACTION_SCHEMA,
        shard_size=shard_size,
    )
    xml_paths = sorted(xml_dir.rglob("*.xml"))
    if max_articles is not None:
        xml_paths = xml_paths[:max_articles]

    summary = {
        "source": str(xml_dir),
        "articles_seen": 0,
        "articles_parsed": 0,
        "articles_skipped_license": 0,
        "articles_skipped_retracted": 0,
        "articles_failed": 0,
        "figures_extracted": 0,
    }

    for xml_path in tqdm(xml_paths, desc="Parsing PMC XML"):
        summary["articles_seen"] += 1
        metadata: dict[str, Any] = {}

        if metadata_dir is not None:
            metadata_path = metadata_dir / f"{xml_path.stem}.json"
            if metadata_path.exists():
                try:
                    metadata = _metadata_from_json(metadata_path)
                except Exception:
                    metadata = {}

        if metadata.get("is_retracted"):
            summary["articles_skipped_retracted"] += 1
            continue

        try:
            records = parse_article_xml(
                xml_path,
                source_xml=str(xml_path),
                metadata=metadata,
            )
            license_code = (
                normalize_license(metadata.get("license"))
                or (normalize_license(records[0].get("license")) if records else None)
            )
            if not license_is_allowed(license_code, allowed_licenses):
                summary["articles_skipped_license"] += 1
                continue

            writer.write_many(records)
            summary["articles_parsed"] += 1
            summary["figures_extracted"] += len(records)
        except Exception:
            summary["articles_failed"] += 1

    writer.close()
    write_json(summary_path, summary)
    return summary_path
