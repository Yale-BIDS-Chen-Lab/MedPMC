"""Readers for NLM PMC bulk package file lists."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterator

from .licenses import normalize_license

_PMID_RE = re.compile(r"\bPMC\d+\b", flags=re.IGNORECASE)


def normalize_pmcid(value: object) -> str | None:
    if value is None:
        return None
    match = _PMID_RE.search(str(value))
    if match:
        return match.group(0).upper()

    digits = re.sub(r"\D", "", str(value))
    return f"PMC{digits}" if digits else None


def _header_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _detect_header(path: Path, delimiter: str) -> tuple[int, list[str]] | None:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for index, row in enumerate(reader):
            normalized = {_header_key(cell) for cell in row}
            has_id = any(key in normalized for key in {"accessionid", "pmcid", "accession"})
            has_license = "license" in normalized or "licensetype" in normalized
            if has_id and has_license:
                return index, row
            if index >= 20:
                break
    return None


def _first(record: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    normalized = {_header_key(key): value for key, value in record.items()}
    for alias in aliases:
        value = normalized.get(_header_key(alias))
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def iter_bulk_filelist(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield normalized metadata rows from an NLM CSV or tab-separated file list."""
    path = Path(path)
    delimiter = "\t" if path.suffix.lower() == ".txt" else ","
    detected = _detect_header(path, delimiter)

    if detected is None:
        raise ValueError(
            f"Could not locate a file-list header containing PMCID/accession and license: {path}"
        )

    header_index, _ = detected
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for _ in range(header_index):
            next(handle, None)

        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            pmcid = normalize_pmcid(
                _first(row, ("Accession ID", "PMCID", "Accession", "Article File", "File"))
            )
            if not pmcid:
                # Some lists encode the PMCID only in the article path.
                pmcid = normalize_pmcid(" ".join(str(value) for value in row.values()))
            if not pmcid:
                continue

            retracted_value = _first(row, ("Retracted", "Is Retracted")) or "no"
            yield {
                "pmcid": pmcid,
                "license": normalize_license(
                    _first(row, ("License", "License Type", "license_code"))
                ),
                "retracted": retracted_value.strip().lower() in {"yes", "true", "1"},
                "article_file": _first(
                    row, ("Article File", "File", "Path", "Key", "Article Path")
                ),
                "citation": _first(row, ("Article Citation", "Citation")),
                "pmid": _first(row, ("PMID", "PubMed ID")),
                "last_updated": _first(
                    row, ("Last Updated", "Last Updated UTC (YYYY-MM-DD HH:MM:SS)")
                ),
            }


def load_bulk_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    return {record["pmcid"]: record for record in iter_bulk_filelist(path)}
