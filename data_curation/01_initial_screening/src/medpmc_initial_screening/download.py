"""PMC bulk and selected-PMCID acquisition utilities."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

from .filelist import normalize_pmcid
from .licenses import DEFAULT_ALLOWED_LICENSES, license_is_allowed, normalize_license

PMC_BUCKET = "pmc-oa-opendata"


def download_http(
    url: str,
    destination: str | Path,
    *,
    overwrite: bool = False,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download an HTTP(S) file with simple resume support."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not overwrite:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}

    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        if response.status_code == 416 and partial.exists():
            partial.replace(destination)
            return destination
        response.raise_for_status()

        if existing and response.status_code != 206:
            partial.unlink(missing_ok=True)
            existing = 0

        total = response.headers.get("Content-Length")
        total_bytes = int(total) + existing if total is not None else None
        mode = "ab" if existing else "wb"

        with partial.open(mode) as handle, tqdm(
            total=total_bytes,
            initial=existing,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))

    partial.replace(destination)
    return destination


def download_bulk_assets(
    archive_url: str,
    filelist_url: str,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archive_name = Path(urlparse(archive_url).path).name
    filelist_name = Path(urlparse(filelist_url).path).name
    if not archive_name or not filelist_name:
        raise ValueError("Could not determine filenames from the supplied URLs.")

    archive_path = download_http(
        archive_url, output_dir / archive_name, overwrite=overwrite
    )
    filelist_path = download_http(
        filelist_url, output_dir / filelist_name, overwrite=overwrite
    )
    return archive_path, filelist_path


def read_pmcids(path: str | Path) -> list[str]:
    pmcids: list[str] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pmcid = normalize_pmcid(stripped)
            if pmcid and pmcid not in seen:
                seen.add(pmcid)
                pmcids.append(pmcid)
    return pmcids


def _s3_client():
    return boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            retries={"max_attempts": 8, "mode": "adaptive"},
            max_pool_connections=32,
        ),
    )


def _list_versions(client, pmcid: str) -> list[tuple[int, str]]:
    paginator = client.get_paginator("list_objects_v2")
    prefixes: list[tuple[int, str]] = []
    for page in paginator.paginate(
        Bucket=PMC_BUCKET,
        Prefix=f"{pmcid}.",
        Delimiter="/",
    ):
        for entry in page.get("CommonPrefixes", []):
            prefix = entry["Prefix"]
            match = re.fullmatch(rf"{re.escape(pmcid)}\.(\d+)/", prefix)
            if match:
                prefixes.append((int(match.group(1)), prefix))
    return sorted(prefixes)


def _read_json_object(client, key: str) -> dict[str, Any]:
    body = client.get_object(Bucket=PMC_BUCKET, Key=key)["Body"].read()
    return json.loads(body)


def _key_from_s3_url(url: str | None) -> str | None:
    if not url:
        return None
    clean = url.split("?", 1)[0]
    parsed = urlparse(clean)
    if parsed.scheme != "s3" or parsed.netloc != PMC_BUCKET:
        return None
    return parsed.path.lstrip("/")


def _download_one_pmcid(
    client,
    pmcid: str,
    output_dir: Path,
    allowed_licenses,
    overwrite: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {"pmcid": pmcid, "status": "failed"}

    versions = _list_versions(client, pmcid)
    if not versions:
        result["reason"] = "no_article_version_found"
        return result

    version, prefix = versions[-1]
    article_version = prefix.rstrip("/")
    metadata_key = f"metadata/{article_version}.json"

    try:
        metadata = _read_json_object(client, metadata_key)
    except Exception:
        # Fallback to the copy located inside the article-version prefix.
        metadata_key = f"{prefix}{article_version}.json"
        metadata = _read_json_object(client, metadata_key)

    license_code = normalize_license(metadata.get("license_code"))
    result.update(
        {
            "article_version": article_version,
            "version": version,
            "license": license_code,
            "is_retracted": bool(metadata.get("is_retracted")),
            "is_pmc_openaccess": bool(metadata.get("is_pmc_openaccess")),
        }
    )

    if not result["is_pmc_openaccess"]:
        result.update(status="skipped", reason="not_pmc_open_access")
        return result
    if result["is_retracted"]:
        result.update(status="skipped", reason="retracted")
        return result
    if not license_is_allowed(license_code, allowed_licenses):
        result.update(status="skipped", reason="license_not_allowed")
        return result

    xml_key = _key_from_s3_url(metadata.get("xml_url"))
    if xml_key is None:
        xml_key = f"{prefix}{article_version}.xml"

    xml_dir = output_dir / "xml"
    metadata_dir = output_dir / "metadata"
    xml_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    xml_path = xml_dir / f"{article_version}.xml"
    metadata_path = metadata_dir / f"{article_version}.json"

    if overwrite or not xml_path.exists():
        client.download_file(PMC_BUCKET, xml_key, str(xml_path))
    if overwrite or not metadata_path.exists():
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    result.update(
        status="downloaded",
        reason=None,
        xml_path=str(xml_path),
        metadata_path=str(metadata_path),
    )
    return result


def download_pmcids(
    pmcid_file: str | Path,
    output_dir: str | Path,
    *,
    allowed_licenses=DEFAULT_ALLOWED_LICENSES,
    workers: int = 8,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "download_manifest.jsonl"

    pmcids = read_pmcids(pmcid_file)
    client = _s3_client()

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _download_one_pmcid,
                client,
                pmcid,
                output_dir,
                allowed_licenses,
                overwrite,
            ): pmcid
            for pmcid in pmcids
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading PMC XML",
        ):
            pmcid = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "pmcid": pmcid,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )

    results.sort(key=lambda item: item["pmcid"])
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    return manifest_path
