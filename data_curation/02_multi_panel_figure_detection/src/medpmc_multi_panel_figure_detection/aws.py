"""Anonymous PMC Open Data S3 access and media resolution."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, unquote, urlparse

import boto3
from botocore import UNSIGNED
from botocore.config import Config

DEFAULT_BUCKET = "pmc-oa-opendata"
DEFAULT_REGION = "us-east-1"
_VERSIONED_PMCID_RE = re.compile(r"^(PMC\d+)\.(\d+)$", re.IGNORECASE)
_RASTER_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".gif",
    ".webp",
    ".bmp",
)
_EXTENSION_RANK = {ext: rank for rank, ext in enumerate(_RASTER_EXTENSIONS)}


@dataclass(frozen=True)
class S3ObjectRef:
    bucket: str
    key: str
    expected_md5: str
    original_url: str


@dataclass(frozen=True)
class MediaMatch:
    requested_href: str
    media_url: str
    match_type: str


@dataclass(frozen=True)
class ResolvedArticle:
    metadata: dict[str, Any] | None
    metadata_key: str
    versioned_pmcid: str
    resolution_method: str
    candidate_versions: int
    matched_figures: int


def create_unsigned_s3_client(
    *,
    region_name: str = DEFAULT_REGION,
    max_pool_connections: int = 64,
):
    return boto3.client(
        "s3",
        region_name=region_name,
        config=Config(
            signature_version=UNSIGNED,
            retries={"max_attempts": 8, "mode": "standard"},
            max_pool_connections=max_pool_connections,
        ),
    )


def normalize_pmcid(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    match = re.search(r"PMC\d+", text)
    return match.group(0) if match else ""


def normalize_versioned_pmcid(pmcid: object, article_version: object) -> str:
    normalized_pmcid = normalize_pmcid(pmcid)
    raw = str(article_version or "").strip().upper()
    if not normalized_pmcid or not raw:
        return ""

    match = _VERSIONED_PMCID_RE.match(raw)
    if match and match.group(1).upper() == normalized_pmcid:
        return f"{normalized_pmcid}.{int(match.group(2))}"

    if raw.isdigit():
        return f"{normalized_pmcid}.{int(raw)}"

    suffix_match = re.search(r"\.(\d+)$", raw)
    if suffix_match:
        return f"{normalized_pmcid}.{int(suffix_match.group(1))}"
    return ""


def version_number(versioned_pmcid: str) -> int:
    match = _VERSIONED_PMCID_RE.match(versioned_pmcid.strip())
    return int(match.group(2)) if match else -1


def parse_s3_url(url: str) -> S3ObjectRef:
    parsed = urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"Not a valid S3 URL: {url}")
    query = parse_qs(parsed.query)
    expected_md5 = (query.get("md5") or [""])[0].lower()
    return S3ObjectRef(
        bucket=parsed.netloc,
        key=unquote(parsed.path.lstrip("/")),
        expected_md5=expected_md5,
        original_url=url,
    )


def media_basename(value: str) -> str:
    parsed = urlparse(str(value or ""))
    path = parsed.path if parsed.scheme else str(value or "")
    return Path(unquote(path)).name.lower()


def media_stem(value: str) -> str:
    """Normalize a media name by removing only known raster extensions.

    JATS hrefs often end in figure identifiers such as ``.f1`` rather than a
    file extension (for example, ``JCB9806103.f1``). ``Path.stem`` would
    incorrectly remove that identifier. PMC media files may add a real raster
    extension, such as ``JCB9806103.f1.gif``.
    """
    name = media_basename(value)
    for extension in sorted(_RASTER_EXTENSIONS, key=len, reverse=True):
        if name.endswith(extension):
            return name[: -len(extension)]
    return name


def _media_sort_key(url: str) -> tuple[int, str]:
    suffix = Path(media_basename(url)).suffix.lower()
    return (_EXTENSION_RANK.get(suffix, len(_EXTENSION_RANK) + 1), url)


def match_media_url(
    requested_hrefs: Iterable[str],
    media_urls: Iterable[str],
) -> MediaMatch | None:
    """Match a JATS href to PMC media, preferring exact basename then stem."""
    requested = [href for href in requested_hrefs if str(href or "").strip()]
    available = sorted(
        [url for url in media_urls if str(url or "").strip()],
        key=_media_sort_key,
    )
    if not requested or not available:
        return None

    by_basename: dict[str, list[str]] = {}
    by_stem: dict[str, list[str]] = {}
    for url in available:
        by_basename.setdefault(media_basename(url), []).append(url)
        by_stem.setdefault(media_stem(url), []).append(url)

    for href in requested:
        basename = media_basename(href)
        exact = by_basename.get(basename, [])
        if exact:
            return MediaMatch(href, exact[0], "basename")

    for href in requested:
        stem = media_stem(href)
        stem_matches = by_stem.get(stem, [])
        if stem_matches:
            return MediaMatch(href, stem_matches[0], "stem")
    return None


def match_all_media_urls(
    requested_hrefs: Iterable[str],
    media_urls: Iterable[str],
) -> list[MediaMatch]:
    """Return one best media match per requested href, without duplicates."""
    available = list(media_urls)
    matches: list[MediaMatch] = []
    seen_urls: set[str] = set()
    for href in requested_hrefs:
        match = match_media_url([href], available)
        if match is not None and match.media_url not in seen_urls:
            matches.append(match)
            seen_urls.add(match.media_url)
    return matches


def metadata_key(versioned_pmcid: str) -> str:
    return f"metadata/{versioned_pmcid}.json"


def read_json_object(client, *, bucket: str, key: str) -> dict[str, Any]:
    response = client.get_object(Bucket=bucket, Key=key)
    stream = response["Body"]
    try:
        body = stream.read()
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    return json.loads(body.decode("utf-8"))


def list_metadata_keys(client, *, bucket: str, pmcid: str) -> list[str]:
    prefix = f"metadata/{normalize_pmcid(pmcid)}."
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            name = Path(key).name
            if re.match(rf"^{re.escape(normalize_pmcid(pmcid))}\.\d+\.json$", name):
                keys.append(key)
    return sorted(
        set(keys),
        key=lambda key: version_number(Path(key).stem),
        reverse=True,
    )


def _requested_figure_hrefs(records: Iterable[dict[str, Any]]) -> list[list[str]]:
    """Return the primary JATS graphic href for each figure.

    The original MedPMC pipeline used ``fig.find(".//graphic")`` and therefore
    selected the first graphic in XML document order. Article-version
    resolution must use the same primary href rather than allowing a later
    alternative graphic to determine the version.
    """
    requested: list[list[str]] = []
    for record in records:
        hrefs = [
            str(value)
            for value in (record.get("image_hrefs") or [])
            if str(value or "").strip()
        ]
        requested.append(hrefs[:1])
    return requested


def _coverage(metadata: dict[str, Any], requested: list[list[str]]) -> int:
    media_urls = metadata.get("media_urls") or []
    return sum(match_media_url(hrefs, media_urls) is not None for hrefs in requested)


def resolve_article_metadata(
    client,
    *,
    bucket: str,
    pmcid: str,
    article_version: str = "",
    records: Iterable[dict[str, Any]] = (),
) -> ResolvedArticle:
    """Resolve the article version whose media best match the extracted hrefs."""
    normalized_pmcid = normalize_pmcid(pmcid)
    if not normalized_pmcid:
        return ResolvedArticle(None, "", "", "invalid_pmcid", 0, 0)

    requested = _requested_figure_hrefs(records)
    explicit = normalize_versioned_pmcid(normalized_pmcid, article_version)

    # Avoid an S3 LIST request when the XML supplies an exact article version
    # and all requested figures are present in that version's metadata.
    if explicit:
        explicit_key = metadata_key(explicit)
        try:
            explicit_metadata = read_json_object(client, bucket=bucket, key=explicit_key)
        except Exception as exc:
            response = getattr(exc, "response", {})
            code = str(response.get("Error", {}).get("Code", ""))
            if code not in {"NoSuchKey", "404", "NotFound"}:
                raise
        else:
            explicit_coverage = _coverage(explicit_metadata, requested)
            if explicit_coverage == len(requested):
                return ResolvedArticle(
                    explicit_metadata,
                    explicit_key,
                    explicit,
                    "explicit_version",
                    1,
                    explicit_coverage,
                )

    candidate_keys = list_metadata_keys(client, bucket=bucket, pmcid=normalized_pmcid)
    if explicit:
        explicit_key = metadata_key(explicit)
        if explicit_key not in candidate_keys:
            candidate_keys.insert(0, explicit_key)
        else:
            candidate_keys.remove(explicit_key)
            candidate_keys.insert(0, explicit_key)

    loaded: list[tuple[str, dict[str, Any], int]] = []
    for key in candidate_keys:
        try:
            metadata = read_json_object(client, bucket=bucket, key=key)
        except client.exceptions.NoSuchKey:
            continue
        except Exception as exc:  # botocore maps some 404s to ClientError
            response = getattr(exc, "response", {})
            code = str(response.get("Error", {}).get("Code", ""))
            if code in {"NoSuchKey", "404", "NotFound"}:
                continue
            raise
        loaded.append((key, metadata, _coverage(metadata, requested)))

    if not loaded:
        return ResolvedArticle(None, "", "", "metadata_not_found", 0, 0)

    explicit_loaded = next(
        (
            item
            for item in loaded
            if explicit and Path(item[0]).stem.upper() == explicit.upper()
        ),
        None,
    )
    if explicit_loaded and explicit_loaded[2] == len(requested):
        key, metadata, coverage = explicit_loaded
        return ResolvedArticle(
            metadata,
            key,
            Path(key).stem,
            "explicit_version",
            len(loaded),
            coverage,
        )

    key, metadata, coverage = max(
        loaded,
        key=lambda item: (
            item[2],
            version_number(Path(item[0]).stem),
        ),
    )
    method = "media_match" if coverage > 0 else "latest_version_no_media_match"
    return ResolvedArticle(
        metadata,
        key,
        Path(key).stem,
        method,
        len(loaded),
        coverage,
    )


def download_s3_object(
    client,
    ref: S3ObjectRef,
    destination: str | Path,
    *,
    force: bool = False,
) -> tuple[str, str]:
    """Download an S3 object atomically and return (action, md5)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def file_md5(path: Path) -> str:
        digest = hashlib.md5()  # nosec B324 - checksum verification only
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    if destination.exists() and not force:
        digest = file_md5(destination)
        if not ref.expected_md5 or digest == ref.expected_md5:
            return "reused", digest

    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    digest = hashlib.md5()  # nosec B324 - checksum verification only
    try:
        response = client.get_object(Bucket=ref.bucket, Key=ref.key)
        stream = response["Body"]
        try:
            with partial.open("wb") as handle:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    handle.write(chunk)
                    digest.update(chunk)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        actual_md5 = digest.hexdigest()
        if ref.expected_md5 and actual_md5 != ref.expected_md5:
            raise ValueError(
                f"MD5 mismatch for {ref.key}: expected {ref.expected_md5}, "
                f"got {actual_md5}"
            )
        partial.replace(destination)
        return "downloaded", actual_md5
    except Exception:
        partial.unlink(missing_ok=True)
        raise
