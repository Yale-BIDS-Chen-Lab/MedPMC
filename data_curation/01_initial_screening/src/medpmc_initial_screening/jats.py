"""Figure-level extraction from PMC JATS XML."""

from __future__ import annotations

import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from lxml import etree

from .filelist import normalize_pmcid
from .licenses import normalize_license

XLINK_HREF = "{http://www.w3.org/1999/xlink}href"

# JATS permits floating objects to appear inside paragraphs. The cleaned
# downstream representation excludes them, while the screening-compatible
# representation preserves the full recursive paragraph text expected by the
# released classifier.
_REFERENCE_EXCLUDED_ELEMENTS = frozenset(
    {
        "fig",
        "fig-group",
        "table-wrap",
        "table-wrap-group",
        "supplementary-material",
        "media",
        "boxed-text",
        "disp-formula",
        "disp-formula-group",
        "preformat",
        "code",
    }
)


def _local_name(node: etree._Element) -> str:
    return etree.QName(node).localname


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _node_text(node: etree._Element) -> str:
    """Return mixed-content text without inserting spaces around inline tags."""
    return _clean_text("".join(node.itertext()))


def _first_text(root: etree._Element, xpath: str) -> str:
    values = root.xpath(xpath)
    if not values:
        return ""
    value = values[0]
    if isinstance(value, etree._Element):
        return _node_text(value)
    return _clean_text(str(value))


def _first_article_id(root: etree._Element, id_types: tuple[str, ...]) -> str:
    accepted = {value.lower() for value in id_types}
    for node in root.xpath("//*[local-name()='article-id']"):
        id_type = (node.get("pub-id-type") or "").strip().lower()
        if id_type in accepted:
            value = _node_text(node)
            if value:
                return value
    return ""


def _top_level_text_blocks(
    container: etree._Element,
    names: frozenset[str] = frozenset({"title", "p"}),
) -> list[str]:
    """Return top-level caption text blocks without double-counting nested text."""
    parts: list[str] = []

    for node in container.iter():
        if node is container or _local_name(node) not in names:
            continue

        parent = node.getparent()
        is_nested_block = False
        while parent is not None and parent is not container:
            if _local_name(parent) in names:
                is_nested_block = True
                break
            parent = parent.getparent()

        if is_nested_block:
            continue

        text = _node_text(node)
        if text:
            parts.append(text)

    return parts


def _caption_text(caption: etree._Element) -> str:
    parts = _top_level_text_blocks(caption)
    return _clean_text(" ".join(parts)) if parts else _node_text(caption)


def _remove_element_preserving_tail(node: etree._Element) -> None:
    parent = node.getparent()
    if parent is None:
        return

    tail = node.tail or ""
    previous = node.getprevious()
    if previous is not None:
        previous.tail = (previous.tail or "") + tail
    else:
        parent.text = (parent.text or "") + tail

    parent.remove(node)


def _reference_container_text(container: etree._Element) -> str:
    """Extract paragraph text while excluding nested floating objects."""
    clone = deepcopy(container)
    for node in list(clone.iter()):
        if node is clone:
            continue
        if _local_name(node) in _REFERENCE_EXCLUDED_ELEMENTS:
            _remove_element_preserving_tail(node)
    return _node_text(clone)


def _xml_license(root: etree._Element) -> str | None:
    candidates: list[str] = []

    # PMC supplies a canonical license value in many current XML files.
    custom_meta_nodes = root.xpath(
        "//*[local-name()='custom-meta']["
        "*[local-name()='meta-name' and "
        "normalize-space(text())='pmc-license-ref']]"
    )
    for custom_meta in custom_meta_nodes:
        values = custom_meta.xpath("./*[local-name()='meta-value'][1]")
        if values:
            candidates.append(_node_text(values[0]))

    for node in root.xpath("//*[local-name()='license']"):
        license_type = node.get("license-type")
        if license_type:
            candidates.append(license_type)

        for child in node.iter():
            local = _local_name(child)
            if local in {"license_ref", "ext-link"}:
                href = child.get(XLINK_HREF) or child.get("href")
                if href:
                    candidates.append(href)

        candidates.append(_node_text(node))

    for candidate in candidates:
        normalized = normalize_license(candidate)
        if normalized:
            return normalized
    return None


def _article_metadata(root: etree._Element) -> dict[str, str]:
    pmcid = _first_article_id(root, ("pmcid", "pmc"))
    article_version = _first_article_id(
        root,
        ("pmcid-ver", "pmc-version", "pmcid-version"),
    )
    pmid = _first_article_id(root, ("pmid",))

    return {
        "pmcid": normalize_pmcid(pmcid) or "",
        "article_version": article_version,
        "pmid": pmid,
        "article_title": _first_text(
            root,
            "//*[local-name()='article-title'][1]",
        ),
        "journal_title": _first_text(
            root,
            "//*[local-name()='journal-title'][1] | "
            "//*[local-name()='journal-id'][1]",
        ),
        "xml_license": _xml_license(root) or "",
    }


def _inline_references(
    root: etree._Element,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return cleaned and screening-compatible figure-reference paragraphs.

    The cleaned representation removes nested floating objects and is the
    canonical value stored for downstream use. The screening-compatible
    representation recursively preserves all paragraph text, matching the
    preprocessing used to train the released Initial Screening model.
    """
    clean_references: dict[str, list[str]] = defaultdict(list)
    screening_references: dict[str, list[str]] = defaultdict(list)
    clean_seen: dict[str, set[str]] = defaultdict(set)
    screening_seen: dict[str, set[str]] = defaultdict(set)

    xrefs = root.xpath(
        "//*[local-name()='xref' and "
        "translate(@ref-type, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
        "'abcdefghijklmnopqrstuvwxyz')='fig']"
    )

    for xref in xrefs:
        rids = [
            rid
            for rid in re.split(r"\s+", (xref.get("rid") or "").strip())
            if rid
        ]
        if not rids:
            continue

        containers = xref.xpath("ancestor::*[local-name()='p'][1]")
        if not containers:
            continue

        container = containers[0]
        clean_text = _reference_container_text(container)
        screening_text = _node_text(container)

        for rid in rids:
            if clean_text and clean_text not in clean_seen[rid]:
                clean_seen[rid].add(clean_text)
                clean_references[rid].append(clean_text)
            if screening_text and screening_text not in screening_seen[rid]:
                screening_seen[rid].add(screening_text)
                screening_references[rid].append(screening_text)

    return clean_references, screening_references


def parse_article_xml(
    xml_content: bytes | str | Path,
    *,
    source_xml: str = "",
    metadata: dict[str, Any] | None = None,
    require_caption: bool = True,
) -> list[dict[str, Any]]:
    """Parse a PMC article and return one record per captioned figure."""
    parser = etree.XMLParser(
        recover=True,
        resolve_entities=False,
        no_network=True,
        huge_tree=True,
        remove_comments=True,
    )

    if isinstance(xml_content, Path):
        root = etree.parse(str(xml_content), parser).getroot()
    elif isinstance(xml_content, str) and Path(xml_content).exists():
        root = etree.parse(xml_content, parser).getroot()
    elif isinstance(xml_content, str):
        root = etree.fromstring(xml_content.encode("utf-8"), parser=parser)
    else:
        root = etree.fromstring(xml_content, parser=parser)

    article = _article_metadata(root)
    metadata = metadata or {}

    pmcid = (
        normalize_pmcid(metadata.get("pmcid"))
        or normalize_pmcid(metadata.get("accession_id"))
        or article["pmcid"]
    )
    article_version = str(
        metadata.get("article_version")
        or metadata.get("versioned_pmcid")
        or metadata.get("pmcid_ver")
        or article["article_version"]
        or ""
    )
    pmid = str(metadata.get("pmid") or article["pmid"] or "")

    license_code = (
        normalize_license(metadata.get("license_code"))
        or normalize_license(metadata.get("license"))
        or normalize_license(article["xml_license"])
    )

    clean_references, screening_references = _inline_references(root)
    figures = root.xpath("//*[local-name()='fig']")
    records: list[dict[str, Any]] = []

    for index, figure in enumerate(figures, start=1):
        figure_id = figure.get("id") or f"fig-{index:04d}"

        caption_nodes = figure.xpath("./*[local-name()='caption'][1]")
        caption = _caption_text(caption_nodes[0]) if caption_nodes else ""
        if require_caption and not caption:
            continue

        label_nodes = figure.xpath("./*[local-name()='label'][1]")
        figure_label = _node_text(label_nodes[0]) if label_nodes else ""

        image_hrefs: list[str] = []
        for node in figure.xpath(
            ".//*[local-name()='graphic' or local-name()='media' or "
            "local-name()='inline-graphic']"
        ):
            href = node.get(XLINK_HREF) or node.get("href")
            if href and href not in image_hrefs:
                image_hrefs.append(href)

        reference_texts = clean_references.get(figure_id, [])
        screening_reference_texts = screening_references.get(figure_id, [])

        # Avoid duplicating large text columns when the two representations are
        # identical. A null override means screening should use reference_texts.
        screening_override = (
            screening_reference_texts
            if screening_reference_texts != reference_texts
            else None
        )

        records.append(
            {
                "pmcid": pmcid or "",
                "article_version": article_version,
                "pmid": pmid,
                "article_title": metadata.get("title") or article["article_title"],
                "journal_title": article["journal_title"],
                "license": license_code or "",
                "figure_id": figure_id,
                "figure_label": figure_label,
                "caption": caption,
                "reference_texts": reference_texts,
                "reference_texts_screening": screening_override,
                "image_hrefs": image_hrefs,
                "source_xml": source_xml,
            }
        )

    return records


def format_model_input(caption: str, reference_texts: list[str] | None) -> str:
    references = "\n".join(reference_texts or [])
    return f'"Caption": {caption}\n"Reference Text": {references}'
