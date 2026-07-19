"""License normalization and filtering."""

from __future__ import annotations

import re
from collections.abc import Iterable

DEFAULT_ALLOWED_LICENSES = frozenset(
    {"CC BY", "CC BY-NC", "CC BY-NC-SA", "CC0", "CC BY-SA"}
)

_LICENSE_PATTERNS = (
    ("CC BY-NC-ND", ("BY-NC-ND", "BY NC ND", "/BY-NC-ND/")),
    ("CC BY-NC-SA", ("BY-NC-SA", "BY NC SA", "/BY-NC-SA/")),
    ("CC BY-ND", ("BY-ND", "BY ND", "/BY-ND/")),
    ("CC BY-SA", ("BY-SA", "BY SA", "/BY-SA/")),
    ("CC BY-NC", ("BY-NC", "BY NC", "/BY-NC/")),
    ("CC0", ("CC0", "PUBLIC DOMAIN DEDICATION")),
    ("CC BY", ("CC BY", "/BY/")),
)


def normalize_license(value: object) -> str | None:
    """Return a canonical PMC/Creative Commons license code."""
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    upper = re.sub(r"\s+", " ", text.upper())
    upper = upper.replace("_", "-")

    if "NO-CC CODE" in upper or upper in {"NO CC CODE", "NONE", "NULL", "N/A"}:
        return "NO-CC CODE"
    if upper == "TDM":
        return "TDM"

    for canonical, patterns in _LICENSE_PATTERNS:
        if any(pattern in upper for pattern in patterns):
            return canonical

    return upper


def parse_allowed_licenses(value: str | Iterable[str] | None) -> frozenset[str]:
    """Parse a comma-separated or iterable allowlist."""
    if value is None:
        return DEFAULT_ALLOWED_LICENSES

    if isinstance(value, str):
        items = value.split(",")
    else:
        items = list(value)

    normalized = {normalize_license(item) for item in items}
    return frozenset(item for item in normalized if item)


def license_is_allowed(
    license_value: object,
    allowed_licenses: Iterable[str] = DEFAULT_ALLOWED_LICENSES,
) -> bool:
    normalized = normalize_license(license_value)
    allowed = parse_allowed_licenses(allowed_licenses)
    return normalized in allowed
