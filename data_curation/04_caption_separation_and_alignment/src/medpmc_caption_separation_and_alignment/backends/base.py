"""Backend-neutral batch inference interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class InferenceRequest:
    """One caption-separation request."""

    request_id: str
    prompt: str
    image_paths: tuple[Path, ...]

    @property
    def image_count(self) -> int:
        return len(self.image_paths)


@dataclass
class InferenceResult:
    """Backend response for one request."""

    request_id: str
    text: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchInferenceBackend(Protocol):
    """Minimal interface implemented by inference backends."""

    name: str
    engine: str

    def generate_batch(
        self,
        requests: Sequence[InferenceRequest],
        *,
        max_new_tokens: int | None = None,
    ) -> list[InferenceResult]: ...

    def close(self) -> None: ...
