"""Deterministic batching utilities for variable-image VLM requests."""

from __future__ import annotations

from collections.abc import Sequence

from .backends.base import InferenceRequest


def plan_request_batches(
    requests: Sequence[InferenceRequest],
    *,
    max_requests: int,
    max_images: int | None,
    group_by_image_count: bool = True,
) -> list[list[InferenceRequest]]:
    """Pack requests under both request-count and image-count limits.

    Grouping requests with similar image counts reduces padding and vision
    encoder imbalance while request IDs preserve deterministic output order.
    """
    if max_requests < 1:
        raise ValueError("max_requests must be positive")
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive")

    indexed = list(enumerate(requests))
    if group_by_image_count:
        indexed.sort(key=lambda item: (item[1].image_count, item[0]))

    batches: list[list[InferenceRequest]] = []
    current: list[InferenceRequest] = []
    current_images = 0

    for _, request in indexed:
        exceeds_requests = len(current) >= max_requests
        exceeds_images = (
            max_images is not None
            and current
            and current_images + request.image_count > max_images
        )
        if exceeds_requests or exceeds_images:
            batches.append(current)
            current = []
            current_images = 0

        current.append(request)
        current_images += request.image_count

    if current:
        batches.append(current)
    return batches
