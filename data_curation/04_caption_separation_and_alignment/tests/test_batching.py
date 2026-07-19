from pathlib import Path

from medpmc_caption_separation_and_alignment.backends import InferenceRequest
from medpmc_caption_separation_and_alignment.batching import plan_request_batches


def _request(name: str, images: int) -> InferenceRequest:
    return InferenceRequest(
        request_id=name,
        prompt="prompt",
        image_paths=tuple(Path(f"{name}-{index}.jpg") for index in range(images)),
    )


def test_plan_request_batches_respects_request_and_image_limits():
    requests = [
        _request("a", 3),
        _request("b", 7),
        _request("c", 4),
        _request("d", 4),
    ]
    batches = plan_request_batches(
        requests,
        max_requests=2,
        max_images=8,
        group_by_image_count=True,
    )
    assert all(len(batch) <= 2 for batch in batches)
    assert all(sum(request.image_count for request in batch) <= 8 for batch in batches)
    assert {request.request_id for batch in batches for request in batch} == {"a", "b", "c", "d"}
