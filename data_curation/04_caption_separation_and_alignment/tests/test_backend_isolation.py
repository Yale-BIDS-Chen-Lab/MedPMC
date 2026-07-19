from pathlib import Path

from medpmc_caption_separation_and_alignment.alignment import _run_batch_with_isolation
from medpmc_caption_separation_and_alignment.backends import InferenceRequest, InferenceResult


class SplittingBackend:
    name = "fake"
    engine = "batch"

    def generate_batch(self, requests):
        if any(request.request_id == "bad" for request in requests):
            if len(requests) > 1:
                raise RuntimeError("batch failed")
            raise RuntimeError("bad sample")
        return [InferenceResult(request_id=request.request_id, text="ok") for request in requests]

    def close(self):
        pass


def test_batch_failure_isolates_single_bad_request():
    requests = [
        InferenceRequest("good1", "p", (Path("a"),)),
        InferenceRequest("bad", "p", (Path("b"),)),
        InferenceRequest("good2", "p", (Path("c"),)),
    ]
    results = _run_batch_with_isolation(SplittingBackend(), requests)
    by_id = {result.request_id: result for result in results}
    assert by_id["good1"].text == "ok"
    assert by_id["good2"].text == "ok"
    assert "bad sample" in by_id["bad"].error


class BudgetAwareBackend:
    name = "fake"
    engine = "batch"

    def __init__(self):
        self.budgets = []

    def generate_batch(self, requests, *, max_new_tokens=None):
        self.budgets.append(max_new_tokens)
        return [InferenceResult(request_id=request.request_id, text="ok") for request in requests]

    def close(self):
        pass


def test_batch_override_is_forwarded_when_supported():
    backend = BudgetAwareBackend()
    request = InferenceRequest("one", "p", (Path("a"),))
    result = _run_batch_with_isolation(backend, [request], max_new_tokens=1024)
    assert result[0].text == "ok"
    assert backend.budgets == [1024]
