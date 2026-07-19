from medpmc_initial_screening.screen import _reference_texts_for_screening


def test_model_compatible_prefers_screening_override():
    record = {
        "reference_texts": ["clean"],
        "reference_texts_screening": ["recursive"],
    }
    assert _reference_texts_for_screening(record, "model-compatible") == [
        "recursive"
    ]
    assert _reference_texts_for_screening(record, "clean") == ["clean"]


def test_model_compatible_falls_back_when_override_is_null():
    record = {
        "reference_texts": ["same"],
        "reference_texts_screening": None,
    }
    assert _reference_texts_for_screening(record, "model-compatible") == ["same"]
