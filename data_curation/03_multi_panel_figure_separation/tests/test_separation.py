import os

from medpmc_multi_panel_figure_separation.separation import (
    _resolve_device,
    historical_row_major_order,
    parent_image_id,
    semantic_image_id,
    spatial_order,
    trusted_full_checkpoint_loading,
)


def test_parent_image_id_uses_selected_graphic():
    assert (
        parent_image_id({"pmcid": "PMC1", "selected_image_href": "ABC.f1.jpg"})
        == "PMC1_ABC.f1"
    )


def test_device_mapping():
    assert _resolve_device("cuda") == 0
    assert _resolve_device("cuda:2") == 2
    assert _resolve_device("cpu") == "cpu"


def test_historical_order_reproduces_original_example():
    # Original normalized xywh example converted to xyxy pixels at 1000x1000.
    xywh = [
        (0, 0.257111, 0.502172, 0.514223, 0.331965),
        (0, 0.758414, 0.502013, 0.477379, 0.331892),
        (0, 0.256484, 0.836151, 0.512968, 0.327350),
        (0, 0.258016, 0.165869, 0.516032, 0.331464),
        (0, 0.758930, 0.165536, 0.475212, 0.331072),
        (0, 0.758147, 0.836194, 0.478209, 0.327373),
    ]
    boxes = []
    for detector_index, (_, xc, yc, width, height) in enumerate(xywh):
        boxes.append(
            {
                "detector_index": detector_index,
                "x1": (xc - width / 2) * 1000,
                "y1": (yc - height / 2) * 1000,
                "x2": (xc + width / 2) * 1000,
                "y2": (yc + height / 2) * 1000,
            }
        )
    assert historical_row_major_order(
        boxes, image_width=1000, image_height=1000
    ) == {3: 0, 4: 1, 0: 2, 1: 3, 2: 4, 5: 5}


def test_historical_order_uses_adjacent_y_chaining():
    boxes = [
        {"detector_index": 0, "x1": 200, "y1": 0, "x2": 300, "y2": 100},
        {"detector_index": 1, "x1": 100, "y1": 40, "x2": 200, "y2": 140},
        {"detector_index": 2, "x1": 0, "y1": 80, "x2": 100, "y2": 180},
    ]
    # Adjacent normalized y differences are 0.04, so all boxes remain one row
    # even though the first-to-last difference is 0.08.
    assert historical_row_major_order(
        boxes, image_width=1000, image_height=1000, y_threshold=0.05
    ) == {2: 0, 1: 1, 0: 2}


def test_historical_order_starts_new_row_above_threshold():
    boxes = [
        {"detector_index": 0, "x1": 100, "y1": 0, "x2": 190, "y2": 90},
        {"detector_index": 1, "x1": 0, "y1": 51, "x2": 90, "y2": 141},
    ]
    assert historical_row_major_order(
        boxes, image_width=1000, image_height=1000, y_threshold=0.05
    ) == {0: 0, 1: 1}


def test_spatial_order_alias_uses_historical_rule():
    boxes = [
        {"detector_index": 0, "x1": 100, "y1": 100, "x2": 190, "y2": 190},
        {"detector_index": 1, "x1": 0, "y1": 0, "x2": 90, "y2": 90},
        {"detector_index": 2, "x1": 100, "y1": 0, "x2": 190, "y2": 90},
        {"detector_index": 3, "x1": 0, "y1": 100, "x2": 90, "y2": 190},
    ]
    assert spatial_order(boxes, image_width=200, image_height=200) == {
        1: 0,
        2: 1,
        3: 2,
        0: 3,
    }


def test_trusted_checkpoint_loading_sets_and_restores_environment(monkeypatch):
    name = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    monkeypatch.delenv(name, raising=False)
    with trusted_full_checkpoint_loading():
        assert os.environ[name] == "1"
    assert name not in os.environ


def test_trusted_checkpoint_loading_preserves_existing_value(monkeypatch):
    name = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
    monkeypatch.setenv(name, "yes")
    with trusted_full_checkpoint_loading():
        assert os.environ[name] == "1"
    assert os.environ[name] == "yes"


def test_semantic_image_id_uses_reordered_subfigure_index():
    assert semantic_image_id("PMC1_ABC.f1", 7) == "PMC1_ABC.f1_7"
