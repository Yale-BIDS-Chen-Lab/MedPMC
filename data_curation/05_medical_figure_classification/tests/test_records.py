from medpmc_medical_figure_classification.records import (
    candidate_from_singlepanel,
    candidate_from_subfigure,
    predicted_label_from_score,
    singlepanel_image_id,
    singlepanel_is_eligible,
    subfigure_is_eligible,
)


def test_singlepanel_candidate_and_id():
    row = {
        "pmcid": "PMC1",
        "figure_id": "fig1",
        "selected_image_href": "abc.jpg",
        "local_image_path": "images/PMC1/abc.jpg",
        "caption": "caption",
        "image_status": "ready",
        "detection_status": "classified",
        "is_multipanel": False,
    }
    assert singlepanel_is_eligible(row)
    assert singlepanel_image_id(row) == "PMC1_abc"
    candidate = candidate_from_singlepanel(row)
    assert candidate["source_type"] == "single_panel"
    assert candidate["caption"] == "caption"


def test_subfigure_candidate():
    row = {
        "pmcid": "PMC1",
        "parent_image_id": "PMC1_abc",
        "image_id": "PMC1_abc_0",
        "subfigure_index": 0,
        "local_panel_path": "panels/PMC1_abc/panel-0000.jpg",
        "subcaption": "panel caption",
        "main_caption": "main",
        "crop_status": "ready",
        "caption_alignment_status": "aligned",
    }
    assert subfigure_is_eligible(row)
    candidate = candidate_from_subfigure(row)
    assert candidate["source_type"] == "subfigure"
    assert candidate["caption"] == "panel caption"
    assert candidate["subfigure_index"] == 0


def test_strict_binary_threshold_matches_top1_behavior():
    assert predicted_label_from_score(0.5001) == 1
    assert predicted_label_from_score(0.5) == 0
    assert predicted_label_from_score(0.1) == 0
