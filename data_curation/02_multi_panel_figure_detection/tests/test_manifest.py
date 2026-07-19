from medpmc_multi_panel_figure_detection.manifest import _select_primary_image_match


def test_primary_image_selection_uses_first_graphic_document_order():
    hrefs = ["JEM981243.f1a", "JEM981243.f1b"]
    media_urls = [
        "s3://bucket/PMC1887701.1/jem981243.f1a.jpg?md5=a",
        "s3://bucket/PMC1887701.1/jem981243.f1b.jpg?md5=b",
    ]

    selected_href, match, candidate_count = _select_primary_image_match(
        hrefs, media_urls
    )

    assert selected_href == "JEM981243.f1a"
    assert match is not None
    assert "jem981243.f1a.jpg" in match.media_url
    assert candidate_count == 2


def test_primary_image_selection_does_not_substitute_later_graphic():
    hrefs = ["missing.f1a", "present.f1b"]
    media_urls = ["s3://bucket/PMC1.1/present.f1b.jpg?md5=b"]

    selected_href, match, candidate_count = _select_primary_image_match(
        hrefs, media_urls
    )

    assert selected_href == "missing.f1a"
    assert match is None
    assert candidate_count == 1
