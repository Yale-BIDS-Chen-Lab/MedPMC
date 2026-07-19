from medpmc_multi_panel_figure_detection.aws import (
    match_media_url,
    normalize_versioned_pmcid,
    parse_s3_url,
)


def test_normalize_versioned_pmcid():
    assert normalize_versioned_pmcid("PMC123", "1") == "PMC123.1"
    assert normalize_versioned_pmcid("PMC123", "PMC123.2") == "PMC123.2"
    assert normalize_versioned_pmcid("PMC123", "") == ""


def test_parse_s3_url_with_md5():
    ref = parse_s3_url("s3://pmc-oa-opendata/PMC123.1/gr1.jpg?md5=abcdef")
    assert ref.bucket == "pmc-oa-opendata"
    assert ref.key == "PMC123.1/gr1.jpg"
    assert ref.expected_md5 == "abcdef"


def test_media_match_exact_then_stem():
    urls = [
        "s3://bucket/PMC1.1/gr1.png?md5=x",
        "s3://bucket/PMC1.1/gr1.jpg?md5=y",
    ]
    exact = match_media_url(["gr1.png"], urls)
    assert exact is not None
    assert exact.match_type == "basename"
    assert "gr1.png" in exact.media_url

    stem = match_media_url(["gr1"], urls)
    assert stem is not None
    assert stem.match_type == "stem"
    assert "gr1.jpg" in stem.media_url


def test_media_match_preserves_dotted_figure_identifier():
    urls = [
        "s3://bucket/PMC1424221.2/jcb9806103.f1.jpg?md5=x",
        "s3://bucket/PMC1424222.2/jcb.29208f3.jpg?md5=y",
    ]

    first = match_media_url(["JCB9806103.f1"], urls)
    assert first is not None
    assert first.match_type == "stem"
    assert "jcb9806103.f1.jpg" in first.media_url

    second = match_media_url(["JCB.29208f3"], urls)
    assert second is not None
    assert second.match_type == "stem"
    assert "jcb.29208f3.jpg" in second.media_url
