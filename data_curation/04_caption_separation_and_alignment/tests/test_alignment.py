from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from medpmc_caption_separation_and_alignment.alignment import (
    PROMPT_PREFIX,
    align_directory,
    build_prompt,
    parse_subcaptions,
)


def test_build_prompt_preserves_image_order_and_count():
    prompt = build_prompt("Main caption", 3, image_token="<image>")
    assert prompt.startswith(PROMPT_PREFIX.format(num_subfigures=3))
    assert prompt.count("<image>") == 4
    assert prompt.index("# Compound Figure") < prompt.index("# Subfigure")
    assert prompt.endswith("# Main Caption\nMain caption")


def test_parse_subcaptions():
    assert parse_subcaptions("(a) first || (b) second") == ["(a) first", "(b) second"]
    assert parse_subcaptions("```text\n(a) first||(b) second\n```") == [
        "(a) first",
        "(b) second",
    ]
    assert parse_subcaptions("   ") == []


def _write(path: Path, records: list[dict], schema: pa.Schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path)


def test_align_directory_with_fake_runner(tmp_path: Path):
    parent_root = tmp_path / "stage2"
    panel_root = tmp_path / "stage3"
    (parent_root / "images").mkdir(parents=True)
    (panel_root / "panels" / "PARENT").mkdir(parents=True)
    for path in [
        parent_root / "images" / "parent.jpg",
        panel_root / "panels" / "PARENT" / "panel-0000.jpg",
        panel_root / "panels" / "PARENT" / "panel-0001.jpg",
    ]:
        path.write_bytes(b"fake")

    parent_schema = pa.schema(
        [
            ("pmcid", pa.string()),
            ("article_version", pa.string()),
            ("resolved_article_version", pa.string()),
            ("pmid", pa.string()),
            ("figure_id", pa.string()),
            ("figure_label", pa.string()),
            ("parent_image_id", pa.string()),
            ("parent_local_image_path", pa.string()),
            ("separation_status", pa.string()),
        ]
    )
    panel_schema = pa.schema(
        [
            ("pmcid", pa.string()),
            ("parent_image_id", pa.string()),
            ("image_id", pa.string()),
            ("subfigure_index", pa.int64()),
            ("historical_order_index", pa.int64()),
            ("detector_index", pa.int64()),
            ("local_panel_path", pa.string()),
            ("crop_status", pa.string()),
            ("caption", pa.string()),
        ]
    )
    _write(
        tmp_path / "parents" / "part-000000.parquet",
        [
            {
                "pmcid": "PMC1",
                "article_version": "PMC1.1",
                "resolved_article_version": "PMC1.1",
                "pmid": "1",
                "figure_id": "F1",
                "figure_label": "Figure 1",
                "parent_image_id": "PARENT",
                "parent_local_image_path": "images/parent.jpg",
                "separation_status": "separated",
            }
        ],
        parent_schema,
    )
    _write(
        tmp_path / "panels_in" / "part-000000.parquet",
        [
            {
                "pmcid": "PMC1",
                "parent_image_id": "PARENT",
                "image_id": "PARENT_1",
                "subfigure_index": 1,
                "historical_order_index": 1,
                "detector_index": 0,
                "local_panel_path": "panels/PARENT/panel-0001.jpg",
                "crop_status": "ready",
                "caption": "Main caption",
            },
            {
                "pmcid": "PMC1",
                "parent_image_id": "PARENT",
                "image_id": "PARENT_0",
                "subfigure_index": 0,
                "historical_order_index": 0,
                "detector_index": 1,
                "local_panel_path": "panels/PARENT/panel-0000.jpg",
                "crop_status": "ready",
                "caption": "Main caption",
            },
        ],
        panel_schema,
    )

    observed = {}

    def runner(caption: str, image_paths: list[Path]) -> str:
        observed["caption"] = caption
        observed["names"] = [path.name for path in image_paths]
        return "first||second"

    summary_path = align_directory(
        tmp_path / "parents",
        tmp_path / "panels_in",
        tmp_path / "out",
        parent_image_root=parent_root,
        panel_image_root=panel_root,
        parents_per_shard=10,
        runner=runner,
    )
    assert summary_path.exists()
    assert observed["names"] == ["parent.jpg", "panel-0000.jpg", "panel-0001.jpg"]

    output = pq.read_table(tmp_path / "out/manifests/subfigures/part-000000.parquet")
    rows = output.to_pylist()
    assert [row["subcaption"] for row in rows] == ["first", "second"]
    assert all(row["caption_alignment_status"] == "aligned" for row in rows)


def test_resolve_default_and_custom_chat_templates():
    from medpmc_caption_separation_and_alignment.alignment import (
        DEFAULT_MODEL,
        resolve_chat_template,
        resolve_trust_remote_code,
    )

    assert resolve_chat_template(DEFAULT_MODEL, "auto") is None
    assert resolve_chat_template("Qwen/Qwen3-VL", "auto") is None
    assert resolve_chat_template("custom/model", "qwen") == "qwen"
    assert resolve_trust_remote_code(DEFAULT_MODEL, None) is True
    assert resolve_trust_remote_code("custom/model", None) is False
    assert resolve_trust_remote_code("custom/model", True) is True
    assert resolve_trust_remote_code(DEFAULT_MODEL, False) is False


def test_empty_or_truncated_predictions_are_not_retained():
    from medpmc_caption_separation_and_alignment.alignment import _status_for_prediction

    assert _status_for_prediction(["", "b"], 2) == (
        "empty_subcaption",
        "One or more subcaptions are empty",
    )
    assert _status_for_prediction(["a", "b"], 2, finish_reason="length") == (
        "generation_truncated",
        "Generation reached the token limit",
    )
