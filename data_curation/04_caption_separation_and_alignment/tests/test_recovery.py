from __future__ import annotations

from pathlib import Path
import json

import pyarrow as pa
import pyarrow.parquet as pq

from medpmc_caption_separation_and_alignment.alignment import align_directory
from medpmc_caption_separation_and_alignment.backends import InferenceResult
from medpmc_caption_separation_and_alignment.recovery import (
    merge_retry_runs,
    parent_ids_with_status,
    retry_directory,
)


class StaticBackend:
    name = "fake"
    engine = "batch"

    def __init__(self, *, text: str, finish_reason: str):
        self.text = text
        self.finish_reason = finish_reason

    def generate_batch(self, requests, *, max_new_tokens=None):
        return [
            InferenceResult(
                request_id=request.request_id,
                text=self.text,
                metadata={
                    "finish_reason": self.finish_reason,
                    "generate_token_len": 16,
                    "max_new_tokens": int(max_new_tokens or 0),
                },
            )
            for request in requests
        ]

    def close(self):
        pass


def _write(path: Path, records: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path)


def _inputs(tmp_path: Path):
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
        tmp_path / "parents" / "part.parquet",
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
        tmp_path / "panels" / "part.parquet",
        [
            {
                "pmcid": "PMC1",
                "parent_image_id": "PARENT",
                "image_id": f"PARENT_{index}",
                "subfigure_index": index,
                "historical_order_index": index,
                "detector_index": index,
                "local_panel_path": f"panels/PARENT/panel-{index:04d}.jpg",
                "crop_status": "ready",
                "caption": "Main caption",
            }
            for index in range(2)
        ],
        panel_schema,
    )
    return parent_root, panel_root


def test_separate_retry_and_merge(tmp_path: Path):
    parent_root, panel_root = _inputs(tmp_path)
    base = tmp_path / "base"
    retry = tmp_path / "retry4096"
    final = tmp_path / "final"

    align_directory(
        tmp_path / "parents",
        tmp_path / "panels",
        base,
        parent_image_root=parent_root,
        panel_image_root=panel_root,
        backend=StaticBackend(text="partial", finish_reason="length"),
    )
    assert parent_ids_with_status(base) == {"PARENT"}
    assert (base / "retry_candidates.jsonl").read_text().count("PARENT") == 1

    retry_directory(
        base,
        tmp_path / "parents",
        tmp_path / "panels",
        retry,
        parent_image_root=parent_root,
        panel_image_root=panel_root,
        max_new_tokens=4096,
        backend=StaticBackend(text="first||second", finish_reason="stop"),
    )
    retry_rows = pq.read_table(
        retry / "manifests/parents/part-000000.parquet"
    ).to_pylist()
    assert retry_rows[0]["caption_alignment_status"] == "aligned"
    assert retry_rows[0]["caption_run_kind"] == "retry"

    merge_retry_runs(base, [retry], final)
    final_rows = pq.read_table(
        final / "manifests/parents/part-000000.parquet"
    ).to_pylist()
    assert final_rows[0]["caption_alignment_status"] == "aligned"
    assert final_rows[0]["caption_final_prediction_source"] == "retry"
    assert final_rows[0]["caption_base_status"] == "generation_truncated"
    assert final_rows[0]["caption_retry_attempt_count"] == 1
    assert final_rows[0]["caption_latest_attempt_status"] == "aligned"
    assert final_rows[0]["caption_latest_attempt_source_run_dir"] == str(retry)
    summary = json.loads(
        (final / "caption_alignment_merge_summary.json").read_text(encoding="utf-8")
    )
    assert summary["selected_status_counts"] == {"aligned": 1}
    assert summary["latest_attempt_status_counts"] == {"aligned": 1}
    assert summary["retry_transition_counts"] == {
        "generation_truncated->aligned": 1
    }
    assert summary["selected_prediction_source_counts"] == {"retry": 1}
    final_subfigures = pq.read_table(
        final / "manifests/subfigures/part-000000.parquet"
    ).to_pylist()
    assert [row["subcaption"] for row in final_subfigures] == ["first", "second"]
