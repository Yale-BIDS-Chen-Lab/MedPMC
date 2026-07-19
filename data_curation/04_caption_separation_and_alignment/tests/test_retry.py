from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from medpmc_caption_separation_and_alignment.alignment import align_directory
from medpmc_caption_separation_and_alignment.backends import InferenceResult


class RetryBackend:
    name = "fake"
    engine = "pytorch"

    def __init__(self):
        self.budgets = []

    def generate_batch(self, requests, *, max_new_tokens=None):
        self.budgets.append(max_new_tokens)
        outputs = []
        for request in requests:
            if max_new_tokens is None:
                outputs.append(
                    InferenceResult(
                        request_id=request.request_id,
                        text="first",
                        metadata={
                            "finish_reason": "length",
                            "generate_token_len": 512,
                            "max_new_tokens": 512,
                        },
                    )
                )
            else:
                outputs.append(
                    InferenceResult(
                        request_id=request.request_id,
                        text="first||second",
                        metadata={
                            "finish_reason": "stop",
                            "generate_token_len": 20,
                            "max_new_tokens": max_new_tokens,
                        },
                    )
                )
        return outputs

    def close(self):
        pass


def _write(path: Path, records: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records, schema=schema), path)


def test_truncated_generation_is_retried_and_recovered(tmp_path: Path):
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

    backend = RetryBackend()
    summary_path = align_directory(
        tmp_path / "parents",
        tmp_path / "panels",
        tmp_path / "out",
        parent_image_root=parent_root,
        panel_image_root=panel_root,
        backend=backend,
        max_new_tokens=512,
        retry_truncated=True,
        retry_max_new_tokens=1024,
    )

    import json

    summary = json.loads(summary_path.read_text())
    assert backend.budgets == [None, 1024]
    assert summary["aligned_parent_rows"] == 1
    assert summary["retry_requests"] == 1
    assert summary["retry_recovered_parent_rows"] == 1

    rows = pq.read_table(tmp_path / "out/manifests/parents/part-000000.parquet").to_pylist()
    assert rows[0]["caption_generation_attempts"] == 2
    assert rows[0]["caption_retried_after_truncation"] is True
    assert rows[0]["caption_max_new_tokens"] == 1024
