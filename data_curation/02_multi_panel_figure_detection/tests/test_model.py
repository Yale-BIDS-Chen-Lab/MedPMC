import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from medpmc_multi_panel_figure_detection import model as model_module
from medpmc_multi_panel_figure_detection.storage import IMAGE_MANIFEST_SCHEMA


class _DummyPatchEmbed:
    img_size = (256, 256)


class DummyDetector(torch.nn.Module):
    patch_embed = _DummyPatchEmbed()
    pretrained_cfg = {
        "input_size": (3, 256, 256),
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "crop_pct": 0.9,
        "crop_mode": "center",
    }

    def forward(self, inputs):
        logits = torch.zeros((inputs.shape[0], 2), device=inputs.device)
        logits[:, 1] = 2.0
        return logits


def _manifest_row(local_image_path: str, *, status: str):
    row = {name: None for name in IMAGE_MANIFEST_SCHEMA.names}
    row.update(
        {
            "pmcid": "PMC1",
            "article_version": "PMC1.1",
            "pmid": "1",
            "article_title": "test",
            "journal_title": "test",
            "license": "CC BY",
            "figure_id": "F1",
            "figure_label": "Figure 1",
            "caption": "caption",
            "reference_texts": [],
            "reference_texts_screening": [],
            "image_hrefs": ["gr1.png"],
            "source_xml": "PMC1.xml",
            "screening_score": 0.9,
            "screening_label": 1,
            "retained": True,
            "screening_model": "screen",
            "screening_reference_mode": "model-compatible",
            "screening_threshold": 0.5,
            "screening_max_length": 512,
            "screening_positive_label": 1,
            "resolved_article_version": "PMC1.1",
            "article_resolution_method": "explicit_version",
            "article_candidate_versions": 1,
            "article_metadata_key": "metadata/PMC1.1.json",
            "article_is_retracted": False,
            "article_metadata_license": "CC BY",
            "selected_image_href": "gr1.png",
            "image_selection_method": "first_graphic_document_order",
            "image_candidate_count": 1,
            "image_href_match_type": "basename",
            "image_s3_url": "s3://bucket/PMC1.1/gr1.png",
            "image_s3_key": "PMC1.1/gr1.png",
            "image_expected_md5": "",
            "local_image_path": local_image_path,
            "image_status": status,
            "image_transfer": "downloaded",
            "image_error": "" if status == "ready" else "not ready",
            "image_width": 16,
            "image_height": 16,
            "image_format": "PNG",
            "image_mode": "RGB",
            "image_md5": "",
        }
    )
    return row


def test_detect_directory_with_dummy_model(tmp_path: Path, monkeypatch):
    image_root = tmp_path / "run"
    image_path = image_root / "images" / "PMC1.1" / "gr1.png"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(image_path)

    manifest_dir = image_root / "manifests" / "figures"
    manifest_dir.mkdir(parents=True)
    rows = [
        _manifest_row("images/PMC1.1/gr1.png", status="ready"),
        _manifest_row("", status="missing_from_metadata"),
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=IMAGE_MANIFEST_SCHEMA),
        manifest_dir / "part-000000.parquet",
    )

    monkeypatch.setattr(
        model_module,
        "load_detector",
        lambda *args, **kwargs: (DummyDetector(), "dummy", tmp_path / "model.pth.tar"),
    )
    summary_path = model_module.detect_directory(
        manifest_dir,
        image_root / "results",
        image_root=image_root,
        device="cpu",
        batch_size=2,
        loader_workers=0,
    )
    summary = json.loads(summary_path.read_text())
    assert summary["classified_rows"] == 1
    assert summary["multipanel_rows"] == 1
    assert summary["failed_rows"] == 1

    classified = pq.read_table(
        image_root / "results" / "classified" / "part-000000.parquet"
    ).to_pylist()
    assert classified[0]["detection_status"] == "classified"
    assert classified[0]["is_multipanel"] is True
    assert classified[1]["detection_status"] == "skipped_image_not_ready"


def test_resolve_image_size_from_checkpoint():
    model = DummyDetector()
    assert model_module._resolve_image_size(model, None) == 256
    assert model_module._resolve_image_size(model, 256) == 256


def test_reject_mismatched_image_size():
    import pytest

    with pytest.raises(ValueError, match="expects 256x256"):
        model_module._resolve_image_size(DummyDetector(), 224)


def test_resolve_timm_data_config_uses_model_defaults():
    config = model_module._resolve_timm_data_config(DummyDetector(), None)
    assert tuple(config["input_size"]) == (3, 256, 256)
    assert config["interpolation"] == "bicubic"
    assert tuple(config["mean"]) == (0.485, 0.456, 0.406)
    assert tuple(config["std"]) == (0.229, 0.224, 0.225)


def test_default_threshold_matches_binary_argmax_tie_behavior():
    assert model_module._label_from_positive_score(
        0.500001, threshold=0.5, positive_label=1
    ) == 1
    assert model_module._label_from_positive_score(
        0.5, threshold=0.5, positive_label=1
    ) == 0
