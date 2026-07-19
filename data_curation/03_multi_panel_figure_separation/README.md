# Multi-panel Figure Separation

This directory implements **Multi-panel Figure Separation**. It detects, orders, and crops individual panels from figures classified as multi-panel by [Multi-panel Figure Detection](../02_multi_panel_figure_detection/).

## Model

- Repository: [`Yale-BIDS-Chen/medpmc-multi-fig-separation-yolov10`](https://huggingface.co/Yale-BIDS-Chen/medpmc-multi-fig-separation-yolov10)
- Checkpoint: `model.pt`
- Framework: YOLOv10 through the Ultralytics interface
- Default confidence threshold: `0.5`

Only rows satisfying the following conditions are processed:

```text
detection_status == "classified"
is_multipanel == true
```

## Installation

Install a PyTorch build compatible with the CUDA driver, followed by YOLOv10 and this package:

```bash
python -m pip install 'git+https://github.com/THU-MIG/yolov10.git'
cd data_curation/03_multi_panel_figure_separation
python -m pip install -e '.[inference,dev]'
pytest
```

The released checkpoint contains a serialized YOLO model object. Load checkpoints only from trusted repositories.

## Run

```bash
medpmc-multi-panel-figure-separation separate \
  --classified-dir <stage2-output>/results/classified \
  --image-root <stage2-output> \
  --output-dir <stage3-output> \
  --device cuda \
  --batch-size 1 \
  --conf 0.5
```

Use `--max-figures` for a bounded validation run.

## Panel crops

The pipeline uses the Ultralytics crop utility with the following defaults:

```text
gain = 1.02
pad = 10
square = false
JPEG quality = 95
```

## Panel ordering and identifiers

`detector_index` records raw detector output order. The canonical panel order is computed as follows:

1. Convert each bounding box to normalized top-left coordinates.
2. Sort boxes by `y_top`.
3. Group adjacent boxes into the same row when `abs(delta y_top) <= 0.05`.
4. Sort boxes within each row by `x_left`.
5. Flatten rows from top to bottom.

The output manifest stores:

```text
historical_order_index = canonical row-major order
spatial_index = historical_order_index
subfigure_index = historical_order_index
image_id = f"{parent_image_id}_{subfigure_index}"
panel_image_id = image_id
```

The field name `historical_order_index` is retained in the manifest schema used by downstream stages. Panel filenames use `subfigure_index`; `detector_index` remains available as provenance.

## Outputs

```text
<stage3-output>/
├── panels/<parent_image_id>/panel-<subfigure_index:04d>.jpg
├── manifests/
│   ├── parents/part-*.parquet
│   └── panels/part-*.parquet
└── separation_summary.json
```

The parent manifest records detector and crop status for each multi-panel figure. The panel manifest records bounding boxes, ordering indices, crop paths, and inherited article and caption metadata.
