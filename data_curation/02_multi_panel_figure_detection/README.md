# Multi-panel Figure Detection

This directory implements **Multi-panel Figure Detection**, the second stage of the MedPMC data curation pipeline. It downloads figure images retained by [Initial Screening](../01_initial_screening/), validates the image files, and classifies each figure as single-panel or multi-panel.

The default model is:

- [`Yale-BIDS-Chen/medpmc-multi-fig-detection-vit`](https://huggingface.co/Yale-BIDS-Chen/medpmc-multi-fig-detection-vit)

The released class order is:

```text
0 = single-panel
1 = multi-panel
```

## Installation

```bash
cd data_curation/02_multi_panel_figure_detection
python -m pip install -e '.[inference,dev]'
pytest
```

Install a PyTorch build compatible with the CUDA driver on the execution node before installing the inference dependencies.

## Inputs

The stage reads the retained Parquet records produced by Initial Screening. Each record includes article metadata, figure identifiers, the main caption, reference text, and one or more JATS image references.

Stage 1 is text-only. Images are downloaded here only for retained figures, reducing unnecessary network transfer and storage.

## PMC image resolution

The default implementation resolves images from the PMC Open Data object metadata. Article objects may include multiple versions. When an article version is not available in the upstream manifest, the pipeline compares available metadata versions with the JATS image references and selects the best match.

For a JATS `<fig>` containing multiple `<graphic>` elements, the first graphic in XML document order is selected as the primary figure image. All extracted image references remain available in the manifest for provenance.

## Prepare images

```bash
medpmc-multi-panel-figure-detection prepare-images \
  --input-dir <stage1-output>/results/retained \
  --output-dir <stage2-output> \
  --workers 16
```

The image preparation manifest records download status, selected article version and image reference, checksum information, local path, dimensions, and file format. Download and decode failures remain in the manifest with explicit status values.

## Run classification

```bash
medpmc-multi-panel-figure-detection detect \
  --manifest-dir <stage2-output>/manifests/figures \
  --output-dir <stage2-output>/results \
  --device cuda \
  --batch-size 256 \
  --loader-workers 4
```

The complete preparation and classification workflow can also be run with one command:

```bash
medpmc-multi-panel-figure-detection run \
  --input-dir <stage1-output>/results/retained \
  --output-dir <stage2-output> \
  --workers 16 \
  --device cuda \
  --batch-size 256 \
  --loader-workers 4
```

Keeping preparation and classification separate can be useful on clusters where network/CPU work and GPU inference run on different nodes.

## Model preprocessing

The classifier uses architecture `vit_base_patch16_rope_reg1_gap_256.sbb_in1k` with two output classes. Preprocessing is resolved from the loaded timm model configuration, including input size, interpolation, normalization, crop percentage, and crop mode. The released checkpoint uses `256 × 256` inputs.

`--image-size` is optional. When supplied, it is validated against the checkpoint configuration. `--amp` enables mixed-precision inference and may improve throughput or reduce memory use.

## Outputs

```text
<stage2-output>/
├── images/
│   └── PMCxxxxxxx.version/
├── manifests/
│   └── figures/
│       └── part-*.parquet
├── image_preparation_summary.json
└── results/
    ├── classified/
    ├── multipanel/
    ├── singlepanel/
    └── detection_summary.json
```

Classification output fields include:

```text
multipanel_score
multipanel_label
is_multipanel
detection_status
detection_model
detection_architecture
detection_threshold
```

## Resume behavior

- Existing images with matching checksums are reused.
- Downloads are written to temporary files and renamed after validation.
- Existing manifests and completed outputs are reused unless `--force` is supplied.
- Use `--max-figures` for a bounded validation run.

## Pipeline handoff

- `results/multipanel/` is the input to [Multi-panel Figure Separation](../03_multi_panel_figure_separation/).
- `results/singlepanel/` bypasses panel separation and is used later by [Medical Figure Classification](../05_medical_figure_classification/).
