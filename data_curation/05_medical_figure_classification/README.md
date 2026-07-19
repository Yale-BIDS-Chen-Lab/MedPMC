# Medical Figure Classification

This directory implements **Medical Figure Classification**, the final stage of the MedPMC curation pipeline. It applies a two-class ViT classifier to both single-panel figures produced by [Multi-panel Figure Detection](../02_multi_panel_figure_detection/) and aligned subfigures produced by [Caption Separation and Alignment](../04_caption_separation_and_alignment/).

The released class order is:

```text
0 = non-medical
1 = medical
```

Rows are retained when the medical-class score exceeds the configured threshold.

## Model

- Repository: [`Yale-BIDS-Chen/medpmc-med-fig-classification-vit`](https://huggingface.co/Yale-BIDS-Chen/medpmc-med-fig-classification-vit)
- Checkpoint: `model.pth.tar`
- Architecture: `vit_base_patch16_rope_reg1_gap_256.sbb_in1k`
- Output classes: 2

## Installation

```bash
cd data_curation/05_medical_figure_classification
python -m pip install -e '.[inference,dev]'
pytest
```

## Inputs

### Single-panel figures

Use the `results/singlepanel` manifest from Multi-panel Figure Detection together with its image root.

### Aligned subfigures

Use the final merged `manifests/subfigures` output from Caption Separation and Alignment together with the panel image root from Multi-panel Figure Separation.

Only subfigure rows with a successful caption alignment, a ready crop, and a nonempty subcaption are eligible.

## Model preprocessing

The model is created with `pretrained=False`, followed by strict loading of the released checkpoint. The evaluation transform is resolved from the timm model configuration. The released checkpoint uses:

```text
input size: 256 × 256
interpolation: bicubic
mean: (0.5, 0.5, 0.5)
standard deviation: (0.5, 0.5, 0.5)
crop percentage: 0.95
crop mode: center
```

The default retention rule is `medical_score > 0.5` for class index `1`. The threshold can be changed with `--threshold`.

## Run

```bash
medpmc-medical-figure-classification run \
  --singlepanel-dir <stage2-output>/results/singlepanel \
  --singlepanel-image-root <stage2-output> \
  --subfigure-dir <stage4-final-output>/manifests/subfigures \
  --subfigure-image-root <stage3-output> \
  --output-dir <stage5-output> \
  --model Yale-BIDS-Chen/medpmc-med-fig-classification-vit \
  --checkpoint-filename model.pth.tar \
  --batch-size 256 \
  --loader-workers 8 \
  --device cuda
```

Batch size and worker count should be selected according to available memory and storage throughput. `--amp` enables CUDA float16 autocast.

## Outputs

```text
<stage5-output>/
├── classification_config.json
├── medical_figure_classification_summary.json
└── manifests/
    ├── classified/
    ├── medical/
    └── non_medical/
```

Each output row preserves source metadata, image and caption fields, the medical score, predicted class, model provenance, and classification status.

## Useful options

- `--batch-size`: inference batch size
- `--loader-workers`: image-loading worker count
- `--amp`: CUDA float16 autocast
- `--threshold`: medical-class score threshold; default `0.5`
- `--max-images`: bounded validation run
- `--force`: recompute existing outputs

The resolved timm preprocessing configuration is written to `classification_config.json` and the run summary.
