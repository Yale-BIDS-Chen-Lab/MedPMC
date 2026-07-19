# Training

This directory documents model-training reproducibility for two groups of models:

1. **MedPMC-CLIP**, trained on the curated MedPMC image-text corpus; and
2. **data curation models**, used within the five-stage pipeline.

The packages under `data_curation/` implement curation-time processing and inference. They are not the original training implementations. The models were trained with existing open-source frameworks, so this directory records the framework versions, configurations, data formats, hyperparameters, computing environments, and implementation modifications needed to reproduce training with those frameworks.

## MedPMC-CLIP

Documentation covers model initialization, input-data preparation, training objective, optimizer and learning-rate schedule, image resolution and augmentation, distributed-training settings, checkpoint selection, and software versions.

## Data curation models

Training documentation is organized around the canonical pipeline stages:

1. Initial Screening
2. Multi-panel Figure Detection
3. Multi-panel Figure Separation
4. Caption Separation and Alignment
5. Medical Figure Classification

Detailed configurations and links to the corresponding upstream training frameworks will be documented here.
