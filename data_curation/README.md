# Data Curation

This directory contains the inference and data-processing code for the five-stage MedPMC curation pipeline.

## Pipeline stages

1. [Initial Screening](01_initial_screening/)
2. [Multi-panel Figure Detection](02_multi_panel_figure_detection/)
3. [Multi-panel Figure Separation](03_multi_panel_figure_separation/)
4. [Caption Separation and Alignment](04_caption_separation_and_alignment/)
5. [Medical Figure Classification](05_medical_figure_classification/)

The stage names above are the canonical names used throughout the repository. Each stage is packaged independently and documents its inputs, outputs, dependencies, command-line interface, model checkpoint, and validation behavior.

## Data flow

```text
PMC article metadata and JATS XML
  └─ 1. Initial Screening
       └─ retained candidate figures
            └─ 2. Multi-panel Figure Detection
                 ├─ single-panel figures ───────────────────────────────┐
                 └─ multi-panel figures                                │
                      └─ 3. Multi-panel Figure Separation              │
                           └─ ordered subfigures                        │
                                └─ 4. Caption Separation and Alignment │
                                     └─ aligned subfigure-caption pairs│
                                                                        └─ 5. Medical Figure Classification
```

Stage 4 uses a separate environment because its multimodal inference dependencies are more tightly pinned. The other stages can generally share a lighter `medpmc-curation` environment, provided the installed PyTorch build is compatible with the execution hardware.

## Tested Environment

The curation pipeline was tested on the following environment:

- Ubuntu 22.04
- Python 3.11
- 4 CPU cores
- 128 GB RAM
- NVIDIA A40 GPU
- CUDA 12.6/12.8

Additional testing on higher-performance computing environments, including NVIDIA H100 GPUs, will be added.
