# MedPMC

MedPMC is a framework for curating large-scale medical image-text data from biomedical literature. The pipeline combines PubMed Central article metadata and JATS XML with figure screening, multi-panel processing, subcaption alignment, and medical-image classification. The June 2024 release produced the MedPMC-11M dataset and MedPMC-CLIP model.

- **Paper:** [MedPMC: A Systematic Framework for Scaling High-Fidelity Medical Multimodal Data for Foundation Models](https://arxiv.org/abs/2607.07673)
- **Dataset:** [MedPMC-11M](https://huggingface.co/datasets/Yale-BIDS-Chen/medpmc-11m-dataset_jun24_baseline)
- **Model:** [MedPMC-CLIP](https://huggingface.co/Yale-BIDS-Chen/medpmc-clip-l-14_jun24_v1)
- **Resource collection:** [MedPMC on Hugging Face](https://huggingface.co/collections/Yale-BIDS-Chen/medpmc)

## Data curation pipeline

The curation code is organized into five stages. Each stage has its own installable package, command-line interface, tests, and README.

| Stage | Purpose | Released model |
|---|---|---|
| [1. Initial Screening](data_curation/01_initial_screening/) | Screens figure captions and article reference text before image download | [PubMedBERT screening classifier](https://huggingface.co/Yale-BIDS-Chen/medpmc-screening-pubmedbert-caption-reference) |
| [2. Multi-panel Figure Detection](data_curation/02_multi_panel_figure_detection/) | Classifies figures as single-panel or multi-panel | [ViT detector](https://huggingface.co/Yale-BIDS-Chen/medpmc-multi-fig-detection-vit) |
| [3. Multi-panel Figure Separation](data_curation/03_multi_panel_figure_separation/) | Detects, orders, and crops panels from multi-panel figures | [YOLOv10 separator](https://huggingface.co/Yale-BIDS-Chen/medpmc-multi-fig-separation-yolov10) |
| [4. Caption Separation and Alignment](data_curation/04_caption_separation_and_alignment/) | Separates a parent caption and aligns subcaptions to ordered panels | [InternVL caption model](https://huggingface.co/Yale-BIDS-Chen/medpmc-caption-separation-internvl-2.5-4b-mpo) |
| [5. Medical Figure Classification](data_curation/05_medical_figure_classification/) | Filters single-panel figures and aligned subfigures for medical visual content | [ViT medical-figure classifier](https://huggingface.co/Yale-BIDS-Chen/medpmc-med-fig-classification-vit) |

See [data_curation/README.md](data_curation/) for the pipeline-level overview and the README in each stage directory for installation, inputs, outputs, and execution examples.

## Repository structure

```text
MedPMC/
├── data_curation/    # Inference and data-processing code for the five-stage pipeline
├── training/         # Training recipes for MedPMC-CLIP and the curation models
├── evaluation/       # Curation-pipeline and downstream-model evaluation resources
├── analysis/         # Dataset analysis and visualization utilities
├── docs/             # Dataset, model, licensing, and reproducibility documentation
└── assets/           # Figures and other static resources
```

The executable packages under `data_curation/` reproduce curation-time processing and model inference. Training configurations for both MedPMC-CLIP and the models used within the curation pipeline are organized under `training/`. Evaluation resources are separated into curation-pipeline evaluation and downstream MedPMC-CLIP evaluation under `evaluation/`.

## Citation

```bibtex
@article{kim2026medpmc,
  title={MedPMC: A Systematic Framework for Scaling High-Fidelity Medical Multimodal Data for Foundation Models},
  author={Kim, Hyunjae and Kim, Dain and Xiao, Pan and Applebaum, Serina S. and Chung, Younjoon and Ai, Xuguang and Yin, Yu and Jiang, Roy and Du, Yuexi and Wei, Yawen and others},
  journal={arXiv preprint arXiv:2607.07673},
  year={2026}
}
```

## License

The source code in this repository is licensed under the [Apache License 2.0](LICENSE). This license does not apply to datasets, model weights, article content, or third-party software. Those resources remain subject to the licenses and terms specified by their respective repositories and source publications. See [docs/licensing.md](docs/licensing.md) for details.
