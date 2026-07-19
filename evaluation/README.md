# Evaluation

Evaluation resources are organized into two scopes.

## Data curation pipeline evaluation

This section covers the stage-level datasets, metrics, and evaluation procedures used to assess:

1. Initial Screening
2. Multi-panel Figure Detection
3. Multi-panel Figure Separation
4. Caption Separation and Alignment
5. Medical Figure Classification

Detailed evaluation scripts and documentation will be added here.

## MedPMC-CLIP evaluation

The public zero-shot benchmark pipeline for MedPMC-CLIP is maintained as a standalone repository:

- [Yale-BIDS-Chen-Lab/medpmc-clip-eval](https://github.com/Yale-BIDS-Chen-Lab/medpmc-clip-eval)

It provides public benchmark acquisition and preparation, released-checkpoint loading, prediction caching, accuracy, macro F1 and ROC-AUC calculation, optional bootstrap confidence intervals, and paired comparisons with baseline models.

Evaluations that depend on restricted clinical data are not redistributed in this repository.
