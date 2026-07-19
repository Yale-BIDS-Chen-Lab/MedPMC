# Reference representations

Initial Screening stores two reference-text representations for different purposes.

## `reference_texts`

The canonical downstream representation. Nested floating objects such as `<fig>` and `<table-wrap>` are removed before extracting the narrative paragraph. This field should be carried into later curation stages and final released metadata.

## `reference_texts_screening`

A nullable override used only for compatibility with the released PubMedBERT screening model. It preserves recursive paragraph text when that differs from `reference_texts`. A null value means the cleaned and screening-compatible forms are identical.

The default classifier mode is `model-compatible`. `--reference-mode clean` remains available for ablations or future models trained on cleaned references.

