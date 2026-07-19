# Caption Separation and Alignment

This directory implements **Caption Separation and Alignment**. It separates a compound-figure caption into subcaptions and aligns one subcaption to each ordered panel produced by [Multi-panel Figure Separation](../03_multi_panel_figure_separation/).

## Model and design

The default model is:

- [`Yale-BIDS-Chen/medpmc-caption-separation-internvl-2.5-4b-mpo`](https://huggingface.co/Yale-BIDS-Chen/medpmc-caption-separation-internvl-2.5-4b-mpo)

The curation logic is backend-independent. The default LMDeploy backend supports compatible multimodal language models through configurable model, engine, and prompt options.

Panels are supplied in ascending `historical_order_index`:

```text
compound figure, subfigure 0, ..., subfigure N-1
```

A parent is retained only when the prediction:

1. contains exactly one `||`-separated subcaption per panel;
2. contains no empty subcaption; and
3. completes without reaching the generation token limit.

Rejected predictions remain in the parent manifest and raw JSONL outputs for inspection or retry.

## Installation

A separate environment is recommended because multimodal inference runtimes may require tightly coupled PyTorch, Transformers, CUDA, and attention-kernel versions.

```bash
cd data_curation/04_caption_separation_and_alignment
conda env create -f environment.stage4.yml
conda activate medpmc-stage4
python -m pip install -e '.[inference,dev]'
pytest
```

## First pass

```bash
medpmc-caption-separation-and-alignment run \
  --parent-dir <stage3-output>/manifests/parents \
  --panel-dir <stage3-output>/manifests/panels \
  --parent-image-root <stage2-output> \
  --panel-image-root <stage3-output> \
  --output-dir <stage4-first-pass-output> \
  --model Yale-BIDS-Chen/medpmc-caption-separation-internvl-2.5-4b-mpo \
  --engine turbomind \
  --max-new-tokens 1024 \
  --inference-batch-size 2 \
  --engine-max-batch-size 2 \
  --max-images-per-batch 20 \
  --vision-max-batch-size 8 \
  --image-loader-workers 8 \
  --no-retry-truncated \
  --tp 1
```

The pipeline groups requests with similar panel counts, limits total images per batch, and loads images concurrently. Run retries in a separate output directory so completed first-pass results remain unchanged and retry-specific memory settings can be adjusted independently.

The first-pass output contains:

```text
<stage4-first-pass-output>/
├── manifests/parents/
├── manifests/subfigures/
├── raw_predictions/
├── retry_candidates.jsonl
└── caption_alignment_summary.json
```

`retry_candidates.jsonl` lists parents whose generation reached the selected token limit.

## Output token budgets

`--max-new-tokens` sets the maximum output length for the current pass. It does not guarantee that every caption will fit, and short generations stop normally at EOS without consuming the full budget.

The required budget depends on caption length, number of panels, visual-token count, batch composition, session length, and backend cache configuration. Users should select a budget and batch configuration appropriate for their data and hardware.

## Retry truncated predictions

Retry only predictions that were truncated in a previous run. The retry budget is user-controlled and can be increased across multiple passes.

```bash
medpmc-caption-separation-and-alignment retry \
  --source-run-dir <stage4-first-pass-output> \
  --parent-dir <stage3-output>/manifests/parents \
  --panel-dir <stage3-output>/manifests/panels \
  --parent-image-root <stage2-output> \
  --panel-image-root <stage3-output> \
  --output-dir <stage4-retry-output> \
  --model Yale-BIDS-Chen/medpmc-caption-separation-internvl-2.5-4b-mpo \
  --engine turbomind \
  --max-new-tokens 2048 \
  --inference-batch-size 1 \
  --engine-max-batch-size 1 \
  --max-images-per-batch 16 \
  --vision-max-batch-size 4 \
  --cache-max-entry-count 0.3 \
  --image-loader-workers 4 \
  --tp 1
```

By default, `retry` selects `generation_truncated` rows. Additional source statuses can be selected by repeating `--status`.

A retry output can be used as the source of another retry, for example with a larger `--max-new-tokens` value. Larger output budgets generally require smaller request batches and more conservative cache settings.

## Merge retry results

```bash
medpmc-caption-separation-and-alignment merge \
  --base-run-dir <stage4-first-pass-output> \
  --retry-run-dir <stage4-retry-output> \
  --output-dir <stage4-final-output>
```

Retry directories are applied in the order provided. Only aligned retry rows replace the previously selected prediction. The merged output records prediction provenance, retry history, selected-result status, and latest-attempt status.

## Memory and throughput options

The main performance controls are:

- `--max-new-tokens`: maximum output budget for the current pass;
- `--inference-batch-size`: maximum parent requests per call;
- `--max-images-per-batch`: compound and subfigure image cap per call;
- `--vision-max-batch-size`: vision encoder batch limit;
- `--engine-max-batch-size`: backend request capacity;
- `--cache-max-entry-count`: LMDeploy KV-cache allocation;
- `--session-len`: maximum sequence length supported by the engine.

When memory is limited, reduce request batch size, image count, vision batch size, or cache allocation. Start with a small `--max-parents` run when using a new model or hardware configuration.

## Resume behavior

- `--resume` skips completed output shards.
- `--force` recomputes selected outputs.
- `run --retry-truncated` is available for an inline retry, while the separate `retry` command provides independent output and resource settings.
