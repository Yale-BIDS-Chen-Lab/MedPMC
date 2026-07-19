# 1. Initial Screening

This directory provides the **Initial Screening** stage of the MedPMC data curation pipeline.

The stage identifies candidate medical figures **before downloading article images**. For each figure, it extracts the figure caption and the inline text passages that reference that figure from the article XML, concatenates them using the released model's expected format, and applies the MedPMC PubMedBERT classifier.

The default model is:

- [`Yale-BIDS-Chen/medpmc-screening-pubmedbert-caption-reference`](https://huggingface.co/Yale-BIDS-Chen/medpmc-screening-pubmedbert-caption-reference)

The released model predicts:

```text
0: Non-medical
1: Medical
```

## Why XML rather than plain text?

This stage uses PMC JATS XML because it preserves the relationships required for figure-level screening:

- figure identifiers;
- full figure captions;
- inline figure references from the article body;
- image file references;
- article identifiers and metadata; and
- structured license information.

Plain-text PMC files do not reliably preserve these figure-to-reference relationships.

## Environment strategy

All commands in this directory use the same environment. Users do **not** need separate environments for downloading, XML extraction, and model inference within Initial Screening.

Across the complete MedPMC curation pipeline, we recommend:

1. a lightweight common base for downloading, XML parsing, manifests, and tabular I/O;
2. stage-specific dependencies installed only when needed; and
3. a separate environment for stages with large or tightly pinned multimodal-model dependencies, particularly Caption Separation and Alignment.

For Initial Screening, create one environment:

```bash
conda create -n medpmc-curation python=3.11 -y
conda activate medpmc-curation
```

For download and XML extraction only:

```bash
cd data_curation/01_initial_screening
pip install -e .
```

For the complete stage, including PubMedBERT inference, first install a PyTorch build compatible with the NVIDIA driver on the execution node, then run:

```bash
pip install -e '.[inference]'
```

Before a large GPU run, verify the actual PyTorch runtime rather than relying only on `nvcc -V`:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

The installed CUDA toolkit and the NVIDIA driver are separate components. A recent `nvcc` does not guarantee that a PyTorch wheel built for the same CUDA version can be initialized by the node's driver.

For development tests:

```bash
pip install -e '.[inference,dev]'
pytest
```

## Two supported input modes

### 1. PMC bulk XML package

Use this mode for a large baseline or incremental PMC snapshot. It is the preferred approach when screening many articles.

The example below uses the January 23, 2026 non-commercial XML baseline package for the `PMC012xxxxxx` range:

```text
oa_noncomm_xml.PMC012xxxxxx.baseline.2026-01-23.filelist.csv
oa_noncomm_xml.PMC012xxxxxx.baseline.2026-01-23.tar.gz
```

Package dates, PMCID ranges, filenames, and directory layouts can change between releases. The exact URLs in this README are therefore examples for a specific snapshot, not permanent discovery endpoints. Before starting a new run, consult the official NLM PMC FTP or cloud dataset documentation and select an XML archive together with its matching file list.

The URLs below reproduce a specific archived PMC OA XML snapshot from the `deprecated/` layout. Consult the current NLM PMC dataset documentation when starting a new run. The PubMed annual-baseline download page covers citation records and should not be confused with the PMC OA full-text XML packages used here.

### 2. List of PMC IDs

Use this mode when screening a selected list of articles. It retrieves the latest available article version from the current NLM PMC Open Data S3 structure.

This mode is convenient for small targeted collections, but it is slower than processing a bulk archive because each PMCID requires metadata lookup and individual XML retrieval.

## End-to-end one-command workflows

`run-bulk` and `run-pmcids` perform the complete workflow, including PubMedBERT screening. Use them on a node where the requested inference device is available. For CPU/GPU-separated HPC workflows, use the `prepare-*` commands in the next section.

### Bulk package

```bash
medpmc-initial-screening run-bulk \
  --archive-url \
  https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.PMC012xxxxxx.baseline.2026-01-23.tar.gz \
  --filelist-url \
  https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.PMC012xxxxxx.baseline.2026-01-23.filelist.csv \
  --output-dir outputs/pmc012_noncomm_2026_01_23
```

This command:

1. downloads the archive and file list;
2. checks article-level licenses and retraction status;
3. streams the XML archive without extracting every article to disk;
4. extracts caption and inline reference text for each figure;
5. writes sharded Parquet intermediates;
6. runs the released PubMedBERT classifier; and
7. writes both complete and retained screening outputs.

The example archive is roughly 5–6 GB. For a small smoke test, use the `PMC000xxxxxx` package:

```bash
medpmc-initial-screening run-bulk \
  --archive-url \
  https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.PMC000xxxxxx.baseline.2026-01-23.tar.gz \
  --filelist-url \
  https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.PMC000xxxxxx.baseline.2026-01-23.filelist.csv \
  --output-dir outputs/smoke_test
```

### Selected PMC IDs

Prepare one PMCID per line:

```text
PMC10009402
PMC12855588
```

Then run:

```bash
medpmc-initial-screening run-pmcids \
  --pmcid-file examples/pmcids.txt \
  --output-dir outputs/selected_pmcids
```

Anonymous access to the NLM public S3 bucket is used; an AWS account is not required.

## Recommended HPC workflow: prepare on CPU, screen on GPU

For large packages, separate acquisition/XML processing from model inference. This avoids accidentally running PubMedBERT on a login or CPU-only node.

### Prepare a bulk package on a CPU node

```bash
medpmc-initial-screening prepare-bulk \
  --archive-url \
  https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.PMC012xxxxxx.baseline.2026-01-23.tar.gz \
  --filelist-url \
  https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_bulk/oa_noncomm/xml/oa_noncomm_xml.PMC012xxxxxx.baseline.2026-01-23.filelist.csv \
  --output-dir outputs/pmc012_noncomm_2026_01_23
```

This performs download, article-level license checks, and XML extraction, but does not load the screening model.

### Screen the prepared output on a GPU node

```bash
medpmc-initial-screening screen \
  --input-dir outputs/pmc012_noncomm_2026_01_23/intermediate/figure_text \
  --output-dir outputs/pmc012_noncomm_2026_01_23/results \
  --device cuda \
  --batch-size 256
```

Use an explicit `--device cuda` for GPU jobs. If CUDA cannot be initialized, the command stops with an error instead of silently selecting CPU. Start with a conservative batch size and increase it while monitoring GPU utilization and memory.

A selected-PMCID workflow is also available:

```bash
medpmc-initial-screening prepare-pmcids \
  --pmcid-file examples/pmcids.txt \
  --output-dir outputs/selected_pmcids
```

## Advanced: run individual steps

The one-command workflows are recommended for most users. Run steps separately when you need to:

- inspect license decisions before parsing;
- retry a failed download;
- change XML parsing without repeating acquisition;
- rerun the classifier with a different threshold or checkpoint;
- process data on separate CPU and GPU nodes; or
- resume a large job from existing intermediates.

### A. Download a bulk package

```bash
medpmc-initial-screening download-bulk \
  --archive-url URL_TO_XML_TAR_GZ \
  --filelist-url URL_TO_FILELIST_CSV \
  --output-dir work/raw
```

### B. Download XML for a PMCID list

```bash
medpmc-initial-screening download-pmcids \
  --pmcid-file examples/pmcids.txt \
  --output-dir work/raw \
  --workers 8
```

### C. Extract figure-level text from a bulk archive

```bash
medpmc-initial-screening extract \
  --xml-tar work/raw/ARCHIVE_NAME.tar.gz \
  --filelist work/raw/FILELIST_NAME.csv \
  --output-dir work/intermediate/figure_text
```

### D. Extract figure-level text from downloaded XML files

```bash
medpmc-initial-screening extract \
  --xml-dir work/raw/xml \
  --metadata-dir work/raw/metadata \
  --output-dir work/intermediate/figure_text
```

### E. Run Initial Screening inference

```bash
medpmc-initial-screening screen \
  --input-dir work/intermediate/figure_text \
  --output-dir work/results \
  --device cuda \
  --batch-size 64 \
  --threshold 0.5 \
  --reference-mode model-compatible
```

The command displays both shard-level and within-shard inference-batch progress. It also prints the selected device, batch size, and maximum sequence length before loading the model.

To use another compatible checkpoint:

```bash
medpmc-initial-screening screen \
  --input-dir work/intermediate/figure_text \
  --output-dir work/results \
  --model Yale-BIDS-Chen/medpmc-screening-pubmedbert-caption-reference
```

## License filtering

The default article-level license allowlist matches the MedPMC release criteria:

```text
CC BY
CC BY-NC
CC BY-NC-SA
CC0
CC BY-SA
```

The following are excluded by default:

```text
CC BY-ND
CC BY-NC-ND
NO-CC CODE
missing or unrecognized licenses
```

The PMC bulk directories group articles broadly by licensing category. For example, the non-commercial package may include both `CC BY-NC` and `CC BY-NC-ND`. Therefore, selecting a package is **not** a substitute for article-level license filtering.

For bulk packages, the NLM file list is used as the primary source of license and retraction metadata. For selected PMC IDs, the current NLM S3 JSON metadata is used. XML license text is used only as a fallback when structured metadata is unavailable.

Users remain responsible for complying with each source article's license terms.

To change the allowlist:

```bash
medpmc-initial-screening extract \
  ... \
  --allowed-licenses "CC BY,CC0"
```

## Model input

The released caption-plus-reference model expects:

```text
"Caption": {figure_caption}
"Reference Text": {reference_text_1}
{reference_text_2}
...
```

The implementation reproduces this format and uses a maximum sequence length of 512 tokens.

### Two reference representations

Reference text is defined from the `<p>` element containing an `<xref ref-type="fig">` that points to the figure. JATS permits floating objects such as `<fig>` and `<table-wrap>` to be nested inside that paragraph, so the pipeline retains two representations:

- `reference_texts`: a cleaned narrative representation that removes nested floating objects. This is the canonical field for downstream processing and final dataset metadata.
- `reference_texts_screening`: a nullable screening-specific override that recursively preserves the full paragraph text when it differs from the cleaned representation. This matches the preprocessing used to train the released Initial Screening model. A null value means the two representations are identical.

Screening uses `--reference-mode model-compatible` by default. This selects `reference_texts_screening` when present and otherwise falls back to `reference_texts`. The cleaned representation can be used for an ablation or a newly trained model:

```bash
medpmc-initial-screening screen \
  --input-dir work/intermediate/figure_text \
  --output-dir work/results_clean_reference \
  --reference-mode clean \
  --device cuda
```

This separation is intentional. In validation against 59 released MedPMC figures, cleaned and released captions agreed after whitespace normalization for all figures. Using only cleaned reference text retained 57 of 59 final positive figures; the model-compatible recursive representation moved both missed figures above the original 0.5 screening threshold. Both exceeded the 512-token limit, showing that the representation can change the truncated sequence seen by the classifier.

The parser additionally supports both older and current PMC article-ID conventions (`pmc`, `pmcid`, and versioned PMCID fields), preserves inline scientific notation without inserting spaces around tags, and splits multi-target `rid` attributes when a reference points to more than one figure.

## Outputs

A one-command run creates:

```text
<output-dir>/
├── raw/
│   ├── archive and file list, or downloaded XML files
│   ├── metadata/
│   └── download_manifest.jsonl
├── intermediate/
│   └── figure_text/
│       ├── part-000000.parquet
│       ├── part-000001.parquet
│       └── extraction_summary.json
└── results/
    ├── screened/
    │   └── part-*.parquet
    ├── retained/
    │   └── part-*.parquet
    └── screening_summary.json
```

Figure-level records include:

- `pmcid`
- `article_version`
- `pmid`
- `article_title`
- `journal_title`
- `license`
- `figure_id`
- `figure_label`
- `caption`
- `reference_texts` (cleaned downstream representation)
- `reference_texts_screening` (nullable model-compatible override)
- `image_hrefs`
- `source_xml`

Screening outputs additionally include:

- `screening_score`
- `screening_label`
- `retained`
- `screening_model`
- `screening_reference_mode`
- `screening_threshold`
- `screening_max_length`
- `screening_positive_label`

The downstream handoff should use the cleaned `reference_texts` field. The `image_hrefs` and `pmcid` fields are retained so candidate images can be acquired for **Multi-panel Figure Detection**, the next stage.

## Resume and large-scale processing

Downloads are reused when files already exist. Extraction and screening summaries are also reused unless `--force` is supplied.

For large packages:

- use scratch storage rather than a home directory;
- keep the archive compressed;
- stream XML directly from the archive;
- use sharded Parquet outputs;
- perform XML extraction on CPU nodes; and
- run model inference separately on a GPU node when appropriate.

Example:

```bash
# CPU node
medpmc-initial-screening prepare-bulk ...

# GPU node
medpmc-initial-screening screen ... --device cuda
```

Screening output is written one Parquet shard at a time. Completed shard pairs under `results/screened/` and `results/retained/` are reused only when their model, reference mode, threshold, maximum length, and positive-label configuration match the requested run. Otherwise the command stops and asks for `--force` or a new output directory.

## Important scope note

Initial Screening does not download figure images. It uses article text to identify candidate figures before the more storage-intensive image-processing stages.

## References

- MedPMC paper: https://arxiv.org/abs/2607.07673
- PMC FTP service: https://pmc.ncbi.nlm.nih.gov/tools/ftp/
- PMC datasets on AWS: https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/
- PMC Open Access Subset: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/
- PubMed annual baseline (citation records; not PMC full text): https://pubmed.ncbi.nlm.nih.gov/download/#annual-baseline
- Released Initial Screening model: https://huggingface.co/Yale-BIDS-Chen/medpmc-screening-pubmedbert-caption-reference
