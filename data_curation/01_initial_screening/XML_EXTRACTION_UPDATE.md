# XML extraction update

This patch updates the JATS parser after validation with a current PMC XML article.

It adds:

- support for both `pub-id-type="pmcid"` and the older `pub-id-type="pmc"`;
- extraction of `pmcid-ver` and `pmid`;
- preservation of figure labels;
- mixed-content extraction without adding spaces around inline tags such as
  `<sub>`, `<sup>`, and `<italic>`;
- removal of nested figures, tables, and other floating objects from inline
  reference paragraphs; and
- support for the PMC `pmc-license-ref` custom metadata field.

Because the output schema now includes `pmid` and `figure_label`, existing
Intermediate Screening Parquet files should be regenerated:

```bash
medpmc-initial-screening extract \
  ... \
  --force
```

Then rerun the screening step using the regenerated Parquet files.
