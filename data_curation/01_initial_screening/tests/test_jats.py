from medpmc_initial_screening.jats import format_model_input, parse_article_xml


XML = b"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink">
  <front>
    <journal-meta><journal-title>Example Journal</journal-title></journal-meta>
    <article-meta>
      <article-id pub-id-type="pmcid">PMC12345</article-id>
      <article-id pub-id-type="pmcid-ver">PMC12345.2</article-id>
      <article-id pub-id-type="pmid">99999</article-id>
      <title-group><article-title>Example article</article-title></title-group>
      <permissions>
        <license>
          <license-p>This article is distributed under the CC BY 4.0 license.</license-p>
        </license>
      </permissions>
      <custom-meta-group>
        <custom-meta>
          <meta-name>pmc-license-ref</meta-name>
          <meta-value>CC BY</meta-value>
        </custom-meta>
      </custom-meta-group>
    </article-meta>
  </front>
  <body>
    <sec>
      <p>
        The Na<sub>2</sub>C<sub>2</sub>O<sub>4</sub> result is shown in
        <xref ref-type="fig" rid="F1">Figure 1</xref>.
        <fig id="F1">
          <label>Figure 1</label>
          <caption>
            <p>Na<sub>2</sub>C<sub>2</sub>O<sub>4</sub> morphology.</p>
            <p>Second caption paragraph.</p>
          </caption>
          <graphic xlink:href="fig1.jpg"/>
        </fig>
      </p>
      <p>A second description references <xref ref-type="fig" rid="F1">Fig. 1</xref>.</p>
    </sec>
  </body>
</article>
"""


def test_parse_current_pmc_jats():
    records = parse_article_xml(XML, source_xml="example.xml")
    assert len(records) == 1

    record = records[0]
    assert record["pmcid"] == "PMC12345"
    assert record["article_version"] == "PMC12345.2"
    assert record["pmid"] == "99999"
    assert record["figure_id"] == "F1"
    assert record["figure_label"] == "Figure 1"
    assert record["license"] == "CC BY"
    assert record["image_hrefs"] == ["fig1.jpg"]

    # Inline tags must not introduce spaces into chemical formulae.
    assert "Na2C2O4 morphology." in record["caption"]

    # The canonical downstream field is cleaned, while the nullable screening
    # override preserves recursive paragraph text expected by the released model.
    assert len(record["reference_texts"]) == 2
    assert "Na2C2O4 result" in record["reference_texts"][0]
    assert "Na2C2O4 morphology" not in record["reference_texts"][0]
    assert "Second caption paragraph" not in record["reference_texts"][0]

    screening_refs = record["reference_texts_screening"]
    assert screening_refs is not None
    assert len(screening_refs) == 2
    assert "Na2C2O4 morphology" in screening_refs[0]
    assert "Second caption paragraph" in screening_refs[0]


def test_legacy_pmc_id_type_is_supported():
    legacy_xml = XML.replace(
        b'pub-id-type="pmcid"',
        b'pub-id-type="pmc"',
        1,
    )
    records = parse_article_xml(legacy_xml)
    assert records[0]["pmcid"] == "PMC12345"


def test_model_input_format():
    text = format_model_input("Caption text", ["Reference one", "Reference two"])
    assert text == (
        '"Caption": Caption text\n'
        '"Reference Text": Reference one\nReference two'
    )


def test_screening_override_is_null_when_representations_match():
    xml = b"""<?xml version="1.0"?>
    <article xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta><article-id pub-id-type="pmcid">PMC9</article-id></article-meta></front>
      <body><sec>
        <p>See <xref ref-type="fig" rid="F1">Figure 1</xref>.</p>
        <fig id="F1"><caption><p>Caption.</p></caption><graphic xlink:href="f1.jpg"/></fig>
      </sec></body>
    </article>
    """
    records = parse_article_xml(xml)
    assert records[0]["reference_texts"] == ["See Figure 1."]
    assert records[0]["reference_texts_screening"] is None
