"""Citation validation - the core anti-hallucination defence.

Every test here scripts the model into misbehaving and asserts the system
catches it in code. None of these rely on the model being well-behaved.
"""
from __future__ import annotations

from citations.validator import CitationValidator, normalise
from generation.client import RawCitation


def _citations(retriever, query="control plane", limit=4):
    return retriever.retrieve(query, limit=limit)


def test_genuine_citation_is_verified(retriever):
    chunks = _citations(retriever)
    real_text = chunks[0].chunk.text[:60]
    report = CitationValidator().validate(
        [RawCitation(0, None, real_text)], chunks
    )
    assert report.all_verified
    assert report.valid[0].verified is True
    assert report.valid[0].url == chunks[0].chunk.metadata.url


def test_fabricated_quote_is_rejected(retriever):
    """The model 'quotes' text that appears in no source."""
    chunks = _citations(retriever)
    report = CitationValidator().validate(
        [RawCitation(0, None, "Amazon guarantees one hundred percent uptime forever")], chunks
    )
    assert not report.valid
    assert "does not appear" in report.rejected[0][1]


def test_out_of_range_document_index_is_rejected(retriever):
    chunks = _citations(retriever)
    report = CitationValidator().validate(
        [RawCitation(999, None, chunks[0].chunk.text[:50])], chunks
    )
    assert not report.valid
    assert "out of range" in report.rejected[0][1]


def test_negative_document_index_is_rejected(retriever):
    chunks = _citations(retriever)
    report = CitationValidator().validate([RawCitation(-1, None, "some text here at all")], chunks)
    assert not report.valid


def test_citation_pointing_at_the_wrong_document_is_rejected(retriever):
    """Text real in chunk A, but attributed to chunk B, must not survive."""
    chunks = _citations(retriever, limit=6)
    assert len(chunks) >= 2
    other = next(
        (i for i, c in enumerate(chunks[1:], start=1)
         if chunks[0].chunk.text[:40] not in c.chunk.text), None
    )
    assert other is not None, "fixture needs two dissimilar chunks"
    report = CitationValidator().validate(
        [RawCitation(other, None, chunks[0].chunk.text[:40])], chunks
    )
    assert not report.valid, "citation was bound to the wrong source document"


def test_trivially_short_citation_is_rejected(retriever):
    chunks = _citations(retriever)
    report = CitationValidator().validate([RawCitation(0, None, "the")], chunks)
    assert not report.valid
    assert "too short" in report.rejected[0][1]


def test_empty_cited_text_is_rejected(retriever):
    chunks = _citations(retriever)
    report = CitationValidator().validate([RawCitation(0, None, "   ")], chunks)
    assert not report.valid


def test_whitespace_and_smart_quote_differences_do_not_break_verification(retriever):
    """Formatting normalisation is not a hallucination."""
    chunks = _citations(retriever)
    original = chunks[0].chunk.text[:60]
    reformatted = original.replace(" ", "  ").replace("'", "’")
    report = CitationValidator().validate([RawCitation(0, None, reformatted)], chunks)
    assert report.valid, "legitimate citation rejected over whitespace/quote formatting"


def test_duplicate_citations_are_collapsed(retriever):
    chunks = _citations(retriever)
    text = chunks[0].chunk.text[:50]
    report = CitationValidator().validate(
        [RawCitation(0, None, text), RawCitation(0, None, text)], chunks
    )
    assert len(report.valid) == 1


def test_mixed_valid_and_invalid_keeps_only_the_valid(retriever):
    chunks = _citations(retriever)
    report = CitationValidator().validate(
        [
            RawCitation(0, None, chunks[0].chunk.text[:55]),
            RawCitation(0, None, "entirely invented supporting sentence here"),
            RawCitation(500, None, chunks[0].chunk.text[:55]),
        ],
        chunks,
    )
    assert len(report.valid) == 1
    assert len(report.rejected) == 2
    assert not report.all_verified


def test_citation_indices_are_renumbered_contiguously(retriever):
    chunks = _citations(retriever, limit=4)
    raws = [RawCitation(i, None, c.chunk.text[:45]) for i, c in enumerate(chunks[:3])]
    report = CitationValidator().validate(raws, chunks)
    assert [c.index for c in report.valid] == list(range(1, len(report.valid) + 1))


def test_no_citations_yields_empty_report(retriever):
    report = CitationValidator().validate([], _citations(retriever))
    assert not report.valid and report.all_verified and report.total == 0


def test_normalise_is_idempotent():
    text = "  Some text  with “quotes” and — dashes  "
    assert normalise(normalise(text)) == normalise(text)
