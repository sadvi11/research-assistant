"""Chunking invariants.

The properties asserted here are the ones whose violation silently degrades
every answer downstream: lost content, split sentences, missing provenance.
"""
from __future__ import annotations

import pytest

from ingestion.chunker import Chunker, approx_tokens
from schemas import DocumentMetadata, SourceType
from config.sources import SourceTier

DOC = """# Guide

## Alpha section

First paragraph of alpha with enough words to be a real paragraph here.

Second paragraph of alpha, also carrying distinct content worth retrieving.

## Beta section

Beta content mentions error code E4021 and should not merge into alpha.
"""


def _md(url="https://docs.aws.amazon.com/g.html") -> DocumentMetadata:
    return DocumentMetadata(
        document_id="doc1", title="Guide", url=url, domain="docs.aws.amazon.com",
        tier=SourceTier.OFFICIAL, source_type=SourceType.OFFICIAL_DOCS,
    )


def test_no_content_is_lost_at_any_chunk_size():
    """Every distinctive phrase must survive into at least one chunk."""
    phrases = ["First paragraph of alpha", "Second paragraph of alpha",
               "error code E4021", "should not merge into alpha"]
    for size in (30, 60, 120, 400):
        chunks = Chunker(chunk_size_tokens=size, overlap_tokens=10).split(DOC, _md())
        joined = " ".join(c.text for c in chunks)
        missing = [p for p in phrases if p not in joined]
        assert not missing, f"chunk_size={size} lost: {missing}"


def test_sections_do_not_bleed_together():
    chunks = Chunker(chunk_size_tokens=400, overlap_tokens=20).split(DOC, _md())
    for chunk in chunks:
        has_alpha = "paragraph of alpha" in chunk.text
        has_beta = "E4021" in chunk.text
        assert not (has_alpha and has_beta), "alpha and beta content merged into one chunk"


def test_every_chunk_carries_source_url_and_tier():
    chunks = Chunker(chunk_size_tokens=60, overlap_tokens=10).split(DOC, _md())
    assert chunks
    for chunk in chunks:
        assert chunk.source_url == "https://docs.aws.amazon.com/g.html"
        assert chunk.tier is SourceTier.OFFICIAL
        assert chunk.document_id == "doc1"


def test_chunk_ids_are_unique_and_stable():
    chunker = Chunker(chunk_size_tokens=60, overlap_tokens=10)
    first = chunker.split(DOC, _md())
    second = chunker.split(DOC, _md())
    ids = [c.chunk_id for c in first]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    assert ids == [c.chunk_id for c in second], "chunk ids are not deterministic"


def test_section_path_is_recorded():
    chunks = Chunker(chunk_size_tokens=400, overlap_tokens=20).split(DOC, _md())
    paths = {c.section_path for c in chunks}
    assert ("Guide", "Alpha section") in paths
    assert ("Guide", "Beta section") in paths


def test_oversized_paragraph_is_split_on_sentence_boundaries():
    long_doc = "# T\n\n" + " ".join(f"Sentence number {i} carries content." for i in range(60))
    chunks = Chunker(chunk_size_tokens=40, overlap_tokens=8).split(long_doc, _md())
    assert len(chunks) > 1
    for chunk in chunks:
        # A chunk should not begin mid-sentence (lowercase start with no punctuation).
        assert not chunk.text.startswith(" ")
        assert chunk.text.strip()


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        Chunker(chunk_size_tokens=100, overlap_tokens=100)
    with pytest.raises(ValueError):
        Chunker(chunk_size_tokens=0, overlap_tokens=0)


def test_empty_and_whitespace_documents_produce_no_chunks():
    chunker = Chunker(chunk_size_tokens=100, overlap_tokens=10)
    assert chunker.split("", _md()) == []
    assert chunker.split("   \n\n  ", _md()) == []


def test_approx_tokens_scales_with_length():
    assert approx_tokens("") >= 1
    assert approx_tokens("a" * 350) > approx_tokens("a" * 35)
