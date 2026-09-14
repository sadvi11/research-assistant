"""Turning retrieved chunks into citable document blocks.

This is the load-bearing decision of the whole system. Each chunk becomes its
own `document` block with `citations: {enabled: true}`, so:

  * the API returns citations with `cited_text` taken verbatim from the chunk
  * `document_index` maps 1:1 onto our chunk list, so a citation identifies
    exactly which stored chunk it came from
  * `char_location` gives offsets into that chunk, which is precisely what the
    evidence inspector needs to highlight the supporting passage

A model asked to write "[1]" in prose can fabricate the marker. A citation
produced by this mechanism is generated against the actual document content,
and we re-verify it against our own stored copy afterwards regardless.
"""
from __future__ import annotations

from schemas import ScoredChunk


def chunks_to_document_blocks(chunks: list[ScoredChunk]) -> list[dict]:
    """One document block per chunk, in stable order.

    Order matters: `document_index` in the returned citations is an index into
    this list, so the caller must not reorder chunks afterwards.
    """
    blocks: list[dict] = []
    for sc in chunks:
        md = sc.chunk.metadata
        context_bits = [f"{md.domain}", f"tier {int(md.tier)}"]
        if sc.chunk.section_path:
            context_bits.append(" > ".join(sc.chunk.section_path))
        if md.publication_date:
            context_bits.append(md.publication_date.date().isoformat())

        blocks.append(
            {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": "text/plain",
                    "data": sc.chunk.text,
                },
                "title": md.title[:200],
                # Context is metadata for the model; it is not citable text.
                "context": " | ".join(context_bits),
                "citations": {"enabled": True},
            }
        )
    return blocks
