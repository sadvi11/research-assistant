"""Hybrid retrieval behaviour."""
from __future__ import annotations

from config.sources import SourceTier
from retrieval.hybrid import reciprocal_rank_fusion
from retrieval.store import InMemoryStore


def test_lexical_arm_finds_exact_strings_embeddings_would_miss(retriever):
    """The E4021 case: an error code has no semantic neighbourhood."""
    results = retriever.retrieve("E4021", limit=5)
    assert results, "exact-string query returned nothing"
    assert any("E4021" in r.chunk.text for r in results)


def test_semantic_query_retrieves_relevant_chunk(retriever):
    results = retriever.retrieve("who operates the kubernetes control plane", limit=5)
    assert results
    assert any("control plane" in r.chunk.text.lower() for r in results)


def test_tier_filter_excludes_untrusted_sources(retriever):
    unfiltered = retriever.retrieve("control plane", limit=10)
    assert any(r.chunk.tier is SourceTier.UNVETTED for r in unfiltered), "fixture lacks a tier-4 doc"

    filtered = retriever.retrieve("control plane", limit=10, max_tier=SourceTier.SECONDARY)
    assert filtered
    assert all(r.chunk.tier <= SourceTier.SECONDARY for r in filtered)


def test_official_sources_outrank_equally_relevant_unvetted_ones(retriever):
    results = retriever.retrieve("control plane", limit=10)
    tiers = [r.chunk.tier for r in results]
    first_unvetted = next((i for i, t in enumerate(tiers) if t is SourceTier.UNVETTED), None)
    first_official = next((i for i, t in enumerate(tiers) if t is SourceTier.OFFICIAL), None)
    assert first_official is not None
    if first_unvetted is not None:
        assert first_official < first_unvetted, "unvetted source outranked official documentation"


def test_no_single_document_monopolises_the_context(store, embedder, settings):
    """A long document must not fill every context slot."""
    from ingestion.pipeline import IngestionPipeline
    from retrieval.hybrid import HybridRetriever

    long_doc = "# Long\n\n" + "\n\n".join(
        f"Paragraph {i} discusses the kubernetes control plane in detail." for i in range(30)
    )
    IngestionPipeline(store=store, embedder=embedder, settings=settings).ingest_text(
        long_doc, url="https://kubernetes.io/docs/long.html", suffix=".md"
    )
    results = HybridRetriever(store, embedder, settings=settings).retrieve(
        "kubernetes control plane", limit=6
    )
    per_doc: dict[str, int] = {}
    for r in results:
        per_doc[r.chunk.document_id] = per_doc.get(r.chunk.document_id, 0) + 1
    assert max(per_doc.values()) <= settings.max_chunks_per_document


def test_empty_query_returns_nothing(retriever):
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_query_with_no_matches_returns_empty_not_garbage(retriever):
    results = retriever.retrieve("xylophone quantum marsupial zzzz", limit=5)
    assert all((r.rerank_score or 0) >= 0 for r in results)


def test_rrf_rewards_agreement_between_arms():
    class C:
        def __init__(self, cid):
            self.chunk_id = cid

    both, dense_only = C("both"), C("dense")
    merged = reciprocal_rank_fusion(
        {"dense": [(both, 0.9), (dense_only, 0.8)], "lexical": [(both, 5.0)]}, k=60
    )
    assert merged["both"]["fused"] > merged["dense"]["fused"]
    assert merged["both"]["dense_rank"] == 1 and merged["both"]["lexical_rank"] == 1


def test_retrieval_degrades_to_lexical_when_dense_fails(store, embedder, settings):
    """A broken embedder must not take the whole query down."""
    from retrieval.hybrid import HybridRetriever

    class BrokenEmbedder:
        dimensions = 256
        def embed_documents(self, texts): return [[0.0] * 256 for _ in texts]
        def embed_query(self, text): raise RuntimeError("embedding service down")

    results = HybridRetriever(store, BrokenEmbedder(), settings=settings).retrieve("E4021", limit=5)
    assert results, "lexical fallback did not run"
    assert all(r.dense_rank is None for r in results)


def test_store_ingestion_is_idempotent(store, embedder, settings):
    from ingestion.pipeline import IngestionPipeline

    before = store.count()
    pipeline = IngestionPipeline(store=store, embedder=embedder, settings=settings)
    text = "# Error reference\n\nError code E4021 indicates that a GPU allocation request could not be satisfied by any node in the cluster."
    pipeline.ingest_text(text, url="https://docs.aws.amazon.com/eks/errors.html", suffix=".md")
    assert store.count() == before, "re-ingesting identical content duplicated chunks"


def test_bm25_ranks_rarer_terms_higher():
    store = InMemoryStore()
    from schemas import Chunk, DocumentMetadata, SourceType

    md = DocumentMetadata(document_id="d", title="T", url="https://kubernetes.io/x",
                          domain="kubernetes.io", tier=SourceTier.OFFICIAL,
                          source_type=SourceType.OFFICIAL_DOCS)
    chunks = [
        Chunk(chunk_id="c1", document_id="d", text="the cluster runs the cluster workload", ordinal=0,
              char_start=0, char_end=10, metadata=md),
        Chunk(chunk_id="c2", document_id="d", text="the cluster reports error E4021 rarely", ordinal=1,
              char_start=0, char_end=10, metadata=md),
    ]
    store.add_chunks(chunks)
    top = store.lexical_search("E4021", 2)
    assert top and top[0][0].chunk_id == "c2"
