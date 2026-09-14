"""Integration tests for PgVectorStore — the documented production store.

Until the 2026-09-14 audit this class had ZERO test coverage. It worked when
finally executed, but nothing in CI would have noticed it breaking, and it had
silently diverged from InMemoryStore (no similarity floor).

These are skipped unless a Postgres with pgvector is reachable:

    docker compose up -d
    pytest tests/test_pgvector_integration.py
"""
from __future__ import annotations

import os
import uuid

import pytest

from config.sources import SourceTier
from ingestion.pipeline import IngestionPipeline
from retrieval.embeddings import HashEmbedder
from retrieval.hybrid import HybridRetriever
from retrieval.store import InMemoryStore, PgVectorStore

DSN = os.environ.get("RA_TEST_DATABASE_URL", "postgresql://research:research@localhost:5432/research")
DIMS = 384

pytestmark = pytest.mark.integration


def _postgres_available() -> bool:
    try:
        import psycopg
    except ImportError:
        return False
    try:
        with psycopg.connect(DSN, connect_timeout=3):
            return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(
    not _postgres_available(),
    reason="Postgres with pgvector not reachable — run `docker compose up -d`",
)


@pytest.fixture
def pg_store():
    """A store scoped to this test run, cleaned up afterwards."""
    import psycopg

    store = PgVectorStore(DSN, dimensions=DIMS)
    store.initialise()
    marker = uuid.uuid4().hex[:8]
    yield store, marker
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("DELETE FROM chunks WHERE url LIKE %s", (f"%{marker}%",))


@pytest.fixture
def pg_embedder():
    return HashEmbedder(DIMS)


def _ingest(store, embedder, marker: str) -> IngestionPipeline:
    pipeline = IngestionPipeline(store=store, embedder=embedder)
    pipeline.ingest_text(
        "# EKS\n\nAmazon EKS runs and manages the Kubernetes control plane across zones.",
        url=f"https://docs.aws.amazon.com/{marker}/eks.html", suffix=".md",
    )
    pipeline.ingest_text(
        "# Errors\n\nError code E4021 indicates a GPU allocation request could not be satisfied.",
        url=f"https://docs.aws.amazon.com/{marker}/err.html", suffix=".md",
    )
    pipeline.ingest_text(
        "# Blog\n\nSome unverified opinion about clusters and control planes.",
        url=f"https://farm-{marker}.example/p", suffix=".md",
    )
    return pipeline


@requires_pg
def test_schema_initialises_idempotently(pg_store):
    store, _ = pg_store
    store.initialise()
    store.initialise()  # must not raise on repeat


@requires_pg
def test_ingest_and_dense_search(pg_store, pg_embedder):
    store, marker = pg_store
    _ingest(store, pg_embedder, marker)
    results = store.dense_search(pg_embedder.embed_query("who manages the control plane"), 5)
    assert results
    assert all(score > 0 for _, score in results)


@requires_pg
def test_dense_search_has_a_similarity_floor_matching_inmemory(pg_store, pg_embedder):
    """Regression: PgVectorStore returned zero-similarity rows; InMemoryStore did not.

    ORDER BY ... LIMIT n always returns n rows, so without an explicit floor the
    two backends disagreed about what counts as a hit — meaning the tested
    behaviour was not the deployed behaviour.
    """
    store, marker = pg_store
    _ingest(store, pg_embedder, marker)

    memory = InMemoryStore()
    _ingest(memory, pg_embedder, marker)

    query = pg_embedder.embed_query("completely unrelated pasta cooking instructions")
    pg_scores = [round(s, 6) for _, s in store.dense_search(query, 10)]
    mem_scores = [round(s, 6) for _, s in memory.dense_search(query, 10)]

    assert all(s > 0 for s in pg_scores), f"pgvector returned zero-similarity rows: {pg_scores}"
    assert len(pg_scores) == len(mem_scores), (
        f"store divergence: pgvector returned {len(pg_scores)} rows, "
        f"InMemoryStore returned {len(mem_scores)}"
    )


@requires_pg
def test_lexical_search_finds_exact_identifiers(pg_store, pg_embedder):
    store, marker = pg_store
    _ingest(store, pg_embedder, marker)
    results = store.lexical_search("E4021", 5)
    assert results
    assert any("E4021" in c.text for c, _ in results)


@requires_pg
def test_tier_filter_applies_to_both_arms(pg_store, pg_embedder):
    store, marker = pg_store
    _ingest(store, pg_embedder, marker)

    dense = store.dense_search(
        pg_embedder.embed_query("control plane"), 10, max_tier=SourceTier.OFFICIAL
    )
    lexical = store.lexical_search("control plane", 10, max_tier=SourceTier.OFFICIAL)
    for chunk, _ in dense + lexical:
        assert chunk.tier <= SourceTier.OFFICIAL, f"tier filter leaked {chunk.metadata.domain}"


@requires_pg
def test_full_metadata_survives_the_database_round_trip(pg_store, pg_embedder):
    store, marker = pg_store
    _ingest(store, pg_embedder, marker)
    chunk, _ = store.dense_search(pg_embedder.embed_query("control plane"), 1)[0]
    fetched = store.get_chunk(chunk.chunk_id)

    assert fetched is not None
    md = fetched.metadata
    assert md.title and md.url and md.domain
    assert md.tier is SourceTier.OFFICIAL
    assert md.source_type
    assert md.retrieved_at is not None
    assert fetched.document_id and fetched.chunk_id
    assert fetched.char_start >= 0 and fetched.char_end >= fetched.char_start
    assert fetched.text == chunk.text


@requires_pg
def test_reingestion_is_idempotent(pg_store, pg_embedder):
    store, marker = pg_store
    _ingest(store, pg_embedder, marker)
    before = store.count()
    _ingest(store, pg_embedder, marker)
    assert store.count() == before, "ON CONFLICT did not prevent duplicate chunks"


@requires_pg
def test_hybrid_retrieval_over_the_real_database(pg_store, pg_embedder):
    store, marker = pg_store
    _ingest(store, pg_embedder, marker)
    results = HybridRetriever(store, pg_embedder).retrieve(
        "who manages the kubernetes control plane", limit=5
    )
    assert results
    assert any(r.retrieval_method == "hybrid" for r in results)
    assert all(r.chunk.metadata.url for r in results)


@requires_pg
def test_add_chunks_without_vectors_is_rejected(pg_store):
    """pgvector cannot store a chunk with no embedding; fail loudly, not silently."""
    store, _ = pg_store
    from schemas import Chunk, DocumentMetadata, SourceType

    md = DocumentMetadata(
        document_id="d", title="T", url="https://kubernetes.io/x", domain="kubernetes.io",
        tier=SourceTier.OFFICIAL, source_type=SourceType.OFFICIAL_DOCS,
    )
    chunk = Chunk(chunk_id="c", document_id="d", text="t", ordinal=0,
                  char_start=0, char_end=1, metadata=md)
    with pytest.raises(ValueError):
        store.add_chunks([chunk], None)
