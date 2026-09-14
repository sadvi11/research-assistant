"""Regression tests for defects found in the 2026-09-14 audit.

Each test here failed before its fix. They exist because the original suite
passed while all of these were broken - which is the only real evidence that a
test suite is worth having.

See RAG-AUDIT-REPORT.md for the findings these correspond to.
"""
from __future__ import annotations

import pytest

from citations.validator import CitationValidator
from citations.verifier import ClaimVerifier
from config.sources import SourceTier
from generation.client import ClaudeClient, GenerationResult, RawCitation
from ingestion.pipeline import IngestionPipeline
from retrieval.hybrid import HybridRetriever
from retrieval.store import InMemoryStore
from schemas import AnswerStatus


# ── CRITICAL-1 ──────────────────────────────────────────────────────────────

class _CitesLastChunk(ClaudeClient):
    """A plausibly-behaved model: it cites whichever chunk it was handed."""

    def __init__(self, settings):
        super().__init__(settings, client=object())

    def generate_grounded_answer(self, q, chunks, *, subquestions=None, web_tools=None):
        i = len(chunks) - 1
        text = chunks[i].chunk.text[:60]
        return GenerationResult(text=text, citations=[RawCitation(i, None, text)])

    def extract_claims(self, answer):
        return [answer]

    def verify_claim(self, claim, evidence):
        return ("supported", "")


def _store_with_mixed_trust(embedder, settings) -> InMemoryStore:
    store = InMemoryStore()
    pipeline = IngestionPipeline(store=store, embedder=embedder, settings=settings)
    pipeline.ingest_text(
        "# EKS\n\nAmazon EKS runs and manages the Kubernetes control plane across zones.",
        url="https://docs.aws.amazon.com/eks.html", suffix=".md",
    )
    pipeline.ingest_text(
        "# Hot takes\n\nThe control plane is basically magic and nobody should audit it.",
        url="https://content-farm.example/p", suffix=".md",
    )
    return store


def test_answer_citing_only_unvetted_sources_is_withheld(embedder, settings, assistant_factory):
    """CRITICAL-1: a Tier-4 content farm was able to be the sole citation.

    The citation verified - the text really was in that chunk - but citation
    authenticity and source trust are different properties.
    """
    from assistant import ResearchAssistant

    store = _store_with_mixed_trust(embedder, settings)
    client = _CitesLastChunk(settings)
    assistant = ResearchAssistant(
        HybridRetriever(store, embedder, settings=settings), client, settings=settings,
        validator=CitationValidator(), verifier=ClaimVerifier(client, settings=settings),
    )
    result = assistant.answer("who manages the kubernetes control plane")

    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE, (
        "an answer citing only an unvetted source was released"
    )
    assert any("unvetted" in n.lower() for n in result.uncertainty_notes)


def test_caveats_describe_cited_sources_not_merely_retrieved_ones(
    embedder, settings, fake_client_factory, assistant_factory
):
    """CRITICAL-1 root cause: caveats were computed from retrieved chunks.

    A Tier 1 document sitting unused in the retrieved set suppressed the
    "no official source" warning for an answer that cited neither.
    """
    from assistant import ResearchAssistant

    store = _store_with_mixed_trust(embedder, settings)
    retriever = HybridRetriever(store, embedder, settings=settings)
    chunks = retriever.retrieve("control plane", limit=5)
    assert any(c.chunk.tier is SourceTier.OFFICIAL for c in chunks), "fixture needs a tier-1 chunk"

    client = fake_client_factory(
        answer_text="The control plane is basically magic and nobody should audit it.",
        citation_specs=[],
    )
    assistant = ResearchAssistant(
        retriever, client, settings=settings,
        validator=CitationValidator(), verifier=ClaimVerifier(client, settings=settings),
    )
    result = assistant.answer("who manages the kubernetes control plane")
    # With no verifiable citation the answer must not be presented as grounded.
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE


# ── CRITICAL-2 ──────────────────────────────────────────────────────────────

def test_absolutely_irrelevant_chunks_do_not_reach_the_model(embedder, settings):
    """CRITICAL-2: min_relevance_score compared NORMALISED scores.

    The top hit for any query normalises to ~1.0, so a chunk with absolute
    cosine 0.144 scored 0.656 and passed a 0.25 threshold.
    """
    store = InMemoryStore()
    pipeline = IngestionPipeline(store=store, embedder=embedder, settings=settings)
    pipeline.ingest_text(
        "# EKS\n\nAmazon EKS runs and manages the Kubernetes control plane across zones.",
        url="https://docs.aws.amazon.com/eks.html", suffix=".md",
    )
    pipeline.ingest_text(
        "# Pasta\n\nBoil pasta in salted water for eight minutes then drain thoroughly.",
        url="https://docs.aws.amazon.com/pasta.html", suffix=".md",
    )
    results = HybridRetriever(store, embedder, settings=settings).retrieve(
        "who manages the kubernetes control plane", limit=5
    )
    for item in results:
        assert (
            item.absolute_relevance >= settings.min_absolute_relevance
            or item.lexical_rank is not None
        ), (
            f"chunk with absolute relevance {item.absolute_relevance:.3f} reached the "
            f"context (floor is {settings.min_absolute_relevance})"
        )


def test_exact_term_hits_survive_the_absolute_floor(retriever):
    """Guard on the CRITICAL-2 fix: an error-code match has low cosine but is
    real evidence. The floor must not delete lexical-only hits."""
    results = retriever.retrieve("E4021", limit=5)
    assert results, "the absolute-relevance floor removed a valid exact-term match"
    assert any("E4021" in r.chunk.text for r in results)


# ── Diversity cap (found while fixing CRITICAL-2) ───────────────────────────

def test_diversity_cap_is_hard_not_a_preference(embedder, settings):
    """`_diversify` backfilled from over-cap documents to fill the budget, so a
    single long document could exceed the cap. The old test passed by accident
    because other documents happened to fill the slots."""
    store = InMemoryStore()
    long_doc = "# Long\n\n" + "\n\n".join(
        f"Paragraph {i} discusses the kubernetes control plane in detail." for i in range(30)
    )
    IngestionPipeline(store=store, embedder=embedder, settings=settings).ingest_text(
        long_doc, url="https://kubernetes.io/docs/long.html", suffix=".md"
    )
    results = HybridRetriever(store, embedder, settings=settings).retrieve(
        "kubernetes control plane", limit=8
    )
    assert len(results) <= settings.max_chunks_per_document


# ── MEDIUM-4 ────────────────────────────────────────────────────────────────

def test_unexpected_exception_becomes_an_error_status_not_a_crash(
    retriever, settings, assistant_factory, fake_client_factory
):
    """The orchestrator is a boundary; nothing may propagate to the caller."""
    client = fake_client_factory()

    def boom(*args, **kwargs):
        raise TypeError("unexpected internal state")

    client.generate_grounded_answer = boom
    result = assistant_factory(client).answer("who manages the kubernetes control plane")
    assert result.status is AnswerStatus.ERROR
    assert result.request_id
