"""Source conflict detection.

Closes audit finding #5: `SourceConflict` existed as a schema and a prompt rule,
the UI rendered a panel for it, the eval set contained two deliberately
conflicting Tier-1 sources — and no code path ever populated it.

The property under test throughout: **the model reports what each source says;
code decides which source is more authoritative.**
"""
from __future__ import annotations

import pytest

from citations.conflicts import ConflictDetector, conflict_notes
from config.sources import SourceTier
from generation.client import ClaudeClient, GenerationError
from ingestion.pipeline import IngestionPipeline
from retrieval.hybrid import HybridRetriever
from retrieval.store import InMemoryStore


class _ScriptedConflictClient(ClaudeClient):
    def __init__(self, settings, payload=None, raises=None):
        super().__init__(settings, client=object())
        self.payload = payload if payload is not None else []
        self.raises = raises
        self.calls = 0
        self.last_sources: str = ""

    def detect_conflicts(self, question, chunks):
        self.calls += 1
        self.last_sources = "\n".join(c.chunk.metadata.domain for c in chunks)
        if self.raises:
            raise self.raises
        return list(self.payload)


def _scored(url: str, domain: str, text: str, score: float = 0.8):
    """Build a ScoredChunk directly.

    The detector's unit tests must not depend on retrieval quality - the test
    fixtures use HashEmbedder, which is lexical-only, so a semantically relevant
    dissenting source can score below the relevance floor for reasons that have
    nothing to do with conflict detection.
    """
    from config.sources import get_trust_policy
    from schemas import Chunk, DocumentMetadata, ScoredChunk, SourceType

    tier = get_trust_policy().tier_for_url(url)
    md = DocumentMetadata(
        document_id=domain, title=domain, url=url, domain=domain,
        tier=tier, source_type=SourceType.OFFICIAL_DOCS,
    )
    chunk = Chunk(chunk_id=f"{domain}::0", document_id=domain, text=text, ordinal=0,
                  char_start=0, char_end=len(text), metadata=md)
    return ScoredChunk(chunk=chunk, score=score, dense_score=score, rerank_score=score)


@pytest.fixture
def conflicting_chunks():
    """Two Tier-1 sources giving opposite scaling guidance."""
    return [
        _scored("https://kubernetes.io/docs/scaling-order.html", "kubernetes.io",
                "Configure Pod autoscaling before node autoscaling.", 0.9),
        _scored("https://learn.microsoft.com/azure/aks/scaling-order.html", "learn.microsoft.com",
                "Provision node capacity ahead of demand rather than scaling Pods first.", 0.8),
    ]


def test_conflict_between_two_sources_is_surfaced(conflicting_chunks, settings):
    chunks = conflicting_chunks
    client = _ScriptedConflictClient(settings, payload=[{
        "topic": "scaling order",
        "source_a_index": 0, "position_a": "Pods first",
        "source_b_index": 1, "position_b": "Nodes first for GPU workloads",
    }])
    conflicts = ConflictDetector(client, settings=settings).detect("scaling order?", chunks)
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.topic == "scaling order"
    assert c.source_a_url != c.source_b_url
    assert c.position_a and c.position_b


def test_authority_is_decided_in_code_from_tier_not_by_the_model(embedder, settings):
    """A Tier 1 source must outrank a Tier 3 one regardless of what the model says."""
    store = InMemoryStore()
    p = IngestionPipeline(store=store, embedder=embedder, settings=settings)
    p.ingest_text("# Official\n\nThe control plane is managed by the provider.",
                  url="https://kubernetes.io/docs/a.html", suffix=".md")
    p.ingest_text("# Wiki\n\nThe control plane is managed by the customer.",
                  url="https://en.wikipedia.org/wiki/b", suffix=".md")
    chunks = HybridRetriever(store, embedder, settings=settings).retrieve(
        "who manages the control plane", limit=6)
    assert len(chunks) >= 2

    client = _ScriptedConflictClient(settings, payload=[{
        "topic": "control plane ownership",
        "source_a_index": 0, "position_a": "provider-managed",
        "source_b_index": 1, "position_b": "customer-managed",
    }])
    conflicts = ConflictDetector(client, settings=settings).detect("who manages it?", chunks)
    assert conflicts
    c = conflicts[0]
    assert c.more_authoritative_url is not None
    winner_tier = c.source_a_tier if c.more_authoritative_url == c.source_a_url else c.source_b_tier
    loser_tier = c.source_b_tier if c.more_authoritative_url == c.source_a_url else c.source_a_tier
    assert winner_tier < loser_tier, "the lower-tier source was named more authoritative"


def test_equal_tier_conflict_names_no_winner(conflicting_chunks, settings):
    """Both Tier 1: the system must not invent a hierarchy."""
    chunks = conflicting_chunks
    client = _ScriptedConflictClient(settings, payload=[{
        "topic": "scaling order",
        "source_a_index": 0, "position_a": "Pods first",
        "source_b_index": 1, "position_b": "Nodes first",
    }])
    c = ConflictDetector(client, settings=settings).detect("q", chunks)[0]
    assert c.source_a_tier is c.source_b_tier is SourceTier.OFFICIAL
    assert c.more_authoritative_url is None
    assert "neither is automatically" in c.rationale


def test_detection_is_skipped_when_only_one_source_exists(embedder, settings):
    """No second opinion means no possible conflict — and no wasted call."""
    store = InMemoryStore()
    IngestionPipeline(store=store, embedder=embedder, settings=settings).ingest_text(
        "# Only\n\nA single source discussing the control plane in detail.",
        url="https://kubernetes.io/docs/only.html", suffix=".md",
    )
    chunks = HybridRetriever(store, embedder, settings=settings).retrieve("control plane", limit=6)
    client = _ScriptedConflictClient(settings, payload=[{"topic": "x"}])
    assert ConflictDetector(client, settings=settings).detect("q", chunks) == []
    assert client.calls == 0, "a model call was made with nothing to compare"


def test_multiple_chunks_from_one_domain_are_not_a_conflict(embedder, settings):
    """Two passages from the same page elaborating on each other is not disagreement."""
    store = InMemoryStore()
    IngestionPipeline(store=store, embedder=embedder, settings=settings).ingest_text(
        "# Doc\n\nFirst paragraph about the control plane and its management.\n\n"
        "Second paragraph continuing about the control plane in more depth.",
        url="https://kubernetes.io/docs/one.html", suffix=".md",
    )
    chunks = HybridRetriever(store, embedder, settings=settings).retrieve("control plane", limit=6)
    client = _ScriptedConflictClient(settings)
    ConflictDetector(client, settings=settings).detect("q", chunks)
    if client.calls:
        assert len(set(client.last_sources.split("\n"))) >= 2


def test_out_of_range_source_index_is_discarded(conflicting_chunks, settings):
    chunks = conflicting_chunks
    client = _ScriptedConflictClient(settings, payload=[{
        "topic": "x", "source_a_index": 0, "position_a": "a",
        "source_b_index": 99, "position_b": "b",
    }])
    assert ConflictDetector(client, settings=settings).detect("q", chunks) == []


def test_self_conflict_is_discarded(conflicting_chunks, settings):
    """A source cannot contradict itself; such a report is malformed."""
    chunks = conflicting_chunks
    client = _ScriptedConflictClient(settings, payload=[{
        "topic": "x", "source_a_index": 0, "position_a": "a",
        "source_b_index": 0, "position_b": "b",
    }])
    assert ConflictDetector(client, settings=settings).detect("q", chunks) == []


def test_incomplete_conflict_report_is_discarded(conflicting_chunks, settings):
    chunks = conflicting_chunks
    client = _ScriptedConflictClient(settings, payload=[
        {"topic": "", "source_a_index": 0, "position_a": "a", "source_b_index": 1, "position_b": "b"},
        {"topic": "t", "source_a_index": 0, "position_a": "", "source_b_index": 1, "position_b": "b"},
    ])
    assert ConflictDetector(client, settings=settings).detect("q", chunks) == []


def test_detection_failure_never_blocks_an_answer(conflicting_chunks, settings):
    """Conflicts are additive information; losing them must not lose the answer."""
    chunks = conflicting_chunks
    for exc in (GenerationError("upstream down"), TypeError("unexpected")):
        client = _ScriptedConflictClient(settings, raises=exc)
        assert ConflictDetector(client, settings=settings).detect("q", chunks) == []


def test_conflict_notes_are_readable_and_state_which_source_wins(conflicting_chunks, settings):
    chunks = conflicting_chunks
    client = _ScriptedConflictClient(settings, payload=[{
        "topic": "scaling order", "source_a_index": 0, "position_a": "Pods first",
        "source_b_index": 1, "position_b": "Nodes first",
    }])
    notes = conflict_notes(ConflictDetector(client, settings=settings).detect("q", chunks))
    assert notes and "disagree" in notes[0].lower()
    assert "contested" in notes[0].lower() or "follows" in notes[0].lower()


def test_conflicts_reach_the_final_answer(embedder, settings, fake_client_factory):
    """End to end: a detected conflict must appear on ResearchAnswer.conflicts."""
    from assistant import ResearchAssistant
    from citations.validator import CitationValidator
    from citations.verifier import ClaimVerifier
    from schemas import AnswerStatus

    store = InMemoryStore()
    p = IngestionPipeline(store=store, embedder=embedder, settings=settings)
    p.ingest_text("# Scaling\n\nConfigure Pod autoscaling before node autoscaling always.",
                  url="https://kubernetes.io/docs/s.html", suffix=".md")
    p.ingest_text("# Scaling\n\nConfigure Pod autoscaling before node autoscaling, except for GPU.",
                  url="https://learn.microsoft.com/azure/s.html", suffix=".md")
    retriever = HybridRetriever(store, embedder, settings=settings)
    chunks = retriever.retrieve("Configure Pod autoscaling before node autoscaling", limit=6)
    cited = chunks[0].chunk.text[:60]

    client = fake_client_factory(answer_text=cited, citation_specs=[(0, cited)])
    detector_client = _ScriptedConflictClient(settings, payload=[{
        "topic": "scaling order", "source_a_index": 0, "position_a": "Pods first",
        "source_b_index": 1, "position_b": "Nodes first",
    }])
    assistant = ResearchAssistant(
        retriever, client, settings=settings,
        validator=CitationValidator(), verifier=ClaimVerifier(client, settings=settings),
        conflict_detector=ConflictDetector(detector_client, settings=settings),
    )
    result = assistant.answer("should node autoscaling come before pod autoscaling?")
    if result.status is AnswerStatus.ANSWERED:
        assert result.conflicts, "a detected conflict did not reach the answer"
        assert any("disagree" in n.lower() for n in result.uncertainty_notes)
