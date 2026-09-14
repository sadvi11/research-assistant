"""Test fixtures.

Everything here runs offline with no credentials and no network. The fake
client models the *shape* of the Anthropic API - document blocks in, text plus
citation objects out - so the pipeline under test is the real one. What is
faked is the model, not the plumbing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import ResearchAssistant  # noqa: E402
from citations.validator import CitationValidator  # noqa: E402
from citations.verifier import ClaimVerifier  # noqa: E402
from config.settings import Settings  # noqa: E402
from generation.client import ClaudeClient, GenerationResult, RawCitation  # noqa: E402
from ingestion.pipeline import IngestionPipeline  # noqa: E402
from retrieval.embeddings import HashEmbedder  # noqa: E402
from retrieval.hybrid import HybridRetriever  # noqa: E402
from retrieval.store import InMemoryStore  # noqa: E402

CORPUS = [
    (
        "https://docs.aws.amazon.com/eks/control-plane.html",
        "# Amazon EKS control plane\n\n"
        "Amazon EKS runs and manages the Kubernetes control plane across multiple "
        "availability zones. The control plane is automatically patched and is covered "
        "by a service level agreement.\n\n"
        "Customers consume the Kubernetes API and do not operate control plane nodes.",
    ),
    (
        "https://kubernetes.io/docs/concepts/scaling.html",
        "# Scaling workloads\n\n"
        "The Horizontal Pod Autoscaler adds Pod replicas when utilisation rises. "
        "Node autoscaling adds worker machines when Pods cannot be scheduled.\n\n"
        "Pod scaling uses capacity that already exists, so it is attempted before "
        "node scaling.",
    ),
    (
        "https://docs.aws.amazon.com/eks/errors.html",
        "# Error reference\n\n"
        "Error code E4021 indicates that a GPU allocation request could not be "
        "satisfied by any node in the cluster.",
    ),
    (
        "https://random-content-farm.example/kubernetes-hot-takes",
        "# Ten things about Kubernetes\n\n"
        "In my opinion the control plane is basically magic and you should never "
        "think about it. Node scaling is always better than pod scaling.",
    ),
]


@pytest.fixture
def settings() -> Settings:
    s = Settings(
        chunk_size_tokens=120,
        chunk_overlap_tokens=20,
        final_context_chunks=6,
        retrieve_candidates=20,
        min_relevance_score=0.2,
        min_evidence_score_to_answer=0.3,
        min_supported_claim_ratio=0.8,
        web_research_enabled=False,
    )
    s.validate()
    return s


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(dimensions=256)


@pytest.fixture
def store(embedder, settings) -> InMemoryStore:
    s = InMemoryStore()
    pipeline = IngestionPipeline(store=s, embedder=embedder, settings=settings)
    for url, text in CORPUS:
        pipeline.ingest_text(text, url=url, suffix=".md")
    return s


@pytest.fixture
def retriever(store, embedder, settings) -> HybridRetriever:
    return HybridRetriever(store, embedder, settings=settings)


class FakeClaudeClient(ClaudeClient):
    """Scriptable stand-in for the model.

    `answer_text` is what the model "writes"; `citation_specs` are (document
    index, exact substring) pairs it "cites". Because the validator re-checks
    cited text against the stored chunk, a test can script a *fabricated*
    citation simply by passing a substring that is not present - which is how
    the hallucination defences are exercised.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        answer_text: str = "Amazon EKS runs and manages the Kubernetes control plane.",
        citation_specs: list[tuple[int, str]] | None = None,
        claims: list[str] | None = None,
        claim_verdicts: dict[str, str] | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        super().__init__(settings, client=object())
        self.answer_text = answer_text
        self.citation_specs = citation_specs if citation_specs is not None else [(0, "runs and manages the Kubernetes control plane")]
        self.scripted_claims = claims
        self.claim_verdicts = claim_verdicts or {}
        self.fail_with = fail_with
        self.calls: list[dict] = []
        self.last_documents: list[dict] = []

    def generate_grounded_answer(self, question, chunks, *, subquestions=None, web_tools=None):
        if self.fail_with:
            raise self.fail_with
        from generation.documents import chunks_to_document_blocks

        self.last_documents = chunks_to_document_blocks(chunks)
        self.calls.append(
            {"question": question, "chunks": len(chunks), "web_tools": web_tools,
             "subquestions": subquestions}
        )
        return GenerationResult(
            text=self.answer_text,
            citations=[
                RawCitation(document_index=i, document_title=None, cited_text=t)
                for i, t in self.citation_specs
            ],
            model=self.settings.model,
        )

    def extract_claims(self, answer: str) -> list[str]:
        if self.scripted_claims is not None:
            return list(self.scripted_claims)
        return [s.strip() for s in answer.split(".") if len(s.strip()) > 20]

    def verify_claim(self, claim: str, evidence: str) -> tuple[str, str]:
        if claim in self.claim_verdicts:
            return self.claim_verdicts[claim], "scripted verdict"
        # Default: a claim is supported when its content words appear in the
        # evidence. Deliberately mechanical, so tests assert on pipeline
        # behaviour rather than on model judgement.
        words = [w for w in claim.lower().split() if len(w) > 4]
        if not words:
            return "unsupported", "no substantive content"
        hits = sum(1 for w in words if w.strip(".,;:()") in evidence.lower())
        return ("supported", "content words present") if hits / len(words) >= 0.6 else (
            "unsupported", "content words absent from evidence"
        )


@pytest.fixture
def fake_client_factory(settings):
    def _make(**kwargs) -> FakeClaudeClient:
        return FakeClaudeClient(settings=settings, **kwargs)
    return _make


@pytest.fixture
def assistant_factory(retriever, settings):
    def _make(client: FakeClaudeClient) -> ResearchAssistant:
        return ResearchAssistant(
            retriever, client, settings=settings,
            validator=CitationValidator(),
            verifier=ClaimVerifier(client, settings=settings),
        )
    return _make
