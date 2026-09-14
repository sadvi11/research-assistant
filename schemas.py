"""Core domain models shared across the pipeline.

Pydantic is used at every boundary: ingestion output, retrieval results, and the
API surface. Validation failures are loud rather than silently coerced.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from config.sources import SourceTier


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SourceType(StrEnum):
    OFFICIAL_DOCS = "official_documentation"
    GOVERNMENT = "government"
    STANDARD = "standard"
    PEER_REVIEWED = "peer_reviewed"
    ACADEMIC = "academic"
    SECONDARY = "secondary"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Provenance for an ingested document. Required by spec section 3."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    url: str
    domain: str
    source_type: SourceType = SourceType.UNKNOWN
    tier: SourceTier = SourceTier.UNVETTED
    publication_date: datetime | None = None
    author: str | None = None
    retrieved_at: datetime = Field(default_factory=_utcnow)
    content_sha256: str | None = None

    @field_validator("title")
    @classmethod
    def _title_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document title must not be blank")
        return v.strip()


class Chunk(BaseModel):
    """A retrievable unit. Always carries its source URL - spec section 7."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    text: str
    ordinal: int
    char_start: int
    char_end: int
    metadata: DocumentMetadata
    section_path: tuple[str, ...] = ()

    @property
    def source_url(self) -> str:
        return self.metadata.url

    @property
    def tier(self) -> SourceTier:
        return self.metadata.tier


class ScoredChunk(BaseModel):
    """A chunk with its retrieval provenance attached.

    Keeping the component scores (not just the fused one) is what makes
    retrieval debuggable - you can see whether a hit came from semantic
    similarity, lexical match, or both.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None

    @property
    def absolute_relevance(self) -> float:
        """Relevance on an absolute scale, for gating.

        `score` and `rerank_score` are *relative* - they are fused ranks and
        normalised boosts, so the best hit for any query scores ~1.0 whether or
        not it is actually relevant. Gating on those would let an off-topic
        question through, because something is always the best of a bad set.

        Dense cosine similarity is genuinely absolute and comparable across
        queries, so it is the measure a refusal decision can rest on.
        """
        return self.dense_score if self.dense_score is not None else 0.0

    @property
    def retrieval_method(self) -> str:
        if self.dense_rank is not None and self.lexical_rank is not None:
            return "hybrid"
        if self.dense_rank is not None:
            return "dense"
        if self.lexical_rank is not None:
            return "lexical"
        return "unknown"


class Citation(BaseModel):
    """A citation returned by the model AND verified against stored evidence.

    `verified` is set by citations.validator - it is never taken on trust from
    the model. An unverified citation is dropped before the answer is returned.
    """

    model_config = ConfigDict(frozen=True)

    index: int
    chunk_id: str
    document_id: str
    title: str
    url: str
    domain: str
    tier: SourceTier
    cited_text: str
    char_start: int | None = None
    char_end: int | None = None
    relevance_score: float | None = None
    retrieved_at: datetime | None = None
    verified: bool = False


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


class Claim(BaseModel):
    """A factual assertion extracted from the draft answer (spec section 5.1)."""

    text: str
    status: ClaimStatus = ClaimStatus.UNSUPPORTED
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    reasoning: str | None = None


class SourceConflict(BaseModel):
    """Two trusted sources disagreeing - surfaced, never silently merged."""

    topic: str
    position_a: str
    position_b: str
    source_a_url: str
    source_b_url: str
    source_a_tier: SourceTier
    source_b_tier: SourceTier
    more_authoritative_url: str | None = None
    rationale: str | None = None


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REFUSED = "refused"
    ERROR = "error"


class EvidenceSummary(BaseModel):
    """What the user needs in order to judge how much to trust the answer."""

    chunks_retrieved: int = 0
    chunks_used: int = 0
    best_relevance_score: float | None = None
    tier_1_sources: int = 0
    lowest_tier_used: SourceTier | None = None
    independent_sources: int = 0
    claims_total: int = 0
    claims_supported: int = 0

    @property
    def supported_ratio(self) -> float:
        if self.claims_total == 0:
            return 0.0
        return self.claims_supported / self.claims_total


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find sufficient evidence in the trusted sources to answer this "
    "confidently."
)


class ResearchAnswer(BaseModel):
    """The complete response surface. Everything the UI renders comes from here."""

    question: str
    status: AnswerStatus
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[SourceConflict] = Field(default_factory=list)
    evidence: EvidenceSummary = Field(default_factory=EvidenceSummary)
    subquestions: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    model: str | None = None
    latency_ms: float | None = None
    request_id: str | None = None

    @property
    def is_grounded(self) -> bool:
        return self.status is AnswerStatus.ANSWERED and bool(self.citations)

    @classmethod
    def insufficient(
        cls,
        question: str,
        *,
        evidence: EvidenceSummary | None = None,
        notes: list[str] | None = None,
        request_id: str | None = None,
    ) -> "ResearchAnswer":
        return cls(
            question=question,
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            evidence=evidence or EvidenceSummary(),
            uncertainty_notes=notes or [],
            request_id=request_id,
        )


Score = Annotated[float, Field(ge=0.0, le=1.0)]
