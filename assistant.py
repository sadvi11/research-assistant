"""The research assistant: the pipeline that enforces evidence over fluency.

The order of operations is the product. Each stage can only ever *reduce*
confidence - there is no path that adds an unsupported claim back in:

    validate input
      -> decompose (research mode)
      -> retrieve
      -> GATE: is there enough evidence to be worth asking the model?
      -> generate with API citations
      -> GATE: did the model itself declare insufficient evidence?
      -> validate every citation against stored chunks
      -> verify every claim against retrieved evidence
      -> GATE: did enough claims survive?
      -> answer

Any gate that fails produces an insufficient-evidence response, never a
best-effort guess.
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass

from citations.conflicts import ConflictDetector, conflict_notes
from citations.validator import CitationValidator
from citations.verifier import ClaimVerifier, VerificationReport
from config.settings import Settings, get_settings
from config.sources import SourceTier
from generation.client import ClaudeClient, GenerationError, INSUFFICIENT_MARKER
from retrieval.hybrid import HybridRetriever
from schemas import (
    AnswerStatus,
    Claim,
    ClaimStatus,
    Citation,
    EvidenceSummary,
    ResearchAnswer,
    ScoredChunk,
)
from web_research import build_web_tools

logger = logging.getLogger(__name__)

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class InvalidQuestionError(ValueError):
    pass


def validate_question(question: str, settings: Settings) -> str:
    """Input validation. Rejects rather than silently truncating."""
    if question is None:
        raise InvalidQuestionError("question is required")
    cleaned = _CONTROL_CHARS.sub("", question).strip()
    if not cleaned:
        raise InvalidQuestionError("question must not be empty")
    if len(cleaned) > settings.max_question_chars:
        raise InvalidQuestionError(
            f"question exceeds {settings.max_question_chars} characters "
            f"({len(cleaned)} given)"
        )
    return cleaned


@dataclass
class _Gate:
    passed: bool
    note: str | None = None


class ResearchAssistant:
    def __init__(
        self,
        retriever: HybridRetriever,
        client: ClaudeClient,
        *,
        settings: Settings | None = None,
        validator: CitationValidator | None = None,
        verifier: ClaimVerifier | None = None,
        conflict_detector: ConflictDetector | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever
        self.client = client
        self.validator = validator or CitationValidator()
        self.verifier = verifier or ClaimVerifier(client, settings=self.settings)
        self.conflicts = conflict_detector or ConflictDetector(client, settings=self.settings)

    # -- public ------------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        research_mode: bool = False,
        max_tier: SourceTier | None = None,
        use_web: bool | None = None,
    ) -> ResearchAnswer:
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        question = validate_question(question, self.settings)

        logger.info(
            "research request",
            extra={"request_id": request_id, "research_mode": research_mode,
                   "question_chars": len(question)},
        )

        subquestions: list[str] = []
        if research_mode:
            subquestions = self._decompose(question)

        chunks = self._retrieve(question, subquestions, max_tier=max_tier)
        evidence = self._summarise_evidence(chunks)

        web_tools = build_web_tools(settings=self.settings) if (use_web is not False) else []

        gate = self._evidence_gate(chunks, bool(web_tools))
        if not gate.passed:
            return self._insufficient(question, evidence, [gate.note or ""], request_id, started)

        try:
            generated = self.client.generate_grounded_answer(
                question, chunks, subquestions=subquestions or None,
                web_tools=web_tools or None,
            )
        except Exception as exc:
            # Deliberately broad: the orchestrator is a boundary. Any escaping
            # exception becomes a caller's unhandled crash rather than a safe
            # ERROR status, so nothing is allowed through.
            logger.exception("generation failed", extra={"request_id": request_id})
            return ResearchAnswer(
                question=question, status=AnswerStatus.ERROR,
                answer=f"The research request could not be completed: {exc}",
                evidence=evidence, request_id=request_id,
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # The model is allowed to refuse. Honour it rather than overriding.
        if generated.declared_insufficient:
            note = generated.text.replace(INSUFFICIENT_MARKER, "").strip()
            return self._insufficient(
                question, evidence,
                [note or "The model judged the retrieved evidence insufficient."],
                request_id, started, subquestions=subquestions,
            )

        validation = self.validator.validate(generated.citations, chunks)
        verification = self.verifier.verify(generated.text, chunks)

        evidence = evidence.model_copy(
            update={
                "claims_total": verification.total,
                "claims_supported": verification.supported,
                "chunks_used": len({c.chunk_id for c in validation.valid}),
            }
        )

        notes: list[str] = []
        if validation.rejected:
            notes.append(
                f"{len(validation.rejected)} citation(s) could not be verified against the "
                "stored source text and were removed."
            )
        if not verification.checked and verification.note:
            notes.append(f"Claim verification incomplete: {verification.note}")

        claim_gate = self._claim_gate(verification)
        if not claim_gate.passed:
            notes.append(claim_gate.note or "")
            return self._insufficient(
                question, evidence, notes, request_id, started,
                subquestions=subquestions, claims=verification.claims,
            )

        if not validation.valid:
            notes.append(
                "No citation could be bound to a stored source passage, so the answer "
                "cannot be shown as evidence-grounded."
            )
            return self._insufficient(
                question, evidence, notes, request_id, started,
                subquestions=subquestions, claims=verification.claims,
            )

        # A Tier-4-only answer is not evidence-grounded research, however well
        # its citations verify. Citation authenticity and source trust are
        # different properties and must not be conflated.
        trust_gate = self._trust_gate(validation.valid)
        if not trust_gate.passed:
            notes.append(trust_gate.note or "")
            return self._insufficient(
                question, evidence, notes, request_id, started,
                subquestions=subquestions, claims=verification.claims,
            )

        # Conflicts are additive: a failure here degrades the answer's richness,
        # never its correctness, so it must not block the response.
        detected = self.conflicts.detect(question, chunks)
        notes.extend(conflict_notes(detected))
        notes.extend(self._uncertainty_notes(validation.valid, chunks, verification))

        return ResearchAnswer(
            question=question,
            status=AnswerStatus.ANSWERED,
            answer=generated.text,
            citations=validation.valid,
            claims=verification.claims,
            conflicts=detected,
            evidence=evidence,
            subquestions=subquestions,
            uncertainty_notes=[n for n in notes if n],
            model=generated.model,
            request_id=request_id,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    # -- stages ------------------------------------------------------------

    def _decompose(self, question: str) -> list[str]:
        """Research mode: break a complex question into searchable subquestions.

        Failure here is non-fatal - we fall back to searching the original
        question rather than failing the request.
        """
        try:
            subs = self.client.extract_claims  # presence check only
            from generation.prompts import build_research_prompt  # noqa: F401
            result = self.client._structured(  # noqa: SLF001 - internal by design
                "Break this research question into at most "
                f"{self.settings.max_subquestions} independent sub-questions that could each "
                "be answered by searching a document corpus. Keep them short and factual.\n\n"
                f"<question>\n{question}\n</question>",
                {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "subquestions": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["subquestions"],
                        "additionalProperties": False,
                    },
                },
                max_tokens=1024,
            )
            return [s.strip() for s in result.get("subquestions", []) if s.strip()][
                : self.settings.max_subquestions
            ]
        except Exception as exc:
            logger.warning("subquestion decomposition failed", extra={"error": str(exc)})
            return []

    def _retrieve(
        self, question: str, subquestions: list[str], *, max_tier: SourceTier | None
    ) -> list[ScoredChunk]:
        """Retrieve for the question plus every subquestion, then merge.

        Merging keeps the best score per chunk so a chunk relevant to several
        subquestions is not double-counted, and stays ordered by relevance.
        """
        queries = [question, *subquestions]
        best: dict[str, ScoredChunk] = {}
        for query in queries:
            for scored in self.retriever.retrieve(query, max_tier=max_tier):
                existing = best.get(scored.chunk.chunk_id)
                if existing is None or (scored.rerank_score or 0) > (existing.rerank_score or 0):
                    best[scored.chunk.chunk_id] = scored

        merged = sorted(best.values(), key=lambda s: s.rerank_score or 0.0, reverse=True)
        return merged[: self.settings.final_context_chunks]

    def _evidence_gate(self, chunks: list[ScoredChunk], has_web: bool) -> _Gate:
        """Do not spend a model call on evidence that cannot support an answer."""
        if not chunks:
            if has_web:
                return _Gate(True)
            return _Gate(False, "No documents in the corpus matched this question.")
        if len(chunks) < self.settings.min_evidence_chunks:
            return _Gate(False, f"Only {len(chunks)} chunk(s) retrieved; below the minimum.")
        # Gate on ABSOLUTE relevance, never the normalised ranking score - the
        # top hit for an off-topic question also normalises to 1.0, so ranking
        # score cannot distinguish "best match" from "good match".
        best_absolute = max(c.absolute_relevance for c in chunks)
        # A strong exact-term hit is real evidence even when cosine is modest
        # (error codes, identifiers, proper nouns).
        strong_lexical = any(
            c.lexical_rank is not None and (c.lexical_score or 0.0) >= 1.0 for c in chunks
        )

        if best_absolute < self.settings.min_evidence_score_to_answer and not strong_lexical:
            if has_web:
                return _Gate(True)
            return _Gate(
                False,
                f"The closest passage in the corpus scored {best_absolute:.2f} for semantic "
                f"relevance, below the {self.settings.min_evidence_score_to_answer:.2f} "
                "threshold required to answer. Nothing ingested addresses this question.",
            )
        return _Gate(True)

    def _trust_gate(self, citations: list[Citation]) -> _Gate:
        """Refuse when every cited source is unvetted.

        The citation validator proves a passage really came from the chunk it
        names. It says nothing about whether that source deserved to be trusted.
        This gate supplies the missing half.
        """
        if not citations:
            return _Gate(True)  # handled separately by the no-citation path
        if all(c.tier is SourceTier.UNVETTED for c in citations):
            domains = sorted({c.domain for c in citations})
            return _Gate(
                False,
                "Every supporting source is unvetted "
                f"({', '.join(domains)}), so the answer was withheld. Unvetted "
                "sources are never authoritative on their own.",
            )
        return _Gate(True)

    def _claim_gate(self, verification: VerificationReport) -> _Gate:
        if verification.contradicted:
            return _Gate(
                False,
                f"{len(verification.contradicted)} claim(s) were contradicted by the "
                "retrieved evidence, so the answer was withheld.",
            )
        if not verification.checked:
            # Could not check is not the same as checked and fine.
            return _Gate(False, verification.note or "Claim verification could not be completed.")
        if not verification.claims:
            return _Gate(True)  # nothing factual asserted; nothing to disprove
        if not verification.meets_threshold(self.settings.min_supported_claim_ratio):
            return _Gate(
                False,
                f"Only {verification.supported}/{verification.total} claims were supported by "
                f"retrieved evidence, below the required "
                f"{self.settings.min_supported_claim_ratio:.0%}.",
            )
        return _Gate(True)

    # -- helpers -----------------------------------------------------------

    def _summarise_evidence(self, chunks: list[ScoredChunk]) -> EvidenceSummary:
        if not chunks:
            return EvidenceSummary()
        return EvidenceSummary(
            chunks_retrieved=len(chunks),
            best_relevance_score=max(c.absolute_relevance for c in chunks),
            tier_1_sources=sum(1 for c in chunks if c.chunk.tier is SourceTier.OFFICIAL),
            lowest_tier_used=max(c.chunk.tier for c in chunks),
            independent_sources=len({c.chunk.metadata.domain for c in chunks}),
        )

    def _uncertainty_notes(
        self,
        citations: list[Citation],
        chunks: list[ScoredChunk],
        verification: VerificationReport,
    ) -> list[str]:
        """Caveats describe the sources the answer actually RESTS ON.

        Computing these from retrieved chunks was a real defect: an answer could
        cite only a content farm while a Tier 1 document sat unused in the
        retrieved set, and the "no official source" warning would not fire.
        """
        notes: list[str] = []
        cited_domains = {c.domain for c in citations}
        cited_tiers = {c.tier for c in citations}

        if citations and len(cited_domains) < self.settings.prefer_independent_sources:
            notes.append(
                f"This answer rests on a single source domain "
                f"({next(iter(cited_domains), 'unknown')}). Corroboration from an "
                "independent source would strengthen it."
            )
        if citations and SourceTier.OFFICIAL not in cited_tiers:
            lowest = max(cited_tiers)
            notes.append(
                f"No Tier 1 (official/primary) source supports this answer - the most "
                f"authoritative source cited is {lowest.label}."
            )
        elif not citations and not any(c.chunk.tier is SourceTier.OFFICIAL for c in chunks):
            notes.append("No Tier 1 (official/primary) source was available for this question.")
        if verification.unsupported:
            notes.append(
                f"{len(verification.unsupported)} statement(s) in the draft could not be tied to "
                "retrieved evidence; treat them with caution."
            )
        return notes

    def _insufficient(
        self,
        question: str,
        evidence: EvidenceSummary,
        notes: list[str],
        request_id: str,
        started: float,
        *,
        subquestions: list[str] | None = None,
        claims: list[Claim] | None = None,
    ) -> ResearchAnswer:
        answer = ResearchAnswer.insufficient(
            question, evidence=evidence, notes=[n for n in notes if n], request_id=request_id
        )
        logger.info(
            "insufficient evidence",
            extra={"request_id": request_id, "chunks": evidence.chunks_retrieved,
                   "notes": len(answer.uncertainty_notes)},
        )
        return answer.model_copy(
            update={
                "subquestions": subquestions or [],
                "claims": claims or [],
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
        )
