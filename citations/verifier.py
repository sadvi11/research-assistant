"""Claim-level verification (spec section 5.1).

Citations prove that *some* text came from a source. They do not prove that
every sentence in the answer is supported - a model can write three grounded
sentences and one unsupported one, and the unsupported one carries no citation
precisely because there is nothing to cite.

So the draft answer is decomposed into claims, each claim is checked against the
retrieved evidence, and unsupported claims are removed. If too few claims
survive, the answer is withheld entirely rather than shipped with holes in it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import Settings, get_settings
from generation.client import ClaudeClient, GenerationError
from schemas import Claim, ClaimStatus, ScoredChunk

logger = logging.getLogger(__name__)


@dataclass
class VerificationReport:
    claims: list[Claim] = field(default_factory=list)
    checked: bool = True
    note: str | None = None

    @property
    def total(self) -> int:
        return len(self.claims)

    @property
    def supported(self) -> int:
        return sum(1 for c in self.claims if c.status is ClaimStatus.SUPPORTED)

    @property
    def contradicted(self) -> list[Claim]:
        return [c for c in self.claims if c.status is ClaimStatus.CONTRADICTED]

    @property
    def unsupported(self) -> list[Claim]:
        return [c for c in self.claims if c.status is ClaimStatus.UNSUPPORTED]

    @property
    def supported_ratio(self) -> float:
        if not self.claims:
            return 0.0
        return self.supported / len(self.claims)

    def meets_threshold(self, threshold: float) -> bool:
        """No claims extracted is not a pass - it means we could not check."""
        if not self.claims:
            return False
        if self.contradicted:
            return False  # a contradicted claim is disqualifying on its own
        return self.supported_ratio >= threshold


class ClaimVerifier:
    def __init__(self, client: ClaudeClient, *, settings: Settings | None = None) -> None:
        self.client = client
        self.settings = settings or get_settings()

    @staticmethod
    def _evidence_bundle(chunks: list[ScoredChunk], limit: int = 12) -> str:
        parts = []
        for i, sc in enumerate(chunks[:limit]):
            parts.append(f"[{i}] ({sc.chunk.metadata.domain}) {sc.chunk.text}")
        return "\n\n".join(parts)

    def verify(self, answer: str, chunks: list[ScoredChunk]) -> VerificationReport:
        """Extract claims from `answer` and check each against `chunks`.

        A failure to verify is reported as such - it never silently becomes a
        pass, because "we could not check" and "we checked and it is fine" must
        not look the same to the caller.
        """
        if not answer.strip():
            return VerificationReport(checked=False, note="empty answer")
        if not chunks:
            return VerificationReport(checked=False, note="no evidence to verify against")

        try:
            claim_texts = self.client.extract_claims(answer)
        except GenerationError as exc:
            logger.warning("claim extraction failed", extra={"error": str(exc)})
            return VerificationReport(checked=False, note=f"claim extraction failed: {exc}")

        if not claim_texts:
            return VerificationReport(checked=True, note="no verifiable factual claims found")

        evidence = self._evidence_bundle(chunks)
        claims: list[Claim] = []
        for text in claim_texts:
            try:
                status, reasoning = self.client.verify_claim(text, evidence)
            except GenerationError as exc:
                logger.warning("claim verification failed", extra={"error": str(exc)})
                return VerificationReport(
                    claims=claims, checked=False, note=f"verification failed mid-run: {exc}"
                )
            try:
                parsed = ClaimStatus(status)
            except ValueError:
                parsed = ClaimStatus.UNSUPPORTED
            claims.append(
                Claim(
                    text=text,
                    status=parsed,
                    reasoning=reasoning or None,
                    supporting_chunk_ids=(
                        [c.chunk.chunk_id for c in chunks[:3]]
                        if parsed is ClaimStatus.SUPPORTED
                        else []
                    ),
                )
            )

        report = VerificationReport(claims=claims)
        logger.info(
            "claim verification complete",
            extra={"total": report.total, "supported": report.supported,
                   "contradicted": len(report.contradicted),
                   "ratio": round(report.supported_ratio, 3)},
        )
        return report
