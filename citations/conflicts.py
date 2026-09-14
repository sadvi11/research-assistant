"""Source conflict detection.

Two trusted sources disagreeing is a research finding, not a retrieval error.
Silently picking one destroys the thing that makes the answer worth trusting:
the reader's ability to see that the question is contested.

**The split matters.** The model is asked only to spot a *semantic*
contradiction between passages — it is good at that. It is never asked which
source deserves more weight. Authority is a policy decision derived in code from
the trust tier, because a model's opinion about which domain is more
authoritative is exactly the kind of judgement this system refuses to delegate.

Cost: one additional structured call per query, not one per source pair. A
pairwise comparison would be O(n²) and, at eight chunks, twenty-eight calls.

⚠️ Known tradeoff: detection can only compare sources that survived retrieval.
A dissenting source scoring below `min_absolute_relevance` is filtered out
before it reaches this stage, so a conflict with a marginally-relevant source
goes unreported. Widening the candidate set for detection alone was considered
and rejected: surfacing a conflict against evidence too weak to have been used
in the answer would mislead more than it informs.
"""
from __future__ import annotations

import logging

from config.settings import Settings, get_settings
from config.sources import SourceTier
from generation.client import ClaudeClient, GenerationError
from schemas import ScoredChunk, SourceConflict

logger = logging.getLogger(__name__)


class ConflictDetector:
    def __init__(self, client: ClaudeClient, *, settings: Settings | None = None) -> None:
        self.client = client
        self.settings = settings or get_settings()

    def detect(self, question: str, chunks: list[ScoredChunk]) -> list[SourceConflict]:
        """Find contradictions between sources in the retrieved set.

        Returns [] when detection is impossible or fails. A conflict is additive
        information, so failing to find one must never block an answer — but it
        is logged, because silence and "none found" are different states.
        """
        eligible = self._eligible(chunks)
        if len(eligible) < 2:
            return []

        try:
            raw = self.client.detect_conflicts(question, eligible)
        except GenerationError as exc:
            logger.warning("conflict detection failed", extra={"error": str(exc)})
            return []
        except Exception:
            logger.exception("conflict detection raised unexpectedly")
            return []

        conflicts: list[SourceConflict] = []
        for item in raw:
            conflict = self._build(item, eligible)
            if conflict is not None:
                conflicts.append(conflict)

        if conflicts:
            logger.info(
                "source conflict detected",
                extra={"count": len(conflicts),
                       "domains": sorted({c.source_a_url for c in conflicts})},
            )
        return conflicts

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _eligible(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
        """One chunk per domain, best-scoring first.

        Conflicts are between *sources*, not passages. Two chunks from the same
        page elaborating on each other are not a disagreement.
        """
        by_domain: dict[str, ScoredChunk] = {}
        for scored in chunks:
            domain = scored.chunk.metadata.domain
            current = by_domain.get(domain)
            if current is None or (scored.rerank_score or 0) > (current.rerank_score or 0):
                by_domain[domain] = scored
        return list(by_domain.values())

    def _build(self, item: dict, eligible: list[ScoredChunk]) -> SourceConflict | None:
        """Turn a model-reported contradiction into a verified SourceConflict.

        Indices are bounds-checked and must name two *different* sources - a
        model reporting a source as contradicting itself is discarded rather
        than surfaced.
        """
        try:
            index_a = int(item.get("source_a_index", -1))
            index_b = int(item.get("source_b_index", -1))
        except (TypeError, ValueError):
            return None

        if not (0 <= index_a < len(eligible) and 0 <= index_b < len(eligible)):
            logger.warning("conflict referenced an out-of-range source index")
            return None
        if index_a == index_b:
            return None

        topic = (item.get("topic") or "").strip()
        position_a = (item.get("position_a") or "").strip()
        position_b = (item.get("position_b") or "").strip()
        if not (topic and position_a and position_b):
            return None

        a, b = eligible[index_a].chunk, eligible[index_b].chunk

        # ── Authority is decided HERE, in code, from the trust tier ──────────
        more_authoritative, rationale = self._rank(a.metadata, b.metadata)

        return SourceConflict(
            topic=topic,
            position_a=position_a,
            position_b=position_b,
            source_a_url=a.metadata.url,
            source_b_url=b.metadata.url,
            source_a_tier=a.tier,
            source_b_tier=b.tier,
            more_authoritative_url=more_authoritative,
            rationale=rationale,
        )

    @staticmethod
    def _rank(a, b) -> tuple[str | None, str]:
        """Lower tier number wins. Equal tiers means neither does."""
        if a.tier < b.tier:
            return a.url, (
                f"{a.domain} is {a.tier.label.lower()}; {b.domain} is "
                f"{b.tier.label.lower()}. The higher-tier source is preferred."
            )
        if b.tier < a.tier:
            return b.url, (
                f"{b.domain} is {b.tier.label.lower()}; {a.domain} is "
                f"{a.tier.label.lower()}. The higher-tier source is preferred."
            )
        # Same tier: publication date is a weak tiebreak, and is offered as
        # context rather than as a ruling.
        note = (
            f"Both sources are {a.tier.label.lower()}, so neither is automatically "
            "more authoritative."
        )
        if a.publication_date and b.publication_date and a.publication_date != b.publication_date:
            newer = a if a.publication_date > b.publication_date else b
            note += f" {newer.domain} is the more recent ({newer.publication_date.date()})."
        return None, note


def conflict_notes(conflicts: list[SourceConflict]) -> list[str]:
    """Human-readable caveats for the answer's uncertainty section."""
    notes: list[str] = []
    for c in conflicts:
        if c.more_authoritative_url:
            notes.append(
                f"Sources disagree on {c.topic}. {c.rationale} "
                f"The answer follows {c.more_authoritative_url}."
            )
        else:
            notes.append(
                f"Sources disagree on {c.topic} and {c.rationale.lower()} "
                "Both positions are shown; treat the answer as contested."
            )
    return notes
