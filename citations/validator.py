"""Citation validation.

Every citation the API returns is re-checked against our own stored copy of the
chunk before it reaches the user. A citation survives only if:

  1. its `document_index` maps to a chunk we actually sent, and
  2. its `cited_text` genuinely appears in that chunk's stored text.

This is belt and braces. The API generates citations against the document
content, so fabrication should be impossible - but "should be impossible" is
not a property you want to rely on when the whole product promise is that
citations can be trusted. Verification here is cheap, deterministic, and runs
in code, so it holds regardless of model behaviour.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from generation.client import RawCitation
from schemas import Citation, ScoredChunk

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
# Shorter than this and a "match" is coincidence rather than evidence.
MIN_CITED_CHARS = 12


def normalise(text: str) -> str:
    """Whitespace- and unicode-insensitive form for substring comparison.

    The API may normalise quotes or collapse whitespace relative to our stored
    copy; that is a formatting difference, not a fabricated citation.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace(" ", " ")
    return _WS.sub(" ", text).strip().lower()


@dataclass
class ValidationReport:
    valid: list[Citation] = field(default_factory=list)
    rejected: list[tuple[RawCitation, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.valid) + len(self.rejected)

    @property
    def all_verified(self) -> bool:
        return not self.rejected

    def summary(self) -> str:
        return f"{len(self.valid)}/{self.total} citations verified against stored evidence"


class CitationValidator:
    """Binds API citations to stored chunks, dropping anything unverifiable."""

    def __init__(self, *, min_cited_chars: int = MIN_CITED_CHARS) -> None:
        self.min_cited_chars = min_cited_chars

    def validate(
        self, raw_citations: list[RawCitation], chunks: list[ScoredChunk]
    ) -> ValidationReport:
        """`chunks` must be in the same order as the document blocks that were sent."""
        report = ValidationReport()
        seen: dict[tuple[int, str], int] = {}

        for raw in raw_citations:
            reason = self._reject_reason(raw, chunks)
            if reason:
                logger.warning(
                    "citation rejected",
                    extra={"reason": reason, "document_index": raw.document_index,
                           "cited_text": raw.cited_text[:80]},
                )
                report.rejected.append((raw, reason))
                continue

            scored = chunks[raw.document_index]
            key = (raw.document_index, normalise(raw.cited_text))
            if key in seen:
                continue  # same passage cited twice; keep one entry

            md = scored.chunk.metadata
            seen[key] = len(report.valid)
            report.valid.append(
                Citation(
                    index=len(report.valid) + 1,
                    chunk_id=scored.chunk.chunk_id,
                    document_id=scored.chunk.document_id,
                    title=md.title,
                    url=md.url,
                    domain=md.domain,
                    tier=md.tier,
                    cited_text=raw.cited_text.strip(),
                    char_start=raw.start_char_index,
                    char_end=raw.end_char_index,
                    relevance_score=scored.rerank_score,
                    retrieved_at=md.retrieved_at,
                    verified=True,
                )
            )

        if report.rejected:
            logger.warning("some citations failed verification", extra={"count": len(report.rejected)})
        return report

    def _reject_reason(self, raw: RawCitation, chunks: list[ScoredChunk]) -> str | None:
        """Returns None if the citation is sound, else why it was dropped."""
        if not raw.cited_text or not raw.cited_text.strip():
            return "empty cited_text"
        if raw.document_index is None or raw.document_index < 0:
            return "missing document_index"
        if raw.document_index >= len(chunks):
            return f"document_index {raw.document_index} out of range (sent {len(chunks)})"

        cited = normalise(raw.cited_text)
        if len(cited) < self.min_cited_chars:
            return f"cited_text too short to verify ({len(cited)} chars)"

        stored = normalise(chunks[raw.document_index].chunk.text)
        if cited not in stored:
            return "cited_text does not appear in the stored chunk"
        return None
