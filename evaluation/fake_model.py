"""An extractive stand-in model, for running the pipeline at zero cost.

This is not an attempt to imitate Claude. It is a deliberately simple,
deterministic reader that:

  * answers by quoting the retrieved evidence verbatim, and
  * declines when the question's key terms are absent from that evidence.

That is enough to exercise every branch of the pipeline - citation binding,
claim verification, and both refusal gates - without an API call. If the
pipeline only behaves correctly when the model is clever, the pipeline is
wrong; this fake is how that assumption gets tested.
"""
from __future__ import annotations

import re

from config.settings import Settings
from generation.client import ClaudeClient, GenerationResult, RawCitation
from retrieval.store import _STOPWORDS
from schemas import ScoredChunk

_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

# Question words that signal a demand for a specific value. If the question asks
# for one of these and the evidence contains no such value, answering would mean
# inventing it - which is precisely the behaviour under test.
_QUANTITY_CUES = {
    "how much", "how many", "what percentage", "what percent", "cost", "price",
    "pricing", "revenue", "sla percentage", "throughput", "benchmark",
}
_NUMBER = re.compile(r"\b\d[\d.,]*\s*(?:%|percent|per cent)?\b")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 2}


class ExtractiveFakeClient(ClaudeClient):
    """Quotes evidence, or refuses. Never invents."""

    def __init__(self, settings: Settings, *, coverage_threshold: float = 0.4) -> None:
        super().__init__(settings, client=object())
        self.coverage_threshold = coverage_threshold
        self.calls = 0

    def generate_grounded_answer(self, question, chunks, *, subquestions=None, web_tools=None):
        self.calls += 1
        if not chunks:
            return GenerationResult(text="INSUFFICIENT_EVIDENCE\nNo documents were retrieved.")

        q_words = _content_words(question)
        evidence_text = " ".join(c.chunk.text for c in chunks)
        evidence_words = _content_words(evidence_text)

        # 1. Does the evidence cover what was actually asked?
        covered = q_words & evidence_words
        coverage = len(covered) / len(q_words) if q_words else 0.0
        if coverage < self.coverage_threshold:
            missing = sorted(q_words - evidence_words)[:5]
            return GenerationResult(
                text=f"INSUFFICIENT_EVIDENCE\nThe documents do not discuss: {', '.join(missing)}."
            )

        # 2. A question demanding a specific figure, where the evidence states
        #    none, must be refused rather than answered with a plausible number.
        lowered = question.lower()
        if any(cue in lowered for cue in _QUANTITY_CUES) and not _NUMBER.search(evidence_text):
            return GenerationResult(
                text="INSUFFICIENT_EVIDENCE\nThe documents mention the topic but state no "
                     "specific figure, so none can be given."
            )

        # 3. Answer extractively: quote the sentences that overlap the question.
        best_index, best_sentence, best_overlap = 0, "", 0
        for index, scored in enumerate(chunks):
            for sentence in _SENTENCE.split(scored.chunk.text):
                overlap = len(_content_words(sentence) & q_words)
                if overlap > best_overlap:
                    best_index, best_sentence, best_overlap = index, sentence.strip(), overlap

        if not best_sentence or best_overlap == 0:
            return GenerationResult(
                text="INSUFFICIENT_EVIDENCE\nNo retrieved passage addresses the question."
            )

        return GenerationResult(
            text=best_sentence,
            citations=[RawCitation(best_index, None, best_sentence)],
            model="extractive-fake",
        )

    def extract_claims(self, answer: str) -> list[str]:
        if "INSUFFICIENT_EVIDENCE" in answer:
            return []
        return [s.strip() for s in _SENTENCE.split(answer) if len(s.strip()) > 20]

    def verify_claim(self, claim: str, evidence: str) -> tuple[str, str]:
        """Supported iff the claim's content words are present in the evidence.

        Since this fake only ever quotes evidence verbatim, supported is the
        expected verdict - which is the point: an extractive reader should score
        100% groundedness, so any shortfall indicates a pipeline defect rather
        than a model one.
        """
        words = _content_words(claim)
        if not words:
            return "unsupported", "no substantive content"
        hits = len(words & _content_words(evidence))
        ratio = hits / len(words)
        if ratio >= 0.8:
            return "supported", f"{hits}/{len(words)} content words present in evidence"
        return "unsupported", f"only {hits}/{len(words)} content words present"

    def _structured(self, prompt: str, schema: dict, max_tokens: int = 2048) -> dict:
        return {"subquestions": []}
