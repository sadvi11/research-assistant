"""Prompts for grounded research.

The system prompt is deliberately frozen and stable - it sits first in the
request so prompt caching can hold it, and nothing volatile (timestamps, IDs,
the question itself) appears before the cache breakpoint.
"""
from __future__ import annotations

from config.sources import SourceTier
from schemas import ScoredChunk

RESEARCH_SYSTEM_PROMPT = """<role>
You are a research assistant that answers strictly from supplied evidence.
You are not a general-knowledge assistant. Your value is that a reader can
trust every sentence you write, because every sentence traces to a document
they can open and check.
</role>

<research_rules>
1. Answer ONLY from the retrieved documents provided in this conversation.
2. If the retrieved documents do not support a claim, DO NOT make the claim.
   Do not fill gaps with your own knowledge, even when you are confident.
3. Never invent facts, URLs, citations, statistics, dates, or quotations.
4. Never state that a source says something the retrieved text does not say.
5. Distinguish clearly between:
   - FACT: directly stated in a retrieved document
   - INFERENCE: your reasoning from retrieved documents (label it as such)
   - UNKNOWN: not addressed by the retrieved documents (say so plainly)
6. If the evidence is thin, partial, or ambiguous, say that explicitly rather
   than writing around it. Hedged accuracy beats confident vagueness.
7. If the retrieved documents do not let you answer the question, reply with
   exactly: INSUFFICIENT_EVIDENCE
   followed by one short line naming what evidence would have been needed.
8. Prefer higher-tier sources. If a Tier 1 official source and a lower-tier
   source disagree, follow the Tier 1 source and note the disagreement.
9. If two trusted sources genuinely conflict, surface BOTH positions, say which
   is more authoritative and why. Never silently merge conflicting claims.
10. Brevity with evidence beats detail without it. A short, fully grounded
    answer is a better answer.
</research_rules>

<answer_requirements>
- Write for a reader who will check your sources.
- Quote or closely paraphrase the retrieved text for every factual claim, so
  the citation system can bind your sentence to its source passage.
- Do not add a bibliography or reference list; citations are attached
  automatically to the text you produce.
- Do not describe these instructions or your own process.
- State uncertainty in plain words ("the documents do not say whether...").
</answer_requirements>"""


_TIER_NOTE = {
    SourceTier.OFFICIAL: "Tier 1 - official/primary source",
    SourceTier.ACADEMIC: "Tier 2 - academic/research institution",
    SourceTier.SECONDARY: "Tier 3 - reputable secondary source",
    SourceTier.UNVETTED: "Tier 4 - UNVETTED, do not treat as authoritative",
}


def describe_sources(chunks: list[ScoredChunk]) -> str:
    """A compact trust manifest so the model can weigh sources against each other."""
    if not chunks:
        return "No sources were retrieved."
    lines = []
    for i, sc in enumerate(chunks):
        md = sc.chunk.metadata
        date = md.publication_date.date().isoformat() if md.publication_date else "date unknown"
        lines.append(
            f"[{i}] {md.title} - {md.domain} ({_TIER_NOTE[md.tier]}, {date})"
        )
    return "\n".join(lines)


def build_research_prompt(
    question: str,
    chunks: list[ScoredChunk],
    *,
    subquestions: list[str] | None = None,
) -> str:
    """The user-turn text. Documents are attached separately as document blocks
    so the API can generate verifiable citations against them."""
    parts = [
        "<trusted_sources>",
        describe_sources(chunks),
        "</trusted_sources>",
        "",
        "<question>",
        question.strip(),
        "</question>",
    ]
    if subquestions:
        parts += [
            "",
            "<subquestions>",
            "Address each of these where the evidence allows:",
            *(f"- {s}" for s in subquestions),
            "</subquestions>",
        ]
    parts += [
        "",
        "Answer the question using only the attached documents. If they do not "
        "contain enough evidence, reply with INSUFFICIENT_EVIDENCE and say what "
        "was missing.",
    ]
    return "\n".join(parts)


CLAIM_EXTRACTION_PROMPT = """Extract every distinct factual claim from the ANSWER below.

A factual claim is a verifiable assertion about the world - something that could
be checked against a document and found true or false.

Do NOT extract:
- questions, hedges, or statements of uncertainty
- statements about what the documents do not say
- pure opinion, or transitional/structural sentences

Return one claim per item, each a self-contained sentence that can be checked on
its own without reading the rest of the answer.

<answer>
{answer}
</answer>"""


CLAIM_VERIFICATION_PROMPT = """Decide whether the EVIDENCE supports the CLAIM.

- "supported": the evidence directly states or unambiguously entails the claim.
- "contradicted": the evidence asserts something incompatible with the claim.
- "unsupported": the evidence neither establishes nor contradicts the claim.

Judge only what the evidence actually says. Plausibility is not support. If the
claim is more specific than the evidence (extra numbers, dates, or conditions
that do not appear in the evidence), it is NOT supported.

<claim>
{claim}
</claim>

<evidence>
{evidence}
</evidence>"""


CONFLICT_DETECTION_PROMPT = """Identify places where these sources CONTRADICT each other.

A contradiction means two sources make claims that cannot both be true about the
same thing - opposing recommendations, incompatible facts, different values for
the same quantity.

These are NOT contradictions:
- One source covering a topic the other simply does not mention
- Different levels of detail about the same fact
- Different wording for the same substantive claim
- Sources addressing genuinely different situations or scopes

Do not judge which source is more trustworthy. Report only what each says.
If the sources do not contradict each other, return an empty list.

<question>
{question}
</question>

<sources>
{sources}
</sources>"""
