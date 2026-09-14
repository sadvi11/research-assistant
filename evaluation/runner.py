"""Evaluation runner.

Scores the six dimensions named in the spec. The two that matter most are the
ones a fluent system fails quietly:

  * refusal_accuracy - did it decline when it should have declined?
  * unsupported_claim_rate - of the claims it made, how many had no evidence?

A system can score well on answer correctness while being untrustworthy. It
cannot score well on those two while being untrustworthy.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evaluation.dataset import EVAL_CASES, EvalCase, ExpectedOutcome
from schemas import AnswerStatus, ClaimStatus, ResearchAnswer


@dataclass
class EvalResult:
    case_id: str
    category: str
    question: str
    expected: str
    actual: str
    outcome_correct: bool
    citations: int = 0
    citations_verified: int = 0
    claims_total: int = 0
    claims_supported: int = 0
    mentions_missing: list[str] = field(default_factory=list)
    forbidden_present: list[str] = field(default_factory=list)
    domains_cited: list[str] = field(default_factory=list)
    expected_domains_missing: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def citation_correctness(self) -> float:
        """Fraction of citations that survived verification against stored text."""
        if self.citations == 0:
            return 1.0  # nothing claimed, nothing to get wrong
        return self.citations_verified / self.citations

    @property
    def unsupported_claim_rate(self) -> float:
        if self.claims_total == 0:
            return 0.0
        return 1 - (self.claims_supported / self.claims_total)

    @property
    def passed(self) -> bool:
        return (
            self.outcome_correct
            and not self.forbidden_present
            and not self.mentions_missing
            and self.error is None
        )


@dataclass
class EvalSummary:
    results: list[EvalResult] = field(default_factory=list)
    started_at: str = ""
    duration_s: float = 0.0

    # -- headline metrics --------------------------------------------------

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def outcome_accuracy(self) -> float:
        return self._ratio(sum(1 for r in self.results if r.outcome_correct))

    @property
    def refusal_accuracy(self) -> float:
        """Of the cases that SHOULD be refused, how many were?"""
        should_refuse = [r for r in self.results if r.expected == ExpectedOutcome.INSUFFICIENT]
        if not should_refuse:
            return 1.0
        return sum(1 for r in should_refuse if r.outcome_correct) / len(should_refuse)

    @property
    def false_answer_rate(self) -> float:
        """The dangerous failure: answered when it should have refused."""
        should_refuse = [r for r in self.results if r.expected == ExpectedOutcome.INSUFFICIENT]
        if not should_refuse:
            return 0.0
        return sum(1 for r in should_refuse if not r.outcome_correct) / len(should_refuse)

    @property
    def citation_correctness(self) -> float:
        cited = [r for r in self.results if r.citations > 0]
        if not cited:
            return 1.0
        return sum(r.citation_correctness for r in cited) / len(cited)

    @property
    def citation_completeness(self) -> float:
        """Of answers given, how many carried at least one verified citation?"""
        answered = [r for r in self.results if r.actual == AnswerStatus.ANSWERED]
        if not answered:
            return 1.0
        return sum(1 for r in answered if r.citations_verified > 0) / len(answered)

    @property
    def groundedness(self) -> float:
        answered = [r for r in self.results if r.actual == AnswerStatus.ANSWERED]
        if not answered:
            return 1.0
        return sum(1 - r.unsupported_claim_rate for r in answered) / len(answered)

    @property
    def unsupported_claim_rate(self) -> float:
        answered = [r for r in self.results if r.actual == AnswerStatus.ANSWERED]
        if not answered:
            return 0.0
        return sum(r.unsupported_claim_rate for r in answered) / len(answered)

    @property
    def hallucination_incidents(self) -> list[EvalResult]:
        """Any case where forbidden content appeared, or a bait question was answered."""
        return [
            r for r in self.results
            if r.forbidden_present
            or (r.category == "hallucination_bait" and r.actual == AnswerStatus.ANSWERED)
        ]

    def _ratio(self, n: int) -> float:
        return n / self.total if self.total else 0.0

    def by_category(self) -> dict[str, tuple[int, int]]:
        out: dict[str, tuple[int, int]] = {}
        for r in self.results:
            passed, total = out.get(r.category, (0, 0))
            out[r.category] = (passed + (1 if r.passed else 0), total + 1)
        return out

    def report(self) -> str:
        lines = [
            "=" * 72,
            "RESEARCH ASSISTANT - EVALUATION REPORT",
            "=" * 72,
            f"Cases: {self.total}    Passed: {self.passed}    "
            f"Duration: {self.duration_s:.1f}s",
            "",
            "HEADLINE METRICS",
            f"  Outcome accuracy        {self.outcome_accuracy:6.1%}   "
            "(answered vs refused, as expected)",
            f"  Refusal accuracy        {self.refusal_accuracy:6.1%}   "
            "(declined when it should have)",
            f"  FALSE ANSWER RATE       {self.false_answer_rate:6.1%}   "
            "<- answered when it should have refused",
            f"  Citation correctness    {self.citation_correctness:6.1%}   "
            "(citations that verify against source)",
            f"  Citation completeness   {self.citation_completeness:6.1%}   "
            "(answers carrying >=1 verified citation)",
            f"  Groundedness            {self.groundedness:6.1%}   "
            "(claims supported by evidence)",
            f"  Unsupported claim rate  {self.unsupported_claim_rate:6.1%}",
            "",
            "BY CATEGORY",
        ]
        for category, (passed, total) in sorted(self.by_category().items()):
            lines.append(f"  {category:20} {passed:>2}/{total:<2}")

        incidents = self.hallucination_incidents
        lines += ["", f"HALLUCINATION INCIDENTS: {len(incidents)}"]
        for r in incidents:
            detail = f"forbidden={r.forbidden_present}" if r.forbidden_present else "bait answered"
            lines.append(f"  [{r.case_id}] {r.question[:52]} - {detail}")

        failures = [r for r in self.results if not r.passed]
        if failures:
            lines += ["", "FAILURES"]
            for r in failures:
                why = r.error or (
                    f"expected {r.expected}, got {r.actual}" if not r.outcome_correct
                    else f"missing={r.mentions_missing} forbidden={r.forbidden_present}"
                )
                lines.append(f"  [{r.case_id}] {r.question[:52]}")
                lines.append(f"        {why}")
        lines.append("=" * 72)
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "started_at": self.started_at,
                "duration_s": round(self.duration_s, 2),
                "metrics": {
                    "outcome_accuracy": round(self.outcome_accuracy, 4),
                    "refusal_accuracy": round(self.refusal_accuracy, 4),
                    "false_answer_rate": round(self.false_answer_rate, 4),
                    "citation_correctness": round(self.citation_correctness, 4),
                    "citation_completeness": round(self.citation_completeness, 4),
                    "groundedness": round(self.groundedness, 4),
                    "unsupported_claim_rate": round(self.unsupported_claim_rate, 4),
                },
                "results": [asdict(r) for r in self.results],
            },
            indent=2,
            default=str,
        )


def _score(case: EvalCase, answer: ResearchAnswer) -> EvalResult:
    text = (answer.answer or "").lower()
    answered = answer.status is AnswerStatus.ANSWERED
    expected_answered = case.expected is ExpectedOutcome.ANSWERED

    domains = sorted({c.domain for c in answer.citations})
    return EvalResult(
        case_id=case.id,
        category=case.category,
        question=case.question,
        expected=str(case.expected),
        actual=str(answer.status),
        outcome_correct=(answered == expected_answered),
        citations=len(answer.citations),
        citations_verified=sum(1 for c in answer.citations if c.verified),
        claims_total=len(answer.claims),
        claims_supported=sum(1 for c in answer.claims if c.status is ClaimStatus.SUPPORTED),
        mentions_missing=[m for m in case.must_mention if answered and m.lower() not in text],
        forbidden_present=[m for m in case.must_not_mention if m.lower() in text],
        domains_cited=domains,
        expected_domains_missing=[d for d in case.expect_domains if d not in domains],
        latency_ms=answer.latency_ms or 0.0,
    )


def run_evaluation(assistant, cases=EVAL_CASES, *, verbose: bool = True) -> EvalSummary:
    summary = EvalSummary(started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    start = time.perf_counter()

    for case in cases:
        try:
            answer = assistant.answer(case.question, research_mode=(case.category == "conflict"))
            result = _score(case, answer)
        except Exception as exc:
            result = EvalResult(
                case_id=case.id, category=case.category, question=case.question,
                expected=str(case.expected), actual="error", outcome_correct=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        summary.results.append(result)
        if verbose:
            mark = "PASS" if result.passed else "FAIL"
            print(f"  [{mark}] {case.id:3} {case.question[:56]}")

    summary.duration_s = time.perf_counter() - start
    return summary


def save_report(summary: EvalSummary, directory: str | Path = "data/eval-runs") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"eval-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(summary.to_json(), encoding="utf-8")
    return path
