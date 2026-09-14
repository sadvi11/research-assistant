"""The evaluation harness must itself be trustworthy.

If the scorer can be satisfied by a system that refuses everything, or by one
that answers everything, it is not measuring anything.
"""
from __future__ import annotations

import pytest

from evaluation.dataset import EVAL_CASES, ExpectedOutcome, load_eval_corpus
from evaluation.runner import EvalResult, EvalSummary, run_evaluation
from schemas import AnswerStatus


def test_dataset_meets_the_specified_size_and_coverage():
    assert len(EVAL_CASES) >= 20
    categories = {c.category for c in EVAL_CASES}
    for required in ("answerable", "insufficient", "conflict", "multi_source", "hallucination_bait"):
        assert required in categories, f"eval set lacks a {required} category"


def test_dataset_is_balanced_between_answering_and_refusing():
    refuse = sum(1 for c in EVAL_CASES if c.expected is ExpectedOutcome.INSUFFICIENT)
    assert refuse >= len(EVAL_CASES) // 3, "too few cases where refusal is the correct answer"


def test_case_ids_are_unique():
    ids = [c.id for c in EVAL_CASES]
    assert len(ids) == len(set(ids))


def _summary(pairs: list[tuple[str, str, str]]) -> EvalSummary:
    """pairs: (category, expected, actual)."""
    s = EvalSummary()
    for i, (category, expected, actual) in enumerate(pairs):
        s.results.append(
            EvalResult(
                case_id=f"X{i}", category=category, question="q", expected=expected,
                actual=actual, outcome_correct=(expected == actual),
                citations=1 if actual == AnswerStatus.ANSWERED else 0,
                citations_verified=1 if actual == AnswerStatus.ANSWERED else 0,
                claims_total=2 if actual == AnswerStatus.ANSWERED else 0,
                claims_supported=2 if actual == AnswerStatus.ANSWERED else 0,
            )
        )
    return s


def test_scorer_punishes_a_system_that_answers_everything():
    """The dangerous failure must be visible in the headline metrics."""
    s = _summary([
        ("answerable", "answered", "answered"),
        ("insufficient", "insufficient_evidence", "answered"),
        ("hallucination_bait", "insufficient_evidence", "answered"),
    ])
    assert s.false_answer_rate == 1.0
    assert s.refusal_accuracy == 0.0


def test_scorer_punishes_a_system_that_refuses_everything():
    s = _summary([
        ("answerable", "answered", "insufficient_evidence"),
        ("answerable", "answered", "insufficient_evidence"),
        ("insufficient", "insufficient_evidence", "insufficient_evidence"),
    ])
    assert s.outcome_accuracy < 0.5
    # Refusing everything gets refusal_accuracy right, which is exactly why
    # outcome_accuracy must be reported alongside it.
    assert s.refusal_accuracy == 1.0


def test_scorer_rewards_a_correct_system():
    s = _summary([
        ("answerable", "answered", "answered"),
        ("insufficient", "insufficient_evidence", "insufficient_evidence"),
        ("hallucination_bait", "insufficient_evidence", "insufficient_evidence"),
    ])
    assert s.outcome_accuracy == 1.0
    assert s.false_answer_rate == 0.0
    assert s.groundedness == 1.0


def test_forbidden_content_counts_as_a_hallucination_incident():
    s = EvalSummary(results=[
        EvalResult("F1", "insufficient", "q", "insufficient_evidence", "answered",
                   outcome_correct=False, forbidden_present=["99.9"]),
    ])
    assert len(s.hallucination_incidents) == 1


def test_unsupported_claim_rate_is_computed_correctly():
    r = EvalResult("X", "answerable", "q", "answered", "answered", True,
                   claims_total=4, claims_supported=3)
    assert r.unsupported_claim_rate == pytest.approx(0.25)


def test_full_evaluation_runs_end_to_end(store, embedder, settings, fake_client_factory, retriever):
    """The harness must execute the real pipeline over the real dataset."""
    from assistant import ResearchAssistant
    from citations.validator import CitationValidator
    from citations.verifier import ClaimVerifier
    from ingestion.pipeline import IngestionPipeline

    load_eval_corpus(IngestionPipeline(store=store, embedder=embedder, settings=settings))
    client = fake_client_factory(
        answer_text="Amazon EKS runs and manages the Kubernetes control plane across zones.",
        citation_specs=[(0, "runs and manages the Kubernetes control plane")],
    )
    assistant = ResearchAssistant(
        retriever, client, settings=settings,
        validator=CitationValidator(), verifier=ClaimVerifier(client, settings=settings),
    )
    summary = run_evaluation(assistant, EVAL_CASES[:8], verbose=False)
    assert summary.total == 8
    assert 0.0 <= summary.outcome_accuracy <= 1.0
    assert "EVALUATION REPORT" in summary.report()
    assert summary.to_json()
