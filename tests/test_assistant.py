"""End-to-end pipeline behaviour: the gates that withhold answers.

Each test drives the real pipeline with a scripted model and asserts the system
refuses when it should. The bar throughout is: silence beats a guess.
"""
from __future__ import annotations

import pytest

from generation.client import GenerationError
from schemas import AnswerStatus, INSUFFICIENT_EVIDENCE_MESSAGE


def test_grounded_question_is_answered_with_verified_citations(assistant_factory, fake_client_factory):
    client = fake_client_factory(
        answer_text="Amazon EKS runs and manages the Kubernetes control plane across multiple availability zones.",
        citation_specs=[(0, "runs and manages the Kubernetes control plane")],
    )
    result = assistant_factory(client).answer("Who manages the EKS control plane?")
    assert result.status is AnswerStatus.ANSWERED
    assert result.is_grounded
    assert result.citations and all(c.verified for c in result.citations)
    assert result.evidence.chunks_retrieved > 0


def test_question_with_no_matching_corpus_refuses(assistant_factory, fake_client_factory):
    client = fake_client_factory()
    result = assistant_factory(client).answer(
        "What was the 1923 Peruvian anchovy harvest tonnage?"
    )
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert result.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert not client.calls, "model was called despite there being no usable evidence"


def test_model_declaring_insufficient_evidence_is_honoured(assistant_factory, fake_client_factory):
    client = fake_client_factory(
        answer_text="INSUFFICIENT_EVIDENCE\nThe documents do not cover pricing.",
        citation_specs=[],
    )
    result = assistant_factory(client).answer("How much does the EKS control plane cost?")
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert any("pricing" in n.lower() for n in result.uncertainty_notes)


def test_answer_whose_citations_are_all_fabricated_is_withheld(assistant_factory, fake_client_factory):
    """Every citation fails verification, so the answer cannot be shown as grounded."""
    client = fake_client_factory(
        answer_text="Amazon EKS provides a 99.999 percent uptime guarantee with cash refunds.",
        citation_specs=[(0, "a 99.999 percent uptime guarantee with cash refunds")],
        claims=["Amazon EKS provides a 99.999 percent uptime guarantee with cash refunds."],
        claim_verdicts={"Amazon EKS provides a 99.999 percent uptime guarantee with cash refunds.": "unsupported"},
    )
    result = assistant_factory(client).answer("Does EKS guarantee uptime?")
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert not [c for c in result.citations if c.verified]


def test_unsupported_claims_below_threshold_withhold_the_answer(assistant_factory, fake_client_factory):
    client = fake_client_factory(
        answer_text="EKS manages the control plane. EKS also cures baldness and predicts the weather.",
        citation_specs=[(0, "runs and manages the Kubernetes control plane")],
        claims=[
            "EKS manages the control plane.",
            "EKS cures baldness.",
            "EKS predicts the weather.",
        ],
        claim_verdicts={
            "EKS manages the control plane.": "supported",
            "EKS cures baldness.": "unsupported",
            "EKS predicts the weather.": "unsupported",
        },
    )
    result = assistant_factory(client).answer("What does EKS do?")
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    assert any("supported" in n for n in result.uncertainty_notes)


def test_a_single_contradicted_claim_withholds_the_answer(assistant_factory, fake_client_factory):
    """Contradiction is disqualifying even if the ratio would otherwise pass."""
    claims = [
        f"The Horizontal Pod Autoscaler adds Pod replicas, point {i}." for i in range(9)
    ]
    claims.append("Node scaling should always precede pod scaling.")
    verdicts = {c: "supported" for c in claims[:9]}
    verdicts[claims[9]] = "contradicted"

    client = fake_client_factory(
        answer_text="A long answer.", citation_specs=[(0, "runs and manages the Kubernetes control plane")],
        claims=claims, claim_verdicts=verdicts,
    )
    result = assistant_factory(client).answer(
        "Horizontal Pod Autoscaler replicas versus node autoscaling machines"
    )
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE
    # It must be withheld for the contradiction specifically, not merely because
    # retrieval was weak - otherwise this test would pass without exercising the
    # claim gate at all.
    assert any("contradicted" in n.lower() for n in result.uncertainty_notes), (
        f"expected a contradiction note, got: {result.uncertainty_notes}"
    )


def test_api_failure_surfaces_as_error_not_a_fabricated_answer(assistant_factory, fake_client_factory):
    client = fake_client_factory(fail_with=GenerationError("upstream 503"))
    result = assistant_factory(client).answer("Who manages the EKS control plane?")
    assert result.status is AnswerStatus.ERROR
    assert "503" in result.answer
    assert not result.citations


def test_claim_verification_failure_is_not_treated_as_success(assistant_factory, fake_client_factory):
    """'Could not check' must never be reported as 'checked and fine'."""
    client = fake_client_factory(
        answer_text="EKS manages the control plane across zones.",
        citation_specs=[(0, "runs and manages the Kubernetes control plane")],
    )
    def boom(answer):
        raise GenerationError("extraction endpoint down")
    client.extract_claims = boom

    result = assistant_factory(client).answer("Who manages the control plane?")
    assert result.status is AnswerStatus.INSUFFICIENT_EVIDENCE


def test_tier_filter_restricts_evidence_to_official_sources(assistant_factory, fake_client_factory):
    from config.sources import SourceTier

    client = fake_client_factory(
        answer_text="Amazon EKS runs and manages the Kubernetes control plane.",
        citation_specs=[(0, "runs and manages the Kubernetes control plane")],
    )
    assistant = assistant_factory(client)
    result = assistant.answer("control plane", max_tier=SourceTier.OFFICIAL)
    if result.status is AnswerStatus.ANSWERED:
        assert all(c.tier <= SourceTier.OFFICIAL for c in result.citations)


def test_single_source_answers_carry_a_corroboration_caveat(assistant_factory, fake_client_factory):
    client = fake_client_factory(
        answer_text="Error code E4021 indicates that a GPU allocation request could not be satisfied.",
        citation_specs=[(0, "E4021")],
        claims=["Error code E4021 indicates a GPU allocation request could not be satisfied."],
        claim_verdicts={"Error code E4021 indicates a GPU allocation request could not be satisfied.": "supported"},
    )
    result = assistant_factory(client).answer("What is error E4021?")
    if result.status is AnswerStatus.ANSWERED and result.evidence.independent_sources < 2:
        assert any("single source" in n.lower() for n in result.uncertainty_notes)


def test_research_mode_decomposes_and_records_subquestions(assistant_factory, fake_client_factory):
    client = fake_client_factory(
        answer_text="Amazon EKS runs and manages the Kubernetes control plane.",
        citation_specs=[(0, "runs and manages the Kubernetes control plane")],
    )
    client._structured = lambda prompt, schema, max_tokens=2048: {
        "subquestions": ["Who runs the control plane?", "How is it patched?"]
    }
    result = assistant_factory(client).answer("Explain EKS control plane management", research_mode=True)
    assert result.subquestions == ["Who runs the control plane?", "How is it patched?"]


@pytest.mark.parametrize("bad", ["", "   ", "\x00\x07"])
def test_invalid_questions_are_rejected(assistant_factory, fake_client_factory, bad):
    from assistant import InvalidQuestionError

    with pytest.raises(InvalidQuestionError):
        assistant_factory(fake_client_factory()).answer(bad)


def test_overlong_question_is_rejected(assistant_factory, fake_client_factory):
    from assistant import InvalidQuestionError

    with pytest.raises(InvalidQuestionError):
        assistant_factory(fake_client_factory()).answer("x" * 5000)


def test_every_answer_carries_a_request_id_for_tracing(assistant_factory, fake_client_factory):
    result = assistant_factory(fake_client_factory()).answer("Who manages the control plane?")
    assert result.request_id and len(result.request_id) == 12
    assert result.latency_ms is not None and result.latency_ms >= 0
