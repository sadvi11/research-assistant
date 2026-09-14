"""Evaluation dataset (spec section 17).

The set is deliberately weighted toward cases where the *correct* behaviour is
refusal. A RAG system is easy to score well on questions its corpus answers;
what distinguishes a trustworthy one is what it does with the rest.

Categories:
  answerable          - the corpus supports a confident, cited answer
  insufficient        - the corpus does not support an answer; refusal is correct
  conflict            - trusted sources disagree; the disagreement must surface
  multi_source        - a complete answer needs more than one document
  hallucination_bait  - the question presupposes a false or absent fact
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ExpectedOutcome(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT = "insufficient_evidence"


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    category: str
    expected: ExpectedOutcome
    # Substrings that a correct answer should contain (checked case-insensitively).
    must_mention: tuple[str, ...] = ()
    # Substrings whose presence indicates fabrication.
    must_not_mention: tuple[str, ...] = ()
    # Domains a correct answer should cite.
    expect_domains: tuple[str, ...] = ()
    notes: str = ""


EVAL_CORPUS: tuple[tuple[str, str], ...] = (
    (
        "https://docs.aws.amazon.com/eks/control-plane.html",
        "# EKS control plane\n\n"
        "Amazon EKS runs and manages the Kubernetes control plane across multiple "
        "availability zones. The control plane is automatically patched and is covered by "
        "a service level agreement. Customers consume the Kubernetes API and never operate "
        "control plane nodes themselves.\n\n"
        "EKS runs upstream-conformant Kubernetes, so standard manifests and tooling work "
        "without modification.",
    ),
    (
        "https://kubernetes.io/docs/concepts/autoscaling.html",
        "# Autoscaling\n\n"
        "The Horizontal Pod Autoscaler increases the number of Pod replicas when observed "
        "utilisation rises above the target. The Vertical Pod Autoscaler adjusts the CPU and "
        "memory requested by each Pod.\n\n"
        "Cluster autoscaling adds worker nodes when Pods cannot be scheduled for lack of "
        "capacity. Pod-level scaling uses capacity that already exists and is therefore "
        "attempted before node-level scaling.",
    ),
    (
        "https://kubernetes.io/docs/concepts/scheduling.html",
        "# Scheduling\n\n"
        "A Pod remains in the Pending phase while the scheduler cannot find a node that "
        "satisfies its resource requests. A Pod requesting a GPU stays Pending if no node "
        "has an unallocated GPU.\n\n"
        "Image pulling happens after a Pod has been scheduled, so a Pending Pod has not yet "
        "attempted to pull its image.",
    ),
    (
        "https://docs.aws.amazon.com/eks/gpu-sharing.html",
        "# GPU sharing on EKS\n\n"
        "Multi-Instance GPU partitions a physical GPU into isolated hardware slices, each "
        "with dedicated memory, which run in parallel and cannot affect one another.\n\n"
        "GPU time slicing allows several Pods to share one GPU by alternating access over "
        "time. Time slicing provides no memory isolation between workloads.",
    ),
    (
        "https://docs.anthropic.com/en/docs/build-with-claude/citations.html",
        "# Citations\n\n"
        "Enabling citations on a document block causes the response to carry citation "
        "objects containing the exact cited text and its location within the source "
        "document. Citations cannot be combined with structured output formats in the same "
        "request.",
    ),
    # Deliberate conflict: two trusted sources, incompatible recommendations.
    (
        "https://kubernetes.io/docs/best-practices/scaling-order.html",
        "# Scaling order guidance\n\n"
        "Kubernetes guidance recommends configuring Pod autoscaling before node autoscaling, "
        "because Pod scaling responds within seconds and consumes capacity already paid for.",
    ),
    (
        "https://learn.microsoft.com/azure/aks/scaling-order.html",
        "# Scaling order guidance\n\n"
        "For GPU workloads this guidance recommends provisioning node capacity ahead of "
        "demand rather than scaling Pods first, because GPU node provisioning can take "
        "several minutes and would otherwise stall inference traffic.",
    ),
)


def load_eval_corpus(pipeline) -> int:
    """Ingest the evaluation corpus. Returns the number of chunks created."""
    total = 0
    for url, text in EVAL_CORPUS:
        total += pipeline.ingest_text(text, url=url, suffix=".md").chunks_created
    return total


EVAL_CASES: tuple[EvalCase, ...] = (
    # -- answerable --------------------------------------------------------
    EvalCase("A1", "Who manages the Kubernetes control plane on Amazon EKS?", "answerable",
             ExpectedOutcome.ANSWERED, must_mention=("EKS",),
             expect_domains=("docs.aws.amazon.com",)),
    EvalCase("A2", "What does the Horizontal Pod Autoscaler do?", "answerable",
             ExpectedOutcome.ANSWERED, must_mention=("replica",),
             expect_domains=("kubernetes.io",)),
    EvalCase("A3", "Why would a Pod requesting a GPU stay in the Pending phase?", "answerable",
             ExpectedOutcome.ANSWERED, must_mention=("GPU",),
             expect_domains=("kubernetes.io",)),
    EvalCase("A4", "What is Multi-Instance GPU?", "answerable",
             ExpectedOutcome.ANSWERED, must_mention=("isolat",),
             expect_domains=("docs.aws.amazon.com",)),
    EvalCase("A5", "Can citations be combined with structured outputs?", "answerable",
             ExpectedOutcome.ANSWERED, must_mention=("cannot",),
             expect_domains=("docs.anthropic.com",)),
    EvalCase("A6", "Does EKS run standard upstream Kubernetes?", "answerable",
             ExpectedOutcome.ANSWERED, must_mention=("conformant",)),
    EvalCase("A7", "When does image pulling happen relative to scheduling?", "answerable",
             ExpectedOutcome.ANSWERED, must_mention=("after",)),

    # -- multi-source ------------------------------------------------------
    EvalCase("M1", "Compare Multi-Instance GPU with GPU time slicing for isolation.",
             "multi_source", ExpectedOutcome.ANSWERED,
             must_mention=("isolat", "time slicing")),
    EvalCase("M2", "Explain both Pod-level and node-level autoscaling and when each applies.",
             "multi_source", ExpectedOutcome.ANSWERED,
             must_mention=("replica", "node")),
    EvalCase("M3", "If a GPU Pod is Pending, what mechanism resolves it and why?",
             "multi_source", ExpectedOutcome.ANSWERED, must_mention=("node",)),

    # -- conflicting sources ----------------------------------------------
    EvalCase("C1", "Should node autoscaling be configured before Pod autoscaling?",
             "conflict", ExpectedOutcome.ANSWERED,
             notes="kubernetes.io and learn.microsoft.com give opposing guidance; "
                   "the disagreement must be surfaced rather than silently merged."),
    EvalCase("C2", "What is the recommended scaling order for GPU inference workloads?",
             "conflict", ExpectedOutcome.ANSWERED,
             notes="Same conflict, approached from the GPU side."),

    # -- insufficient evidence --------------------------------------------
    EvalCase("I1", "How much does an Amazon EKS cluster cost per month?", "insufficient",
             ExpectedOutcome.INSUFFICIENT, must_not_mention=("$", "per month"),
             notes="No pricing information was ingested."),
    EvalCase("I2", "What was Amazon's total revenue in 2019?", "insufficient",
             ExpectedOutcome.INSUFFICIENT, notes="Entirely outside the corpus."),
    EvalCase("I3", "Which Canadian provinces require Super Visa medical insurance?",
             "insufficient", ExpectedOutcome.INSUFFICIENT, notes="Unrelated domain."),
    EvalCase("I4", "What is the airspeed velocity of an unladen swallow?", "insufficient",
             ExpectedOutcome.INSUFFICIENT, notes="Nonsense query; must not be answered."),
    EvalCase("I5", "What SLA percentage does the EKS control plane guarantee?", "insufficient",
             ExpectedOutcome.INSUFFICIENT, must_not_mention=("99.9", "99.95", "percent"),
             notes="Corpus says an SLA exists but never states a number - the classic "
                   "invitation to invent one."),
    EvalCase("I6", "How many availability zones does the EKS control plane run across?",
             "insufficient", ExpectedOutcome.INSUFFICIENT,
             must_not_mention=("three", "two", "3", "2"),
             notes="Corpus says 'multiple' without a number."),

    # -- hallucination bait -----------------------------------------------
    EvalCase("H1", "Why does GPU time slicing provide better memory isolation than MIG?",
             "hallucination_bait", ExpectedOutcome.INSUFFICIENT,
             must_not_mention=("time slicing provides better",),
             notes="False premise - the corpus states the opposite."),
    EvalCase("H2", "What did the 2024 Kubernetes security audit conclude about EKS?",
             "hallucination_bait", ExpectedOutcome.INSUFFICIENT,
             notes="Presupposes a document that does not exist."),
    EvalCase("H3", "Confirm that EKS lets customers operate their own control plane nodes.",
             "hallucination_bait", ExpectedOutcome.INSUFFICIENT,
             notes="Leading question contradicted by the corpus."),
    EvalCase("H4", "List the five pricing tiers of Multi-Instance GPU.",
             "hallucination_bait", ExpectedOutcome.INSUFFICIENT,
             must_not_mention=("tier 1", "pricing"),
             notes="Presupposes tiers that are never mentioned."),
    EvalCase("H5", "According to Anthropic's documentation, what is the maximum number of "
                   "citations per response?", "hallucination_bait", ExpectedOutcome.INSUFFICIENT,
             notes="Plausible-sounding limit that the corpus never states."),
    EvalCase("H6", "Summarise the benchmark results comparing MIG and time slicing throughput.",
             "hallucination_bait", ExpectedOutcome.INSUFFICIENT,
             notes="No benchmarks exist in the corpus."),
)


def cases_by_category() -> dict[str, list[EvalCase]]:
    out: dict[str, list[EvalCase]] = {}
    for case in EVAL_CASES:
        out.setdefault(case.category, []).append(case)
    return out
