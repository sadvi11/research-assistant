"""Trust model: tier classification and filtering."""
from __future__ import annotations

import pytest

from config.sources import SourceTier, TrustPolicy, domain_of, get_trust_policy
from ingestion.pipeline import IngestionPipeline, classify_source_type
from schemas import SourceType


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://docs.anthropic.com/en/docs/x", SourceTier.OFFICIAL),
        ("https://docs.aws.amazon.com/eks/", SourceTier.OFFICIAL),
        ("https://kubernetes.io/docs/", SourceTier.OFFICIAL),
        ("https://www.nature.com/articles/1", SourceTier.OFFICIAL),
        ("https://cra.gc.ca/forms", SourceTier.OFFICIAL),
        ("https://arxiv.org/abs/2401.1", SourceTier.ACADEMIC),
        ("https://cs.stanford.edu/page", SourceTier.ACADEMIC),
        ("https://en.wikipedia.org/wiki/X", SourceTier.SECONDARY),
        ("https://random-seo-farm.example/post", SourceTier.UNVETTED),
        ("https://medium.com/@someone/hot-take", SourceTier.UNVETTED),
        ("not a url at all", SourceTier.UNVETTED),
        ("", SourceTier.UNVETTED),
    ],
)
def test_tier_classification(url, expected):
    assert get_trust_policy().tier_for_url(url) is expected


def test_subdomains_inherit_parent_tier():
    policy = get_trust_policy()
    assert policy.tier_for_url("https://deep.sub.kubernetes.io/x") is SourceTier.OFFICIAL


def test_lookalike_domain_is_not_trusted():
    """docs.aws.amazon.com.evil.example must NOT inherit AWS trust."""
    policy = get_trust_policy()
    assert policy.tier_for_url("https://docs.aws.amazon.com.evil.example/x") is SourceTier.UNVETTED


def test_unknown_domains_default_to_untrusted_not_trusted():
    policy = TrustPolicy(frozenset(), frozenset(), frozenset())
    assert policy.tier_for_domain("anything.example") is SourceTier.UNVETTED


def test_allowed_web_domains_excludes_bare_tlds():
    """Bare suffixes like 'gov' are rejected by the web tool API."""
    for domain in get_trust_policy().allowed_web_domains():
        assert "." in domain
        assert not domain.startswith(".")


def test_ingestion_rejects_sources_below_required_tier(settings):
    pipeline = IngestionPipeline(settings=settings)
    result = pipeline.ingest_text(
        "# Post\n\nSome claim about Kubernetes internals.",
        url="https://seo-farm.example/post", suffix=".md",
        min_tier=SourceTier.SECONDARY,
    )
    assert result.documents_ingested == 0
    assert result.skipped and "tier" in result.skipped[0][1].lower()


def test_ingestion_accepts_trusted_source(settings):
    pipeline = IngestionPipeline(settings=settings)
    result = pipeline.ingest_text(
        "# Docs\n\nAmazon EKS manages the control plane for you across zones.",
        url="https://docs.aws.amazon.com/eks/x.html", suffix=".md",
        min_tier=SourceTier.SECONDARY,
    )
    assert result.documents_ingested == 1 and result.chunks_created >= 1


def test_source_type_classification():
    assert classify_source_type("rfc-editor.org", SourceTier.OFFICIAL) is SourceType.STANDARD
    assert classify_source_type("nature.com", SourceTier.OFFICIAL) is SourceType.PEER_REVIEWED
    assert classify_source_type("cra.gc.ca", SourceTier.OFFICIAL) is SourceType.GOVERNMENT
    assert classify_source_type("docs.aws.amazon.com", SourceTier.OFFICIAL) is SourceType.OFFICIAL_DOCS


def test_domain_of_handles_malformed_input():
    assert domain_of("https://Example.COM/x") == "example.com"
    assert domain_of("www.example.com") == "example.com"
    assert domain_of("") == ""
