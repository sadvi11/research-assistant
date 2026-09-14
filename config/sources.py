"""Source trust model.

Tier 1 is preferred whenever it is available. Lower tiers are usable but are
labelled as such in the answer, and a Tier 4 source is never presented as
authoritative on its own.

The allowlist is configurable: RA_TRUSTED_DOMAINS_FILE points at a JSON file
that replaces or extends these defaults.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from urllib.parse import urlparse


class SourceTier(IntEnum):
    """Lower value = more authoritative."""

    OFFICIAL = 1      # official docs, government, standards bodies, peer-reviewed
    ACADEMIC = 2      # universities, research institutions, recognised technical orgs
    SECONDARY = 3     # reputable secondary reporting
    UNVETTED = 4      # blogs, forums, social media

    @property
    def label(self) -> str:
        return {
            SourceTier.OFFICIAL: "Official / primary",
            SourceTier.ACADEMIC: "Academic / research institution",
            SourceTier.SECONDARY: "Reputable secondary",
            SourceTier.UNVETTED: "Unvetted",
        }[self]


# Domain suffixes are matched against the registrable domain and any parent.
_DEFAULT_TIER_1: tuple[str, ...] = (
    # Official vendor / project documentation
    "docs.anthropic.com", "platform.claude.com", "anthropic.com",
    "docs.aws.amazon.com", "aws.amazon.com",
    "kubernetes.io", "k8s.io",
    "docs.python.org", "python.org", "peps.python.org",
    "docs.github.com", "github.blog",
    "cncf.io", "opencontainers.org",
    "cloud.google.com", "learn.microsoft.com", "docs.microsoft.com",
    "docs.docker.com", "developer.hashicorp.com",
    "postgresql.org", "nginx.org", "redis.io",
    # Standards bodies
    "ietf.org", "rfc-editor.org", "w3.org", "iso.org", "nist.gov",
    "iana.org", "unicode.org", "ecma-international.org",
    # Government (national and sub-national suffixes)
    "gov", "gov.uk", "gc.ca", "canada.ca", "europa.eu", "who.int",
    "gov.au", "govt.nz", "gouv.fr", "bund.de", "gov.in", "gov.sg",
    "alberta.ca", "ontario.ca", "quebec.ca", "gov.bc.ca",
    # Peer-reviewed / preprint with editorial process
    "nature.com", "science.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "doi.org", "acm.org", "ieee.org",
)

_DEFAULT_TIER_2: tuple[str, ...] = (
    "edu", "ac.uk", "arxiv.org", "mit.edu", "stanford.edu", "berkeley.edu",
    "ox.ac.uk", "cam.ac.uk", "utoronto.ca", "ubc.ca",
    "linuxfoundation.org", "apache.org", "mozilla.org", "developer.mozilla.org",
)

_DEFAULT_TIER_3: tuple[str, ...] = (
    "wikipedia.org", "stackoverflow.com", "theregister.com", "arstechnica.com",
)


@dataclass(frozen=True)
class TrustPolicy:
    tier_1: frozenset[str]
    tier_2: frozenset[str]
    tier_3: frozenset[str]

    def tier_for_domain(self, domain: str) -> SourceTier:
        """Classify a domain. Anything unrecognised is Tier 4, never Tier 1."""
        domain = (domain or "").strip().lower().removeprefix("www.")
        if not domain:
            return SourceTier.UNVETTED
        for suffixes, tier in (
            (self.tier_1, SourceTier.OFFICIAL),
            (self.tier_2, SourceTier.ACADEMIC),
            (self.tier_3, SourceTier.SECONDARY),
        ):
            if any(domain == s or domain.endswith("." + s) for s in suffixes):
                return tier
        return SourceTier.UNVETTED

    def tier_for_url(self, url: str) -> SourceTier:
        return self.tier_for_domain(domain_of(url))

    def is_trusted(self, url: str, max_tier: SourceTier = SourceTier.SECONDARY) -> bool:
        return self.tier_for_url(url) <= max_tier

    def allowed_web_domains(self) -> list[str]:
        """Domains passed to the web_search tool's allowed_domains parameter.

        Only real hostnames are valid there - bare TLD-style entries such as
        "gov" or "edu" are rejected by the API, so they are filtered out.
        """
        out: list[str] = []
        for d in sorted(self.tier_1 | self.tier_2):
            if "." in d and not d.startswith("."):
                out.append(d)
        return out


def domain_of(url: str) -> str:
    """Registrable host for a URL. Returns '' for anything unparseable."""
    try:
        parsed = urlparse(url if "://" in url else f"https://{url}")
    except ValueError:
        return ""
    return (parsed.hostname or "").lower().removeprefix("www.")


@lru_cache(maxsize=1)
def get_trust_policy() -> TrustPolicy:
    t1, t2, t3 = set(_DEFAULT_TIER_1), set(_DEFAULT_TIER_2), set(_DEFAULT_TIER_3)

    path = os.environ.get("RA_TRUSTED_DOMAINS_FILE")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            extra = json.load(fh)
        if extra.get("replace"):
            t1, t2, t3 = set(), set(), set()
        t1 |= {d.lower() for d in extra.get("tier_1", [])}
        t2 |= {d.lower() for d in extra.get("tier_2", [])}
        t3 |= {d.lower() for d in extra.get("tier_3", [])}

    for key, bucket in (("RA_TIER1_DOMAINS", t1), ("RA_TIER2_DOMAINS", t2)):
        raw = os.environ.get(key, "")
        bucket |= {d.strip().lower() for d in raw.split(",") if d.strip()}

    return TrustPolicy(frozenset(t1), frozenset(t2), frozenset(t3))
