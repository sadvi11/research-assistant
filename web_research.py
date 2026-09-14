"""Allowlist-gated web research.

Uses Anthropic's server-side web_search / web_fetch tools rather than fetching
URLs ourselves. That choice is deliberate and is the security design:

  * `allowed_domains` is enforced server-side, so the trusted-source policy is
    not something a prompt can talk its way around.
  * We never issue outbound HTTP to a model-chosen URL, which removes the
    entire SSRF surface that a self-implemented fetcher would create.

Disabled by default (RA_WEB_RESEARCH=false). Local corpus is always the
primary evidence source.
"""
from __future__ import annotations

import logging

from config.settings import Settings, get_settings
from config.sources import TrustPolicy, get_trust_policy

logger = logging.getLogger(__name__)

# Current tool versions with dynamic filtering (Opus 4.6+ / Sonnet 4.6+).
WEB_SEARCH_TYPE = "web_search_20260209"
WEB_FETCH_TYPE = "web_fetch_20260209"

# The API rejects bare TLDs, single-label names and IPs in allowed_domains,
# and caps the list length.
MAX_ALLOWED_DOMAINS = 64


def build_web_tools(
    *,
    settings: Settings | None = None,
    trust: TrustPolicy | None = None,
    extra_domains: list[str] | None = None,
) -> list[dict]:
    """Tool definitions locked to the trusted-domain allowlist.

    Returns [] when web research is disabled, which callers treat as
    "local corpus only" rather than as an error.
    """
    settings = settings or get_settings()
    trust = trust or get_trust_policy()

    if not settings.web_research_enabled:
        return []

    domains = trust.allowed_web_domains()
    if extra_domains:
        domains.extend(d.strip().lower() for d in extra_domains if "." in d)

    # Deterministic order keeps the tool block byte-stable for prompt caching.
    allowed = sorted({d for d in domains if _is_valid_domain(d)})[:MAX_ALLOWED_DOMAINS]
    if not allowed:
        logger.warning("web research enabled but the allowlist is empty; disabling")
        return []

    logger.info("web research enabled", extra={"allowed_domains": len(allowed)})
    return [
        {
            "type": WEB_SEARCH_TYPE,
            "name": "web_search",
            "max_uses": settings.web_max_uses,
            "allowed_domains": allowed,
        },
        {
            "type": WEB_FETCH_TYPE,
            "name": "web_fetch",
            "max_uses": settings.web_max_uses,
            "allowed_domains": allowed,
            # Citations on fetched pages, so web evidence is as traceable as
            # local evidence rather than arriving as an unattributed snippet.
            "citations": {"enabled": True},
        },
    ]


def _is_valid_domain(domain: str) -> bool:
    """Reject what the API will reject, before it costs us a 400."""
    if not domain or "." not in domain or domain.startswith(".") or domain.endswith("."):
        return False
    if domain.replace(".", "").isdigit():  # IP address
        return False
    if any(ch in domain for ch in " /:@"):
        return False
    labels = domain.split(".")
    return len(labels) >= 2 and all(labels)
