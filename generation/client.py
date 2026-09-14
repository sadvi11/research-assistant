"""Claude client for grounded generation and claim verification.

Two distinct call shapes, and they cannot be combined:

  * `generate_grounded_answer` attaches documents with citations enabled.
  * `extract_claims` / `verify_claim` use structured outputs.

The API rejects `citations` together with `output_config.format` (HTTP 400),
so claim verification is deliberately a second pass rather than something
folded into generation. That separation is a constraint, but it is also the
right design: the verifier should not see the generator's reasoning.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings, get_settings
from generation.documents import chunks_to_document_blocks
from generation.prompts import (
    CLAIM_EXTRACTION_PROMPT,
    CLAIM_VERIFICATION_PROMPT,
    CONFLICT_DETECTION_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    build_research_prompt,
)
from schemas import ScoredChunk

logger = logging.getLogger(__name__)

INSUFFICIENT_MARKER = "INSUFFICIENT_EVIDENCE"


class GenerationError(RuntimeError):
    """Model call failed in a way the caller must surface, not paper over."""


@dataclass
class RawCitation:
    """A citation exactly as the API returned it, before our own verification."""

    document_index: int
    document_title: str | None
    cited_text: str
    start_char_index: int | None = None
    end_char_index: int | None = None


@dataclass
class GenerationResult:
    text: str
    citations: list[RawCitation] = field(default_factory=list)
    model: str | None = None
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def declared_insufficient(self) -> bool:
        return INSUFFICIENT_MARKER in self.text


_CLAIMS_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Self-contained factual assertions from the answer.",
            }
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
}

_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["supported", "unsupported", "contradicted"]},
            "reasoning": {"type": "string"},
        },
        "required": ["status", "reasoning"],
        "additionalProperties": False,
    },
}


_CONFLICT_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "conflicts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "source_a_index": {"type": "integer"},
                        "position_a": {"type": "string"},
                        "source_b_index": {"type": "integer"},
                        "position_b": {"type": "string"},
                    },
                    "required": ["topic", "source_a_index", "position_a",
                                 "source_b_index", "position_b"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["conflicts"],
        "additionalProperties": False,
    },
}


class ClaudeClient:
    """Thin wrapper over the Anthropic SDK. All model config is injectable."""

    def __init__(self, settings: Settings | None = None, *, client: Any = None) -> None:
        self.settings = settings or get_settings()
        self._client = client

    # -- plumbing ----------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise GenerationError(
                    "The Anthropic SDK is required. Install it with: pip install anthropic"
                ) from exc
            # The SDK resolves credentials itself: ANTHROPIC_API_KEY, then
            # ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile.
            self._client = anthropic.Anthropic()
        return self._client

    @staticmethod
    def credentials_available() -> bool:
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return True
        config_dir = os.path.expanduser("~/.config/anthropic")
        return os.path.isdir(config_dir) and bool(os.listdir(config_dir))

    def _call(self, **kwargs):
        """Single place where SDK exceptions are classified.

        Retryable and non-retryable failures are distinguished so callers do not
        retry a malformed request forever.
        """
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise GenerationError("anthropic SDK not installed") from exc

        try:
            return self.client.messages.create(**kwargs)
        except anthropic.NotFoundError as exc:
            raise GenerationError(f"model or endpoint not found: {exc}") from exc
        except anthropic.AuthenticationError as exc:
            raise GenerationError("authentication failed - check ANTHROPIC_API_KEY") from exc
        except anthropic.BadRequestError as exc:
            raise GenerationError(f"malformed request (not retryable): {exc}") from exc
        except anthropic.RateLimitError as exc:
            raise GenerationError(f"rate limited: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise GenerationError(f"API error {exc.status_code}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise GenerationError(f"connection error: {exc}") from exc

    # -- grounded generation ----------------------------------------------

    def generate_grounded_answer(
        self,
        question: str,
        chunks: list[ScoredChunk],
        *,
        subquestions: list[str] | None = None,
        web_tools: list[dict] | None = None,
    ) -> GenerationResult:
        """Answer `question` from `chunks`, with API-generated citations."""
        if not chunks and not web_tools:
            raise ValueError("refusing to generate with no evidence and no web tools")

        content: list[dict] = chunks_to_document_blocks(chunks)
        content.append(
            {"type": "text", "text": build_research_prompt(question, chunks, subquestions=subquestions)}
        )

        request: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            # Stable prefix first so prompt caching holds across queries.
            "system": [
                {
                    "type": "text",
                    "text": RESEARCH_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": content}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.settings.effort},
        }
        if web_tools:
            request["tools"] = web_tools

        start = time.perf_counter()
        response = self._call(**request)
        latency_ms = (time.perf_counter() - start) * 1000

        text_parts: list[str] = []
        citations: list[RawCitation] = []
        for block in response.content:
            if getattr(block, "type", None) != "text":
                continue
            text_parts.append(block.text)
            for raw in getattr(block, "citations", None) or []:
                citations.append(
                    RawCitation(
                        document_index=getattr(raw, "document_index", -1),
                        document_title=getattr(raw, "document_title", None),
                        cited_text=getattr(raw, "cited_text", "") or "",
                        start_char_index=getattr(raw, "start_char_index", None),
                        end_char_index=getattr(raw, "end_char_index", None),
                    )
                )

        usage = getattr(response, "usage", None)
        result = GenerationResult(
            text="".join(text_parts).strip(),
            citations=citations,
            model=getattr(response, "model", self.settings.model),
            stop_reason=getattr(response, "stop_reason", None),
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            latency_ms=latency_ms,
        )
        logger.info(
            "generation complete",
            extra={"documents": len(chunks), "citations": len(citations),
                   "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
                   "cache_read_tokens": result.cache_read_tokens,
                   "latency_ms": round(latency_ms, 1), "stop_reason": result.stop_reason},
        )
        return result

    # -- structured passes (no citations; incompatible with output_config) --

    def _structured(self, prompt: str, schema: dict, *, max_tokens: int = 2048) -> dict:
        import json

        response = self._call(
            model=self.settings.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            output_config={"format": schema, "effort": "low"},
        )
        for block in response.content:
            if getattr(block, "type", None) == "text":
                # Always parse; never string-match a model's serialised JSON.
                return json.loads(block.text)
        raise GenerationError("structured call returned no text block")

    def extract_claims(self, answer: str) -> list[str]:
        """Break a draft answer into individually checkable assertions."""
        if not answer.strip() or INSUFFICIENT_MARKER in answer:
            return []
        data = self._structured(CLAIM_EXTRACTION_PROMPT.format(answer=answer), _CLAIMS_SCHEMA)
        return [c.strip() for c in data.get("claims", []) if c and c.strip()]

    def verify_claim(self, claim: str, evidence: str) -> tuple[str, str]:
        """Return (status, reasoning) for one claim against one evidence bundle."""
        data = self._structured(
            CLAIM_VERIFICATION_PROMPT.format(claim=claim, evidence=evidence),
            _VERDICT_SCHEMA,
            max_tokens=1024,
        )
        return data.get("status", "unsupported"), data.get("reasoning", "")

    def detect_conflicts(self, question: str, chunks: list[ScoredChunk]) -> list[dict]:
        """Find contradictions between sources. One call, not one per pair.

        The model reports only what each source says. It is deliberately NOT
        asked which source is more authoritative - that is decided in code from
        the trust tier.
        """
        if len(chunks) < 2:
            return []
        sources = "\n\n".join(
            f"[{i}] ({c.chunk.metadata.domain}) {c.chunk.text}"
            for i, c in enumerate(chunks)
        )
        data = self._structured(
            CONFLICT_DETECTION_PROMPT.format(question=question, sources=sources),
            _CONFLICT_SCHEMA,
            max_tokens=2048,
        )
        return data.get("conflicts", [])
