"""Configuration. Every knob is env-overridable; no secrets are hard-coded."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Runtime configuration.

    Thresholds are deliberately conservative: this system prefers refusing to
    answer over answering from weak evidence.
    """

    # --- Model -------------------------------------------------------------
    model: str = field(default_factory=lambda: os.environ.get("RA_MODEL", "claude-opus-5"))
    effort: str = field(default_factory=lambda: os.environ.get("RA_EFFORT", "high"))
    max_tokens: int = field(default_factory=lambda: _env_int("RA_MAX_TOKENS", 16000))

    # --- Chunking ----------------------------------------------------------
    chunk_size_tokens: int = field(default_factory=lambda: _env_int("RA_CHUNK_SIZE", 400))
    chunk_overlap_tokens: int = field(default_factory=lambda: _env_int("RA_CHUNK_OVERLAP", 60))

    # --- Retrieval ---------------------------------------------------------
    retrieve_candidates: int = field(default_factory=lambda: _env_int("RA_CANDIDATES", 40))
    final_context_chunks: int = field(default_factory=lambda: _env_int("RA_TOP_K", 8))
    max_chunks_per_document: int = field(default_factory=lambda: _env_int("RA_MAX_PER_DOC", 3))
    # Relative filter: how far below the best hit a chunk may fall.
    min_relevance_score: float = field(default_factory=lambda: _env_float("RA_MIN_RELEVANCE", 0.25))
    # Absolute floor: a chunk this semantically distant is not evidence, however
    # it ranks relative to the rest. Without this, the top hit for ANY query
    # normalises to ~1.0 and near-irrelevant chunks reach the model.
    min_absolute_relevance: float = field(
        default_factory=lambda: _env_float("RA_MIN_ABSOLUTE_RELEVANCE", 0.15)
    )
    # A chunk below the absolute floor is kept only if it is a STRONG lexical
    # hit - at least this fraction of the best lexical score for the same query.
    # Expressed as a ratio because BM25 scores are unbounded and scale with
    # corpus size and term rarity, so any fixed threshold would be brittle.
    min_lexical_ratio: float = field(
        default_factory=lambda: _env_float("RA_MIN_LEXICAL_RATIO", 0.25)
    )
    rrf_k: int = field(default_factory=lambda: _env_int("RA_RRF_K", 60))

    # --- Grounding thresholds ---------------------------------------------
    # Fraction of extracted claims that must be supported by retrieved evidence
    # before an answer is released to the user.
    min_supported_claim_ratio: float = field(
        default_factory=lambda: _env_float("RA_MIN_SUPPORTED_RATIO", 0.8)
    )
    # If the best retrieved chunk scores below this, we do not call the model.
    min_evidence_score_to_answer: float = field(
        default_factory=lambda: _env_float("RA_MIN_EVIDENCE_SCORE", 0.30)
    )
    min_evidence_chunks: int = field(default_factory=lambda: _env_int("RA_MIN_EVIDENCE_CHUNKS", 1))

    # --- Research mode -----------------------------------------------------
    max_subquestions: int = field(default_factory=lambda: _env_int("RA_MAX_SUBQUESTIONS", 5))
    prefer_independent_sources: int = field(
        default_factory=lambda: _env_int("RA_INDEPENDENT_SOURCES", 2)
    )

    # --- Web research ------------------------------------------------------
    web_research_enabled: bool = field(default_factory=lambda: _env_bool("RA_WEB_RESEARCH", False))
    web_max_uses: int = field(default_factory=lambda: _env_int("RA_WEB_MAX_USES", 5))

    # --- Storage -----------------------------------------------------------
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL", "postgresql://research:research@localhost:5432/research"
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get("RA_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    embedding_dimensions: int = field(default_factory=lambda: _env_int("RA_EMBEDDING_DIMS", 384))

    # --- Safety ------------------------------------------------------------
    max_question_chars: int = field(default_factory=lambda: _env_int("RA_MAX_QUESTION_CHARS", 2000))
    rate_limit_per_minute: int = field(default_factory=lambda: _env_int("RA_RATE_LIMIT", 20))

    def validate(self) -> None:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        if not 0.0 <= self.min_supported_claim_ratio <= 1.0:
            raise ValueError("min_supported_claim_ratio must be between 0 and 1")
        if self.final_context_chunks > self.retrieve_candidates:
            raise ValueError("final_context_chunks cannot exceed retrieve_candidates")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
