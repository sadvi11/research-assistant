"""Embedding backends behind a single protocol.

`LocalEmbedder` (sentence-transformers) is the production path: no API key, no
per-token cost, runs offline. `HashEmbedder` is deterministic and dependency
free - it exists so the test suite can exercise the full retrieval pipeline
without a model download or network access, matching the project's rule that
tests run at zero cost.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol, runtime_checkable

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into fixed-length vectors."""

    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def l2_normalise(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Inputs are assumed L2-normalised, so this is a dot
    product; the norms are recomputed defensively in case they are not."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder.

    Not semantically strong - it captures lexical overlap only. That is enough
    to prove the retrieval pipeline works end to end, and it never varies
    between runs, which makes retrieval tests deterministic.
    """

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        return l2_normalise(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class LocalEmbedder:
    """sentence-transformers, loaded lazily so import never blocks startup."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimensions: int = 384) -> None:
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Local embeddings need 'sentence-transformers'. Install it with:\n"
                    "    pip install sentence-transformers\n"
                    "Or set RA_EMBEDDER=hash to use the dependency-free fallback."
                ) from exc
            logger.info("loading embedding model", extra={"model": self.model_name})
            self._model = SentenceTransformer(self.model_name)
            actual = self._model.get_sentence_embedding_dimension()
            if actual != self.dimensions:
                logger.warning(
                    "embedding dimension mismatch; using model's actual value",
                    extra={"configured": self.dimensions, "actual": actual},
                )
                self.dimensions = actual
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedder(settings: Settings | None = None, *, kind: str | None = None) -> Embedder:
    """Select an embedder. RA_EMBEDDER=hash forces the offline fallback."""
    import os

    settings = settings or get_settings()
    kind = (kind or os.environ.get("RA_EMBEDDER", "local")).lower()
    if kind == "hash":
        return HashEmbedder(settings.embedding_dimensions)
    return LocalEmbedder(settings.embedding_model, settings.embedding_dimensions)
