"""Chunk storage behind one protocol, with two implementations.

`InMemoryStore` is the reference implementation and what the tests exercise -
it implements BM25 directly so lexical retrieval is testable with no database.
`PgVectorStore` is the production path: pgvector for dense similarity and
Postgres full-text search for the lexical half, so hybrid retrieval lives in a
single system rather than two.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from typing import Protocol, runtime_checkable

from config.sources import SourceTier
from retrieval.embeddings import cosine_similarity
from schemas import Chunk, DocumentMetadata, ScoredChunk, SourceType

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")
_BM25_K1 = 1.5
_BM25_B = 0.75

# Postgres `to_tsvector('english', ...)` strips English stopwords before
# indexing. The in-memory store must do the same, or the two backends disagree
# about what counts as a lexical match - and a question made mostly of stopwords
# ("what is the airspeed of a laden swallow") scores as though it matched the
# corpus, which is exactly the false-confidence failure this system exists to
# prevent.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren as at be because been
before being below between both but by can cannot could couldn did didn do does
doesn doing don down during each few for from further had hadn has hasn have haven
having he her here hers herself him himself his how i if in into is isn it its
itself let me more most mustn my myself no nor not of off on once only or other
ought our ours ourselves out over own same shan she should shouldn so some such
than that the their theirs them themselves then there these they this those
through to too under until up very was wasn we were weren what when where which
while who whom why with won would wouldn you your yours yourself yourselves
""".split())


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in _STOPWORDS]
    return tokens


@runtime_checkable
class ChunkStore(Protocol):
    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None: ...
    def dense_search(self, vector: list[float], limit: int, *, max_tier: SourceTier | None = None) -> list[tuple[Chunk, float]]: ...
    def lexical_search(self, query: str, limit: int, *, max_tier: SourceTier | None = None) -> list[tuple[Chunk, float]]: ...
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...
    def count(self) -> int: ...


class InMemoryStore:
    """Reference store. Deterministic, no external dependencies."""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._tokens: dict[str, Counter[str]] = {}
        self._doc_freq: Counter[str] = Counter()
        self._total_len = 0

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        if vectors is not None and len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")
        for i, chunk in enumerate(chunks):
            if chunk.chunk_id in self._chunks:
                continue  # idempotent re-ingestion
            self._chunks[chunk.chunk_id] = chunk
            if vectors is not None:
                self._vectors[chunk.chunk_id] = vectors[i]
            counts = Counter(tokenize(chunk.text))
            self._tokens[chunk.chunk_id] = counts
            self._total_len += sum(counts.values())
            for term in counts:
                self._doc_freq[term] += 1

    def _passes(self, chunk: Chunk, max_tier: SourceTier | None) -> bool:
        return max_tier is None or chunk.tier <= max_tier

    def dense_search(self, vector, limit, *, max_tier=None):
        scored = [
            (chunk, cosine_similarity(vector, self._vectors[cid]))
            for cid, chunk in self._chunks.items()
            if cid in self._vectors and self._passes(chunk, max_tier)
        ]
        scored.sort(key=lambda p: p[1], reverse=True)
        return [(c, s) for c, s in scored[:limit] if s > 0.0]

    def lexical_search(self, query, limit, *, max_tier=None):
        """Okapi BM25 over the stored chunks."""
        # Stopword-only queries carry no lexical signal; treat them as no match
        # rather than letting "is/the/of" manufacture a score.
        terms = tokenize(query, drop_stopwords=True)
        if not terms or not self._chunks:
            return []
        n_docs = len(self._chunks)
        avg_len = (self._total_len / n_docs) if n_docs else 0.0

        results: list[tuple[Chunk, float]] = []
        for cid, chunk in self._chunks.items():
            if not self._passes(chunk, max_tier):
                continue
            counts = self._tokens[cid]
            doc_len = sum(counts.values()) or 1
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                df = self._doc_freq.get(term, 0) or 1
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * doc_len / (avg_len or 1))
                score += idf * (tf * (_BM25_K1 + 1)) / denom
            if score > 0.0:
                results.append((chunk, score))

        results.sort(key=lambda p: p[1], reverse=True)
        return results[:limit]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def count(self) -> int:
        return len(self._chunks)

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks.values())


_SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL,
    text             TEXT NOT NULL,
    ordinal          INTEGER NOT NULL,
    char_start       INTEGER NOT NULL,
    char_end         INTEGER NOT NULL,
    section_path     JSONB NOT NULL DEFAULT '[]'::jsonb,
    title            TEXT NOT NULL,
    url              TEXT NOT NULL,
    domain           TEXT NOT NULL,
    source_type      TEXT NOT NULL,
    tier             SMALLINT NOT NULL,
    publication_date TIMESTAMPTZ,
    author           TEXT,
    retrieved_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    content_sha256   TEXT,
    embedding        vector(%(dims)s),
    tsv              tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tsv_idx    ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_tier_idx   ON chunks (tier);
CREATE INDEX IF NOT EXISTS chunks_doc_idx    ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_vec_idx    ON chunks
    USING hnsw (embedding vector_cosine_ops);
"""


class PgVectorStore:
    """Postgres + pgvector. Dense and lexical retrieval in one database.

    Ranking happens in the database - the `<=>` cosine-distance operator uses
    the HNSW index, and `ts_rank_cd` uses the GIN index. Only the top-k rows
    cross the network, rather than every embedding being pulled into Python.
    """

    def __init__(self, dsn: str, *, dimensions: int = 384) -> None:
        self.dsn = dsn
        self.dimensions = dimensions
        self._conn = None

    def _connect(self):
        if self._conn is None or getattr(self._conn, "closed", 1):
            try:
                import psycopg
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "PgVectorStore needs 'psycopg[binary]'. Install it with:\n"
                    "    pip install 'psycopg[binary]'"
                ) from exc
            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def initialise(self) -> None:
        with self._connect().cursor() as cur:
            cur.execute(_SCHEMA % {"dims": self.dimensions})
        logger.info("pgvector schema ready", extra={"dimensions": self.dimensions})

    def add_chunks(self, chunks: list[Chunk], vectors: list[list[float]] | None = None) -> None:
        if not chunks:
            return
        if vectors is None:
            raise ValueError("PgVectorStore requires embeddings; pass vectors")
        if len(vectors) != len(chunks):
            raise ValueError("vectors and chunks must be the same length")

        rows = [
            (
                c.chunk_id, c.document_id, c.text, c.ordinal, c.char_start, c.char_end,
                json.dumps(list(c.section_path)), c.metadata.title, c.metadata.url,
                c.metadata.domain, str(c.metadata.source_type), int(c.metadata.tier),
                c.metadata.publication_date, c.metadata.author, c.metadata.retrieved_at,
                c.metadata.content_sha256, "[" + ",".join(f"{v:.8f}" for v in vectors[i]) + "]",
            )
            for i, c in enumerate(chunks)
        ]
        with self._connect().cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (chunk_id, document_id, text, ordinal, char_start,
                       char_end, section_path, title, url, domain, source_type, tier,
                       publication_date, author, retrieved_at, content_sha256, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                   ON CONFLICT (chunk_id) DO NOTHING""",
                rows,
            )

    @staticmethod
    def _row_to_chunk(row) -> Chunk:
        (chunk_id, document_id, text, ordinal, char_start, char_end, section_path,
         title, url, domain, source_type, tier, pub_date, author, retrieved_at,
         sha, score) = row
        return Chunk(
            chunk_id=chunk_id, document_id=document_id, text=text, ordinal=ordinal,
            char_start=char_start, char_end=char_end,
            section_path=tuple(section_path or ()),
            metadata=DocumentMetadata(
                document_id=document_id, title=title, url=url, domain=domain,
                source_type=SourceType(source_type), tier=SourceTier(tier),
                publication_date=pub_date, author=author, retrieved_at=retrieved_at,
                content_sha256=sha,
            ),
        )

    _SELECT = """chunk_id, document_id, text, ordinal, char_start, char_end,
                 section_path, title, url, domain, source_type, tier,
                 publication_date, author, retrieved_at, content_sha256"""

    def dense_search(self, vector, limit, *, max_tier=None):
        literal = "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
        tier_clause = "WHERE tier <= %s" if max_tier is not None else ""
        params: list = [literal]
        if max_tier is not None:
            params.append(int(max_tier))
        params.extend([literal, limit])
        with self._connect().cursor() as cur:
            cur.execute(
                f"""SELECT {self._SELECT}, 1 - (embedding <=> %s::vector) AS score
                    FROM chunks {tier_clause}
                    ORDER BY embedding <=> %s::vector LIMIT %s""",
                params,
            )
            return [(self._row_to_chunk(r), float(r[-1])) for r in cur.fetchall()]

    def lexical_search(self, query, limit, *, max_tier=None):
        tier_clause = "AND tier <= %s" if max_tier is not None else ""
        params: list = [query, query]
        if max_tier is not None:
            params.append(int(max_tier))
        params.append(limit)
        with self._connect().cursor() as cur:
            cur.execute(
                f"""SELECT {self._SELECT},
                           ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS score
                    FROM chunks
                    WHERE tsv @@ websearch_to_tsquery('english', %s) {tier_clause}
                    ORDER BY score DESC LIMIT %s""",
                params,
            )
            return [(self._row_to_chunk(r), float(r[-1])) for r in cur.fetchall()]

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._connect().cursor() as cur:
            cur.execute(f"SELECT {self._SELECT}, 0.0 FROM chunks WHERE chunk_id = %s", (chunk_id,))
            row = cur.fetchone()
        return self._row_to_chunk(row) if row else None

    def count(self) -> int:
        with self._connect().cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks")
            return int(cur.fetchone()[0])
