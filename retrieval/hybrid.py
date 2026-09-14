"""Hybrid retrieval: dense + lexical, fused, deduplicated, tier-aware.

Dense search finds paraphrase; lexical search finds exact strings - error
codes, product names, section numbers - that embeddings handle badly. Real
research corpora contain both kinds of query, so both run and their ranked
lists are fused.

Fusion uses Reciprocal Rank Fusion rather than score addition, because scores
from two different systems are not on a comparable scale. RRF only uses rank
position, which is exactly the property that makes it safe to combine them.
"""
from __future__ import annotations

import logging
from collections import defaultdict

from config.settings import Settings, get_settings
from config.sources import SourceTier
from retrieval.embeddings import Embedder
from retrieval.store import ChunkStore
from schemas import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[tuple[Chunk, float]]],
    *,
    k: int = 60,
) -> dict[str, dict]:
    """Fuse ranked lists by reciprocal rank.

    Returns chunk_id -> {chunk, fused, dense_score, lexical_score, ranks...}.
    A chunk appearing in both lists accumulates contributions from each, which
    is what makes hybrid retrieval beat either half alone.
    """
    merged: dict[str, dict] = {}
    for source, results in ranked_lists.items():
        for rank, (chunk, score) in enumerate(results, start=1):
            entry = merged.setdefault(
                chunk.chunk_id,
                {"chunk": chunk, "fused": 0.0, "dense_score": None,
                 "lexical_score": None, "dense_rank": None, "lexical_rank": None},
            )
            entry["fused"] += 1.0 / (k + rank)
            entry[f"{source}_score"] = score
            entry[f"{source}_rank"] = rank
    return merged


class HybridRetriever:
    """Runs both retrieval arms, fuses, filters by trust, and diversifies."""

    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings or get_settings()

    def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        candidates: int | None = None,
        max_tier: SourceTier | None = None,
        prefer_tier_1: bool = True,
    ) -> list[ScoredChunk]:
        """Retrieve and rank chunks for `query`.

        Empty result is a legitimate outcome and must not be papered over - the
        caller turns it into an insufficient-evidence response.
        """
        if not query or not query.strip():
            return []

        limit = limit or self.settings.final_context_chunks
        candidates = candidates or self.settings.retrieve_candidates

        dense: list[tuple[Chunk, float]] = []
        try:
            vector = self.embedder.embed_query(query)
            dense = self.store.dense_search(vector, candidates, max_tier=max_tier)
        except Exception:
            # Lexical-only degradation is far better than failing the query.
            logger.exception("dense retrieval failed; continuing with lexical only")

        lexical: list[tuple[Chunk, float]] = []
        try:
            lexical = self.store.lexical_search(query, candidates, max_tier=max_tier)
        except Exception:
            logger.exception("lexical retrieval failed; continuing with dense only")

        if not dense and not lexical:
            return []

        merged = reciprocal_rank_fusion(
            {"dense": dense, "lexical": lexical}, k=self.settings.rrf_k
        )
        scored = [
            ScoredChunk(
                chunk=e["chunk"],
                score=e["fused"],
                dense_score=e["dense_score"],
                lexical_score=e["lexical_score"],
                dense_rank=e["dense_rank"],
                lexical_rank=e["lexical_rank"],
            )
            for e in merged.values()
        ]

        scored = self._rerank(query, scored, prefer_tier_1=prefer_tier_1)
        scored = self._diversify(scored, limit)

        logger.info(
            "retrieval complete",
            extra={"query_len": len(query), "dense": len(dense), "lexical": len(lexical),
                   "fused": len(merged), "returned": len(scored)},
        )
        return scored

    # -- ranking -----------------------------------------------------------

    def _rerank(
        self, query: str, scored: list[ScoredChunk], *, prefer_tier_1: bool
    ) -> list[ScoredChunk]:
        """Authority-aware reranking.

        Relevance dominates; trust tier breaks ties. A Tier 4 blog does not
        outrank official documentation on an equal-relevance hit, but a highly
        relevant Tier 3 chunk still beats an irrelevant Tier 1 one - otherwise
        the system would answer confidently from the wrong page.
        """
        if not scored:
            return []

        best = max(s.score for s in scored) or 1.0
        reranked: list[ScoredChunk] = []
        for item in scored:
            relative = item.score / best
            boost = 1.0
            if prefer_tier_1:
                boost = {
                    SourceTier.OFFICIAL: 1.20,
                    SourceTier.ACADEMIC: 1.10,
                    SourceTier.SECONDARY: 1.00,
                    SourceTier.UNVETTED: 0.80,
                }[item.chunk.tier]
            # Both arms agreeing is real signal, not a scoring artefact.
            if item.retrieval_method == "hybrid":
                boost *= 1.10
            reranked.append(item.model_copy(update={"rerank_score": relative * boost}))

        # Normalise after boosting rather than clamping, so distinct scores stay
        # distinct - clamping collapses the top of the list into artificial ties
        # and the relevance figure shown to the user stops being informative.
        top = max((s.rerank_score or 0.0) for s in reranked) or 1.0
        reranked = [s.model_copy(update={"rerank_score": (s.rerank_score or 0.0) / top})
                    for s in reranked]
        reranked.sort(key=lambda s: (s.rerank_score or 0.0), reverse=True)
        return [s for s in reranked if (s.rerank_score or 0.0) >= self.settings.min_relevance_score]

    def _diversify(self, scored: list[ScoredChunk], limit: int) -> list[ScoredChunk]:
        """Stop one document from monopolising the context window.

        Without this, a long official page can supply every slot and the answer
        rests on a single source - which defeats the requirement to prefer more
        than one independent source for important claims.
        """
        per_doc: defaultdict[str, int] = defaultdict(int)
        kept: list[ScoredChunk] = []
        overflow: list[ScoredChunk] = []

        for item in scored:
            doc = item.chunk.document_id
            if per_doc[doc] < self.settings.max_chunks_per_document:
                per_doc[doc] += 1
                kept.append(item)
            else:
                overflow.append(item)
            if len(kept) >= limit:
                break

        # Backfill only if diversification left us short of the budget.
        if len(kept) < limit:
            kept.extend(overflow[: limit - len(kept)])
        return kept[:limit]
