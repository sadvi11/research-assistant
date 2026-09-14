"""Application wiring. One place where concrete implementations are chosen."""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from assistant import ResearchAssistant
from config.settings import Settings, get_settings
from generation.client import ClaudeClient
from ingestion.pipeline import IngestionPipeline
from retrieval.embeddings import Embedder, get_embedder
from retrieval.hybrid import HybridRetriever
from retrieval.store import ChunkStore, InMemoryStore, PgVectorStore

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_store_and_embedder() -> tuple[ChunkStore, Embedder]:
    settings: Settings = get_settings()
    embedder = get_embedder(settings)

    backend = os.environ.get("RA_STORE", "memory").lower()
    if backend == "pgvector":
        store: ChunkStore = PgVectorStore(
            settings.database_url, dimensions=settings.embedding_dimensions
        )
        store.initialise()
        logger.info("using pgvector store")
    else:
        store = InMemoryStore()
        logger.info("using in-memory store (set RA_STORE=pgvector for persistence)")
    return store, embedder


@lru_cache(maxsize=1)
def get_assistant() -> ResearchAssistant:
    settings = get_settings()
    store, embedder = get_store_and_embedder()
    return ResearchAssistant(
        HybridRetriever(store, embedder, settings=settings),
        ClaudeClient(settings),
        settings=settings,
    )


@lru_cache(maxsize=1)
def get_ingestion() -> IngestionPipeline:
    store, embedder = get_store_and_embedder()
    return IngestionPipeline(store=store, embedder=embedder, settings=get_settings())
