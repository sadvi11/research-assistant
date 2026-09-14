"""FastAPI surface.

The API is the product boundary; the HTML page is a thin client over it. Every
field the UI renders - including the evidence inspector - comes from the JSON
response, so the guarantees are testable without a browser.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.deps import get_assistant, get_ingestion, get_store_and_embedder
from app.logging_config import configure_logging
from assistant import InvalidQuestionError, ResearchAssistant
from config.settings import get_settings
from config.sources import SourceTier, get_trust_policy
from ingestion.pipeline import IngestionPipeline
from schemas import ResearchAnswer

logger = logging.getLogger(__name__)
configure_logging()

app = FastAPI(
    title="Research Assistant",
    version="1.0.0",
    description=(
        "Evidence-grounded research. Answers are generated only from retrieved, "
        "trusted sources; every claim is verified against that evidence before "
        "the answer is returned."
    ),
)

_STATIC = Path(__file__).parent / "static"
if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


# -- rate limiting ---------------------------------------------------------

_HITS: defaultdict[str, deque[float]] = defaultdict(deque)


def _rate_limit(request: Request) -> None:
    limit = get_settings().rate_limit_per_minute
    if limit <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _HITS[client]
    while window and now - window[0] > 60.0:
        window.popleft()
    if len(window) >= limit:
        raise HTTPException(429, detail=f"Rate limit exceeded ({limit}/min). Try again shortly.")
    window.append(now)


# -- request models --------------------------------------------------------

class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    research_mode: bool = False
    max_tier: int | None = Field(default=None, ge=1, le=4)
    use_web: bool | None = None


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)
    url: str = Field(min_length=1, max_length=2000)
    title: str | None = None
    suffix: str = ".md"
    min_tier: int = Field(default=4, ge=1, le=4)


class IngestPathRequest(BaseModel):
    path: str = Field(min_length=1)
    url: str | None = None


# -- routes ----------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    store, _ = get_store_and_embedder()
    settings = get_settings()
    return {
        "status": "healthy",
        "model": settings.model,
        "corpus_chunks": store.count(),
        "web_research_enabled": settings.web_research_enabled,
        "thresholds": {
            "min_supported_claim_ratio": settings.min_supported_claim_ratio,
            "min_evidence_score_to_answer": settings.min_evidence_score_to_answer,
        },
    }


@app.get("/api/trusted-sources")
def trusted_sources() -> dict:
    policy = get_trust_policy()
    return {
        "tier_1": sorted(policy.tier_1),
        "tier_2": sorted(policy.tier_2),
        "tier_3": sorted(policy.tier_3),
        "note": "Anything not listed is treated as Tier 4 and is never authoritative on its own.",
    }


@app.post("/api/research", response_model=ResearchAnswer)
def research(
    payload: ResearchRequest,
    request: Request,
    assistant: ResearchAssistant = Depends(get_assistant),
) -> ResearchAnswer:
    _rate_limit(request)
    try:
        return assistant.answer(
            payload.question,
            research_mode=payload.research_mode,
            max_tier=SourceTier(payload.max_tier) if payload.max_tier else None,
            use_web=payload.use_web,
        )
    except InvalidQuestionError as exc:
        raise HTTPException(422, detail=str(exc)) from exc


@app.post("/api/ingest/text")
def ingest_text(
    payload: IngestTextRequest,
    request: Request,
    pipeline: IngestionPipeline = Depends(get_ingestion),
) -> dict:
    _rate_limit(request)
    result = pipeline.ingest_text(
        payload.text,
        url=payload.url,
        suffix=payload.suffix,
        title=payload.title,
        min_tier=SourceTier(payload.min_tier),
    )
    return {
        "documents_ingested": result.documents_ingested,
        "chunks_created": result.chunks_created,
        "skipped": [{"source": s, "reason": r} for s, r in result.skipped],
        "summary": result.summary(),
    }


@app.post("/api/ingest/path")
def ingest_path(
    payload: IngestPathRequest,
    request: Request,
    pipeline: IngestionPipeline = Depends(get_ingestion),
) -> dict:
    _rate_limit(request)
    path = Path(payload.path)
    result = (
        pipeline.ingest_directory(path, url_prefix=payload.url)
        if path.is_dir()
        else pipeline.ingest_file(path, url=payload.url)
    )
    return {
        "documents_ingested": result.documents_ingested,
        "chunks_created": result.chunks_created,
        "skipped": [{"source": s, "reason": r} for s, r in result.skipped],
        "summary": result.summary(),
    }


@app.get("/api/evidence/{chunk_id}")
def evidence(chunk_id: str) -> dict:
    """Evidence inspector: everything needed to audit one citation."""
    store, _ = get_store_and_embedder()
    chunk = store.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(404, detail="chunk not found")
    md = chunk.metadata
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": md.title,
        "url": md.url,
        "domain": md.domain,
        "tier": int(md.tier),
        "tier_label": md.tier.label,
        "source_type": str(md.source_type),
        "publication_date": md.publication_date,
        "author": md.author,
        "retrieved_at": md.retrieved_at,
        "section_path": list(chunk.section_path),
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "text": chunk.text,
    }


@app.get("/")
def index() -> FileResponse:
    page = _STATIC / "index.html"
    if not page.exists():
        raise HTTPException(404, detail="UI not built")
    return FileResponse(str(page))


@app.exception_handler(Exception)
def unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Never leak internals to the client; the detail goes to the logs."""
    logger.exception("unhandled error", extra={"path": request.url.path})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error. The incident has been logged."},
    )
