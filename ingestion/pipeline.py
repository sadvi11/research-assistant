"""Ingestion pipeline: load -> clean -> chunk -> classify -> embed -> store.

Trust classification happens here, at ingestion, not at query time. A chunk
carries its tier with it, so retrieval never has to re-derive trust from a URL.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import Settings, get_settings
from config.sources import SourceTier, TrustPolicy, domain_of, get_trust_policy
from ingestion.chunker import Chunker
from ingestion.loaders import LoadedDocument, SUPPORTED_SUFFIXES, load_path, load_text
from schemas import Chunk, DocumentMetadata, SourceType

logger = logging.getLogger(__name__)

_TIER_TO_SOURCE_TYPE = {
    SourceTier.OFFICIAL: SourceType.OFFICIAL_DOCS,
    SourceTier.ACADEMIC: SourceType.ACADEMIC,
    SourceTier.SECONDARY: SourceType.SECONDARY,
    SourceTier.UNVETTED: SourceType.UNKNOWN,
}

_GOV_HINTS = ("gov", "gc.ca", "canada.ca", "europa.eu", "who.int")
_STANDARDS_HINTS = ("ietf.org", "rfc-editor.org", "w3.org", "iso.org", "nist.gov", "iana.org")
_JOURNAL_HINTS = ("nature.com", "science.org", "ncbi.nlm.nih.gov", "doi.org", "acm.org", "ieee.org")


def classify_source_type(domain: str, tier: SourceTier) -> SourceType:
    d = (domain or "").lower()
    if any(d == h or d.endswith("." + h) for h in _STANDARDS_HINTS):
        return SourceType.STANDARD
    if any(d == h or d.endswith("." + h) for h in _JOURNAL_HINTS):
        return SourceType.PEER_REVIEWED
    if any(d == h or d.endswith("." + h) for h in _GOV_HINTS):
        return SourceType.GOVERNMENT
    return _TIER_TO_SOURCE_TYPE[tier]


@dataclass
class IngestionResult:
    documents_ingested: int = 0
    chunks_created: int = 0
    skipped: list[tuple[str, str]] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.documents_ingested > 0

    def summary(self) -> str:
        parts = [f"{self.documents_ingested} document(s)", f"{self.chunks_created} chunk(s)"]
        if self.skipped:
            parts.append(f"{len(self.skipped)} skipped")
        return ", ".join(parts)


def make_document_id(url: str, content_sha256: str) -> str:
    """Stable across re-ingestion of identical content at the same URL."""
    return hashlib.sha256(f"{url}|{content_sha256}".encode()).hexdigest()[:16]


class IngestionPipeline:
    def __init__(
        self,
        store=None,
        *,
        settings: Settings | None = None,
        trust: TrustPolicy | None = None,
        embedder=None,
    ) -> None:
        self.settings = settings or get_settings()
        self.trust = trust or get_trust_policy()
        self.store = store
        self.embedder = embedder
        self.chunker = Chunker(
            chunk_size_tokens=self.settings.chunk_size_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )

    # -- building blocks ---------------------------------------------------

    def build_metadata(
        self,
        loaded: LoadedDocument,
        *,
        url: str,
        title: str | None = None,
    ) -> DocumentMetadata:
        domain = domain_of(url)
        tier = self.trust.tier_for_domain(domain)
        return DocumentMetadata(
            document_id=make_document_id(url, loaded.content_sha256),
            title=(title or loaded.title or "Untitled").strip(),
            url=url,
            domain=domain,
            tier=tier,
            source_type=classify_source_type(domain, tier),
            publication_date=loaded.publication_date,
            author=loaded.author,
            content_sha256=loaded.content_sha256,
        )

    def chunk_document(self, loaded: LoadedDocument, *, url: str, title: str | None = None) -> list[Chunk]:
        metadata = self.build_metadata(loaded, url=url, title=title)
        return self.chunker.split(loaded.text, metadata)

    # -- ingestion ---------------------------------------------------------

    def ingest_text(
        self,
        raw: str,
        *,
        url: str,
        suffix: str = ".txt",
        title: str | None = None,
        min_tier: SourceTier = SourceTier.UNVETTED,
    ) -> IngestionResult:
        result = IngestionResult()
        loaded = load_text(raw, suffix=suffix, source=url)
        if loaded.is_empty:
            result.skipped.append((url, "no extractable text"))
            return result

        metadata = self.build_metadata(loaded, url=url, title=title)
        if metadata.tier > min_tier:
            result.skipped.append((url, f"tier {metadata.tier.name} below required {min_tier.name}"))
            return result

        chunks = self.chunker.split(loaded.text, metadata)
        if not chunks:
            result.skipped.append((url, "produced no chunks"))
            return result

        self._persist(chunks)
        result.documents_ingested = 1
        result.chunks_created = len(chunks)
        result.document_ids.append(metadata.document_id)
        logger.info(
            "ingested document",
            extra={"document_id": metadata.document_id, "domain": metadata.domain,
                   "tier": metadata.tier.name, "chunks": len(chunks)},
        )
        return result

    def ingest_file(
        self, path: str | Path, *, url: str | None = None, title: str | None = None
    ) -> IngestionResult:
        path = Path(path)
        result = IngestionResult()
        try:
            loaded = load_path(path)
        except Exception as exc:
            result.skipped.append((str(path), f"{type(exc).__name__}: {exc}"))
            return result

        if loaded.is_empty:
            result.skipped.append((str(path), "no extractable text"))
            return result

        metadata = self.build_metadata(loaded, url=url or path.resolve().as_uri(), title=title)
        chunks = self.chunker.split(loaded.text, metadata)
        if not chunks:
            result.skipped.append((str(path), "produced no chunks"))
            return result

        self._persist(chunks)
        result.documents_ingested = 1
        result.chunks_created = len(chunks)
        result.document_ids.append(metadata.document_id)
        return result

    def ingest_directory(self, directory: str | Path, *, url_prefix: str | None = None) -> IngestionResult:
        directory = Path(directory)
        total = IngestionResult()
        if not directory.is_dir():
            total.skipped.append((str(directory), "not a directory"))
            return total

        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            url = f"{url_prefix.rstrip('/')}/{path.relative_to(directory)}" if url_prefix else None
            one = self.ingest_file(path, url=url)
            total.documents_ingested += one.documents_ingested
            total.chunks_created += one.chunks_created
            total.skipped.extend(one.skipped)
            total.document_ids.extend(one.document_ids)
        return total

    def _persist(self, chunks: list[Chunk]) -> None:
        if self.store is None:
            return
        vectors = self.embedder.embed_documents([c.text for c in chunks]) if self.embedder else None
        self.store.add_chunks(chunks, vectors)
