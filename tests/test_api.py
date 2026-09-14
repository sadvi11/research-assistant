"""HTTP surface tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app import deps, main
from app.deps import get_assistant, get_ingestion, get_store_and_embedder


@pytest.fixture
def client(monkeypatch, retriever, store, embedder, settings, fake_client_factory, assistant_factory):
    from ingestion.pipeline import IngestionPipeline

    fake = fake_client_factory(
        answer_text="Amazon EKS runs and manages the Kubernetes control plane.",
        citation_specs=[(0, "runs and manages the Kubernetes control plane")],
    )
    assistant = assistant_factory(fake)
    ingestion = IngestionPipeline(store=store, embedder=embedder, settings=settings)

    main.app.dependency_overrides[get_assistant] = lambda: assistant
    main.app.dependency_overrides[get_ingestion] = lambda: ingestion
    monkeypatch.setattr(deps, "get_store_and_embedder", lambda: (store, embedder))
    monkeypatch.setattr(main, "get_store_and_embedder", lambda: (store, embedder))
    main._HITS.clear()
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_health_reports_corpus_and_thresholds(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["corpus_chunks"] > 0
    assert 0 < body["thresholds"]["min_supported_claim_ratio"] <= 1


def test_research_returns_grounded_answer_with_citations(client):
    res = client.post("/api/research", json={"question": "Who manages the EKS control plane?"})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "answered"
    assert body["citations"]
    assert all(c["verified"] for c in body["citations"])
    assert body["citations"][0]["url"].startswith("https://")


def test_research_refuses_off_topic_question(client):
    res = client.post("/api/research", json={"question": "What is the airspeed of a laden swallow?"})
    assert res.status_code == 200
    assert res.json()["status"] == "insufficient_evidence"


def test_empty_question_is_rejected_by_validation(client):
    assert client.post("/api/research", json={"question": ""}).status_code == 422
    assert client.post("/api/research", json={}).status_code == 422


def test_overlong_question_is_rejected(client):
    assert client.post("/api/research", json={"question": "x" * 9000}).status_code == 422


def test_evidence_inspector_returns_full_provenance(client):
    body = client.post("/api/research", json={"question": "Who manages the EKS control plane?"}).json()
    chunk_id = body["citations"][0]["chunk_id"]
    ev = client.get(f"/api/evidence/{chunk_id}")
    assert ev.status_code == 200
    data = ev.json()
    for field in ("url", "domain", "tier", "tier_label", "retrieved_at", "text",
                  "chunk_id", "char_start", "char_end", "source_type"):
        assert field in data, f"evidence inspector missing {field}"


def test_evidence_inspector_404s_for_unknown_chunk(client):
    assert client.get("/api/evidence/does-not-exist").status_code == 404


def test_ingest_text_endpoint_adds_chunks(client):
    before = client.get("/health").json()["corpus_chunks"]
    res = client.post("/api/ingest/text", json={
        "text": "# New doc\n\nKubernetes namespaces isolate groups of resources within a cluster.",
        "url": "https://kubernetes.io/docs/namespaces.html", "suffix": ".md",
    })
    assert res.status_code == 200 and res.json()["documents_ingested"] == 1
    assert client.get("/health").json()["corpus_chunks"] > before


def test_ingest_rejects_untrusted_source_when_tier_required(client):
    res = client.post("/api/ingest/text", json={
        "text": "# Blog\n\nSome unverified assertion about clusters.",
        "url": "https://content-farm.example/post", "suffix": ".md", "min_tier": 2,
    })
    assert res.json()["documents_ingested"] == 0
    assert res.json()["skipped"]


def test_trusted_sources_endpoint_lists_the_allowlist(client):
    body = client.get("/api/trusted-sources").json()
    assert "docs.anthropic.com" in body["tier_1"]
    assert "docs.aws.amazon.com" in body["tier_1"]
    assert body["note"]


def test_rate_limit_returns_429(client, monkeypatch, settings):
    from config import settings as settings_mod

    tight = settings.model_copy(update={"rate_limit_per_minute": 3}) if hasattr(settings, "model_copy") else settings
    monkeypatch.setattr(main, "get_settings", lambda: type(settings)(rate_limit_per_minute=3))
    main._HITS.clear()
    codes = [
        client.post("/api/research", json={"question": "Who manages the control plane?"}).status_code
        for _ in range(6)
    ]
    assert 429 in codes, f"rate limit never triggered: {codes}"


def test_openapi_schema_is_generated(client):
    body = client.get("/openapi.json").json()
    assert "/api/research" in body["paths"]
