"""Unit tests for src.retrieval (Milestone 3: dense cosine retrieval).

The vector-store client and embedding backend are faked/monkeypatched, so
these tests verify OUR orchestration logic (what gets searched, with what
parameters, how hits get parsed) without needing a live Zilliz cluster or a
real embedding model.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import AppConfig
from src.retrieval import OUTPUT_FIELDS, RetrievedChunk, dense_search, retrieve, retrieve_dense

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "pinnacle_capstone_data"


# --------------------------------------------------------------------------- #
# RetrievedChunk.from_hit
# --------------------------------------------------------------------------- #


def test_from_hit_parses_nested_entity_shape():
    hit = {
        "id": "attn-p003-c000-abc123",
        "distance": 0.87,
        "entity": {
            "doc_id": "attention_paper",
            "title": "Attention Is All You Need",
            "filename": "attention_paper.pdf",
            "page_number": 3,
            "text": "The Transformer follows this overall architecture...",
            "embedding_model": "BAAI/bge-small-en-v1.5",
        },
    }
    chunk = RetrievedChunk.from_hit(hit)
    assert chunk.chunk_id == "attn-p003-c000-abc123"
    assert chunk.score == pytest.approx(0.87)
    assert chunk.title == "Attention Is All You Need"
    assert chunk.page_number == 3
    assert "Transformer" in chunk.text


def test_from_hit_falls_back_to_flat_dict_without_entity_key():
    hit = {"chunk_id": "x1", "distance": 0.5, "title": "T", "filename": "f.pdf", "page_number": 2, "text": "hi"}
    chunk = RetrievedChunk.from_hit(hit)
    assert chunk.chunk_id == "x1"
    assert chunk.title == "T"
    assert chunk.page_number == 2


def test_to_dict_round_trips_expected_keys():
    hit = {"id": "x1", "distance": 0.5, "entity": {"title": "T", "filename": "f.pdf", "page_number": 1, "text": "hi"}}
    chunk = RetrievedChunk.from_hit(hit)
    d = chunk.to_dict()
    assert d["chunk_id"] == "x1"
    assert d["score"] == 0.5
    assert d["title"] == "T"


# --------------------------------------------------------------------------- #
# dense_search (fake client)
# --------------------------------------------------------------------------- #


class _FakeSearchClient:
    def __init__(self, hits):
        self._hits = hits
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [self._hits]  # one result list per query vector; we send exactly one


def test_dense_search_passes_expected_parameters():
    client = _FakeSearchClient(hits=[])
    dense_search(client, "my_collection", [0.1, 0.2, 0.3], top_k=5)

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["collection_name"] == "my_collection"
    assert call["data"] == [[0.1, 0.2, 0.3]]
    assert call["anns_field"] == "vector"
    assert call["limit"] == 5
    assert call["search_params"] == {"metric_type": "COSINE"}
    assert call["output_fields"] == OUTPUT_FIELDS


def test_dense_search_returns_parsed_chunks_in_order():
    hits = [
        {"id": "a", "distance": 0.9, "entity": {"title": "A", "filename": "a.pdf", "page_number": 1, "text": "..."}},
        {"id": "b", "distance": 0.8, "entity": {"title": "B", "filename": "b.pdf", "page_number": 2, "text": "..."}},
    ]
    client = _FakeSearchClient(hits=hits)
    results = dense_search(client, "my_collection", [0.0, 0.0], top_k=2)
    assert [r.chunk_id for r in results] == ["a", "b"]
    assert results[0].score == pytest.approx(0.9)


def test_dense_search_handles_empty_results():
    client = _FakeSearchClient(hits=[])
    results = dense_search(client, "my_collection", [0.0], top_k=3)
    assert results == []


# --------------------------------------------------------------------------- #
# retrieve_dense / retrieve (wiring, with backend + client monkeypatched)
# --------------------------------------------------------------------------- #


class _FakeBackend:
    def __init__(self, name):
        self.name = name

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


def test_retrieve_dense_wires_backend_collection_and_client(monkeypatch):
    recorded = {}

    def fake_get_backend(kind, config):
        recorded["kind"] = kind
        return _FakeBackend(f"model-for-{kind}")

    def fake_collection_name_for(prefix, model_name):
        recorded["collection_args"] = (prefix, model_name)
        return "resolved_collection"

    def fake_get_client(config):
        return _FakeSearchClient(hits=[{"id": "z", "distance": 0.42, "entity": {"title": "Z", "page_number": 1, "text": "t", "filename": "z.pdf"}}])

    monkeypatch.setattr("src.retrieval.get_backend", fake_get_backend)
    monkeypatch.setattr("src.retrieval.collection_name_for", fake_collection_name_for)
    monkeypatch.setattr("src.retrieval.get_client", fake_get_client)

    cfg = AppConfig(dataset_dir=DATASET_DIR, zilliz_collection_prefix="res-bot", top_k_dense=5)
    results = retrieve_dense("what is attention?", "oss", cfg)

    assert recorded["kind"] == "oss"
    assert recorded["collection_args"] == ("res-bot", "model-for-oss")
    assert len(results) == 1
    assert results[0].chunk_id == "z"


def test_retrieve_dense_respects_explicit_top_k_override(monkeypatch):
    captured = {}

    class _CapturingClient(_FakeSearchClient):
        def search(self, **kwargs):
            captured["limit"] = kwargs["limit"]
            return super().search(**kwargs)

    monkeypatch.setattr("src.retrieval.get_backend", lambda kind, config: _FakeBackend("m"))
    monkeypatch.setattr("src.retrieval.collection_name_for", lambda prefix, model: "c")
    monkeypatch.setattr("src.retrieval.get_client", lambda config: _CapturingClient(hits=[]))

    cfg = AppConfig(dataset_dir=DATASET_DIR, top_k_dense=5)
    retrieve_dense("q", "openai", cfg, top_k=9)
    assert captured["limit"] == 9


def test_retrieve_dispatches_dense_oss_and_dense_openai(monkeypatch):
    seen_kinds = []

    def fake_retrieve_dense(query, embedding_kind, config, top_k=None):
        seen_kinds.append(embedding_kind)
        return []

    monkeypatch.setattr("src.retrieval.retrieve_dense", fake_retrieve_dense)

    cfg = AppConfig(dataset_dir=DATASET_DIR)
    retrieve("q", mode="dense_oss", config=cfg)
    retrieve("q", mode="dense_openai", config=cfg)
    assert seen_kinds == ["oss", "openai"]


def test_retrieve_rejects_unimplemented_mode():
    cfg = AppConfig(dataset_dir=DATASET_DIR)
    with pytest.raises(NotImplementedError, match="hybrid"):
        retrieve("q", mode="hybrid_reranked", config=cfg)
