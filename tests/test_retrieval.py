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
from src import retrieval
from src.retrieval import (
    OUTPUT_FIELDS,
    BM25Index,
    RetrievedChunk,
    dense_search,
    fetch_all_chunks,
    fuse_results,
    retrieve,
    retrieve_dense,
    rerank,
    retrieve_hybrid,
    retrieve_hybrid_reranked,
)

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


def test_retrieve_rejects_unknown_mode():
    cfg = AppConfig(dataset_dir=DATASET_DIR)
    with pytest.raises(ValueError, match="bogus"):
        retrieve("q", mode="bogus", config=cfg)


# --------------------------------------------------------------------------- #
# Hybrid retrieval (Milestone 4, config C)
# --------------------------------------------------------------------------- #


def _chunk(cid, text="", score=0.0):
    return RetrievedChunk(chunk_id=cid, score=score, title="T", filename="f.pdf", page_number=1, text=text, doc_id="d", embedding_model="m")


def test_rrf_scores_match_formula_and_order():
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    sparse = [_chunk("c"), _chunk("b"), _chunk("d")]
    out = fuse_results(dense, sparse, method="rrf", weight_dense=1.0, weight_sparse=1.0, top_k=4)
    expected = {
        "a": 1 / 61,
        "b": 1 / 62 + 1 / 62,
        "c": 1 / 63 + 1 / 61,  # ranked 1st by one list, 3rd by the other; beats b by convexity
        "d": 1 / 63,
    }
    assert [c.chunk_id for c in out] == ["c", "b", "a", "d"]
    for c in out:
        assert c.score == pytest.approx(expected[c.chunk_id])


def test_rrf_weights_shift_ranking():
    dense = [_chunk("a"), _chunk("b")]
    sparse = [_chunk("b"), _chunk("a")]
    assert fuse_results(dense, sparse, "rrf", 0.9, 0.1, top_k=2)[0].chunk_id == "a"
    assert fuse_results(dense, sparse, "rrf", 0.1, 0.9, top_k=2)[0].chunk_id == "b"


def test_weighted_fusion_uses_minmax_normalised_scores():
    dense = [_chunk("a", score=0.9), _chunk("b", score=0.5)]  # -> 1.0, 0.0
    sparse = [_chunk("b", score=10.0), _chunk("c", score=0.0)]  # -> 1.0, 0.0
    out = {c.chunk_id: c.score for c in fuse_results(dense, sparse, "weighted", 0.5, 0.5, top_k=3)}
    assert out == {"a": pytest.approx(0.5), "b": pytest.approx(0.5), "c": pytest.approx(0.0)}


def test_fusion_respects_top_k_handles_empty_and_rejects_bad_method():
    dense = [_chunk("a"), _chunk("b"), _chunk("c")]
    assert len(fuse_results(dense, [], "rrf", top_k=2)) == 2
    assert fuse_results([], [], "weighted", top_k=3) == []
    with pytest.raises(ValueError):
        fuse_results(dense, [], method="bogus")


def test_bm25_ranks_lexical_match_first_and_drops_non_matches():
    idx = BM25Index([
        _chunk("a", "the transformer uses multi head attention"),
        _chunk("b", "reinforcement learning from human feedback"),
        _chunk("c", "mixture of experts routing"),
        _chunk("d", "sliding window attention in mistral"),
    ])
    out = idx.search("RLHF human feedback", top_k=3)
    assert [c.chunk_id for c in out] == ["b"]
    assert out[0].score > 0
    assert idx.search("!!!", top_k=3) == []
    assert BM25Index([]).search("x", top_k=3) == []


class _FakeQueryClient:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        off, lim = kwargs["offset"], kwargs["limit"]
        return self.rows[off : off + lim]


def test_fetch_all_chunks_pages_through_collection(monkeypatch):
    monkeypatch.setattr("src.retrieval._QUERY_PAGE_SIZE", 2)
    rows = [{"chunk_id": f"c{i}", "title": "T", "filename": "f.pdf", "page_number": i, "text": f"t{i}"} for i in range(5)]
    client = _FakeQueryClient(rows)
    chunks = fetch_all_chunks(client, "col")
    assert [c.chunk_id for c in chunks] == ["c0", "c1", "c2", "c3", "c4"]
    assert chunks[3].page_number == 3
    assert client.calls[0]["collection_name"] == "col"


def test_retrieve_hybrid_fuses_dense_and_bm25(monkeypatch):
    rows = [
        {"chunk_id": "a", "title": "A", "filename": "a.pdf", "page_number": 1, "text": "transformer attention mechanism"},
        {"chunk_id": "b", "title": "B", "filename": "b.pdf", "page_number": 2, "text": "rlhf human feedback reward model"},
        {"chunk_id": "c", "title": "C", "filename": "c.pdf", "page_number": 3, "text": "mixture of experts"},
    ]

    class _Client(_FakeQueryClient):
        def search(self, **kwargs):
            # dense ranks a first, c second; BM25 will pull b to the top for the query below
            return [[{"id": "a", "distance": 0.9, "entity": rows[0]}, {"id": "c", "distance": 0.8, "entity": rows[2]}]]

    client = _Client(rows)
    monkeypatch.setattr(retrieval, "_BM25_CACHE", {})
    monkeypatch.setattr("src.retrieval.get_backend", lambda kind, config: _FakeBackend("m"))
    monkeypatch.setattr("src.retrieval.collection_name_for", lambda prefix, model: "col")
    monkeypatch.setattr("src.retrieval.get_client", lambda config: client)

    cfg = AppConfig(dataset_dir=DATASET_DIR, hybrid_candidate_k=3, fusion_method="rrf")
    out = retrieve_hybrid("human feedback", cfg, top_k=3)
    ids = [c.chunk_id for c in out]
    assert set(ids) == {"a", "b", "c"}  # b comes only from the sparse side
    assert ids[0] in {"a", "b"}
    # index is cached: a second call must not re-query the collection
    n_calls = len(client.calls)
    retrieve_hybrid("human feedback", cfg, top_k=3)
    assert len(client.calls) == n_calls


def test_retrieve_dispatches_hybrid(monkeypatch):
    monkeypatch.setattr("src.retrieval.retrieve_hybrid", lambda q, c, k=None: ["sentinel"])
    cfg = AppConfig(dataset_dir=DATASET_DIR)
    assert retrieve("q", mode="hybrid", config=cfg) == ["sentinel"]


# --------------------------------------------------------------------------- #
# Reranking (Milestone 4, config D)
# --------------------------------------------------------------------------- #


class _StubCrossEncoder:
    """Scores by how many query words appear in the text."""

    def __init__(self):
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return [float(sum(w in text for w in q.split())) for q, text in pairs]


def test_rerank_reorders_by_cross_encoder_score_and_truncates():
    chunks = [_chunk("a", "nothing relevant", 0.9), _chunk("b", "human feedback reward", 0.5), _chunk("c", "human only", 0.1)]
    model = _StubCrossEncoder()
    out = rerank("human feedback", chunks, model, top_k=2)
    assert [c.chunk_id for c in out] == ["b", "c"]
    assert out[0].score == 2.0  # cross-encoder score replaces the retrieval score
    assert model.pairs == [("human feedback", c.text) for c in chunks]


def test_rerank_is_stable_on_ties_and_handles_empty():
    chunks = [_chunk("a", "x"), _chunk("b", "y")]
    assert [c.chunk_id for c in rerank("q", chunks, _StubCrossEncoder(), top_k=2)] == ["a", "b"]
    assert rerank("q", [], _StubCrossEncoder(), top_k=3) == []


def test_retrieve_hybrid_reranked_uses_candidate_pool_then_rerank_top_k(monkeypatch):
    seen = {}

    def fake_hybrid(query, config, top_k=None):
        seen["hybrid_top_k"] = top_k
        return [_chunk(str(i), "human feedback" if i == 7 else "other") for i in range(10)]

    monkeypatch.setattr("src.retrieval.retrieve_hybrid", fake_hybrid)
    monkeypatch.setattr("src.retrieval.get_reranker", lambda name: _StubCrossEncoder())

    cfg = AppConfig(dataset_dir=DATASET_DIR, hybrid_candidate_k=10, rerank_top_k=3)
    out = retrieve_hybrid_reranked("human feedback", cfg)
    assert seen["hybrid_top_k"] == 10
    assert len(out) == 3
    assert out[0].chunk_id == "7"

    # explicit top_k overrides rerank_top_k, and the pool never shrinks below it
    cfg = AppConfig(dataset_dir=DATASET_DIR, hybrid_candidate_k=2, rerank_top_k=3)
    retrieve_hybrid_reranked("q", cfg, top_k=6)
    assert seen["hybrid_top_k"] == 6


def test_retrieve_dispatches_hybrid_reranked(monkeypatch):
    monkeypatch.setattr("src.retrieval.retrieve_hybrid_reranked", lambda q, c, k=None: ["sentinel"])
    cfg = AppConfig(dataset_dir=DATASET_DIR)
    assert retrieve("q", mode="hybrid_reranked", config=cfg) == ["sentinel"]
