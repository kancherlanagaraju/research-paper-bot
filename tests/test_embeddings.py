"""Unit tests for src.embeddings.

Both backends' network/model calls are stubbed out (a fake
`sentence_transformers` module, a fake `openai` module) so these tests run
offline and don't require a Hugging Face download or an OpenAI API key.
Live verification: the OpenAI backend was smoke-tested against the real API
during development (see README "Known limitations" / milestone report); the
OSS backend's real model download could not be completed from this sandbox.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import AppConfig, ConfigError
from src.embeddings import OpenAIEmbeddingBackend, OSSEmbeddingBackend, get_backend

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "pinnacle_capstone_data"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer."""

    instances = []  # track every instance created, for assertions

    def __init__(self, model_name):
        self.model_name = model_name
        self.encode_calls = []
        _FakeSentenceTransformer.instances.append(self)

    def encode(self, texts, batch_size=32, normalize_embeddings=False, show_progress_bar=False):
        self.encode_calls.append(
            {"texts": list(texts), "batch_size": batch_size, "normalize_embeddings": normalize_embeddings}
        )
        return np.array([[0.1, 0.2, 0.3, 0.4] for _ in texts])

    def get_sentence_embedding_dimension(self):
        return 4


@pytest.fixture
def fake_sentence_transformers_module(monkeypatch):
    _FakeSentenceTransformer.instances = []
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    return _FakeSentenceTransformer


class _FakeOpenAIEmbeddingsAPI:
    def __init__(self, calls_log, dim=1536):
        self._calls_log = calls_log
        self._dim = dim

    def create(self, model, input):
        self._calls_log.append({"model": model, "input": list(input)})
        data = [SimpleNamespace(embedding=[0.5] * self._dim) for _ in input]
        return SimpleNamespace(data=data)


class _FakeOpenAIClient:
    def __init__(self, api_key, calls_log, dim=1536):
        self.api_key = api_key
        self.embeddings = _FakeOpenAIEmbeddingsAPI(calls_log, dim=dim)


@pytest.fixture
def fake_openai_module(monkeypatch):
    calls_log = []

    def _factory(api_key):
        return _FakeOpenAIClient(api_key, calls_log)

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _factory
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return calls_log


# --------------------------------------------------------------------------- #
# OSS backend
# --------------------------------------------------------------------------- #


def test_oss_backend_lazy_loads_only_on_first_use(fake_sentence_transformers_module):
    backend = OSSEmbeddingBackend("BAAI/bge-small-en-v1.5")
    assert backend._model is None  # importing/constructing must not touch the network
    backend.embed_passages(["hello world"])
    assert backend._model is not None


def test_oss_backend_embed_passages_uses_normalization(fake_sentence_transformers_module):
    backend = OSSEmbeddingBackend("sentence-transformers/all-MiniLM-L6-v2")
    vectors = backend.embed_passages(["a", "b", "c"], batch_size=2)
    assert len(vectors) == 3
    assert all(len(v) == 4 for v in vectors)
    call = backend._model.encode_calls[0]
    assert call["normalize_embeddings"] is True
    assert call["batch_size"] == 2


def test_oss_backend_adds_bge_query_prefix_only_for_bge_models(fake_sentence_transformers_module):
    bge_backend = OSSEmbeddingBackend("BAAI/bge-small-en-v1.5")
    bge_backend.embed_query("what is attention?")
    prefixed_text = bge_backend._model.encode_calls[0]["texts"][0]
    assert prefixed_text.startswith("Represent this sentence for searching relevant passages:")

    minilm_backend = OSSEmbeddingBackend("sentence-transformers/all-MiniLM-L6-v2")
    minilm_backend.embed_query("what is attention?")
    plain_text = minilm_backend._model.encode_calls[0]["texts"][0]
    assert plain_text == "what is attention?"


def test_oss_backend_dimension_property(fake_sentence_transformers_module):
    backend = OSSEmbeddingBackend("BAAI/bge-small-en-v1.5")
    assert backend.dimension == 4


# --------------------------------------------------------------------------- #
# OpenAI backend
# --------------------------------------------------------------------------- #


def test_openai_backend_embeds_and_batches(fake_openai_module):
    backend = OpenAIEmbeddingBackend("text-embedding-3-small", api_key="sk-fake")
    texts = [f"chunk {i}" for i in range(5)]
    vectors = backend.embed_passages(texts, batch_size=2)
    assert len(vectors) == 5
    # 5 texts at batch_size=2 -> 3 API calls (2, 2, 1)
    assert len(fake_openai_module) == 3
    assert [len(c["input"]) for c in fake_openai_module] == [2, 2, 1]


def test_openai_backend_embed_query_returns_single_vector(fake_openai_module):
    backend = OpenAIEmbeddingBackend("text-embedding-3-small", api_key="sk-fake")
    vector = backend.embed_query("a question")
    assert isinstance(vector, list)
    assert len(vector) == 1536


def test_openai_backend_dimension_uses_known_table_without_a_call(fake_openai_module):
    backend = OpenAIEmbeddingBackend("text-embedding-3-small", api_key="sk-fake")
    assert backend.dimension == 1536
    assert len(fake_openai_module) == 0  # no API call needed -- known model


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def test_get_backend_oss_returns_oss_backend(fake_sentence_transformers_module):
    cfg = AppConfig(dataset_dir=DATASET_DIR, embedding_model_oss="BAAI/bge-small-en-v1.5")
    backend = get_backend("oss", cfg)
    assert isinstance(backend, OSSEmbeddingBackend)
    assert backend.name == "BAAI/bge-small-en-v1.5"


def test_get_backend_openai_requires_api_key():
    cfg = AppConfig(dataset_dir=DATASET_DIR, openai_api_key=None)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_backend("openai", cfg)


def test_get_backend_openai_returns_openai_backend(fake_openai_module):
    cfg = AppConfig(dataset_dir=DATASET_DIR, openai_api_key="sk-fake", embedding_model_openai="text-embedding-3-small")
    backend = get_backend("openai", cfg)
    assert isinstance(backend, OpenAIEmbeddingBackend)
    assert backend.name == "text-embedding-3-small"


def test_get_backend_rejects_unknown_kind():
    cfg = AppConfig(dataset_dir=DATASET_DIR)
    with pytest.raises(ValueError):
        get_backend("not-a-real-backend", cfg)
