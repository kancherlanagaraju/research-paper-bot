"""Live smoke tests against real services. SKIPPED by default.

Run after indexing (`python scripts/index_documents.py --embedding-backend both`)
with credentials in `.env`:

    RUN_LIVE_TESTS=1 python -m pytest -m live -v

They exercise the paths the unit tests can only fake: a populated Zilliz
collection, the real BM25 corpus fetch, the cross-encoder weights and a real
LLM answer. Each test names the config it needs so a failure is easy to read.
"""
from __future__ import annotations

import os

import pytest

from src.config import AppConfig
from src.rag import ABSTAIN_MESSAGE, answer_question
from src.retrieval import retrieve

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.getenv("RUN_LIVE_TESTS") != "1", reason="set RUN_LIVE_TESTS=1 to run live tests"),
]

QUESTION = "How many attention heads does the base Transformer use?"


@pytest.fixture(scope="module")
def config():
    return AppConfig.from_env()


@pytest.mark.parametrize("mode", ["dense_oss", "dense_openai", "hybrid", "hybrid_reranked"])
def test_retrieval_mode_returns_cited_chunks(config, mode):
    chunks = retrieve(QUESTION, mode=mode, config=config, top_k=5)
    assert 0 < len(chunks) <= 5
    assert all(c.title and c.filename and c.page_number > 0 and c.text for c in chunks)
    assert any(c.filename == "attention_paper.pdf" for c in chunks)


def test_rag_answers_an_in_corpus_question_with_sources(config):
    result = answer_question(QUESTION, retrieval_mode="hybrid_reranked", config=config)
    assert not result.abstained
    assert 1 <= len(result.sources) <= 3
    assert "[" in result.answer  # inline citation


def test_rag_abstains_on_an_out_of_corpus_question(config):
    result = answer_question("What is the capital of France?", retrieval_mode="hybrid_reranked", config=config)
    assert result.abstained and result.answer == ABSTAIN_MESSAGE and result.sources == []
