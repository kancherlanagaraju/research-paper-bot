"""Unit tests for src.rag (Milestone 5). Retrieval and the LLM are stubbed."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import AppConfig
from src.rag import (
    ABSTAIN_MARKER,
    ABSTAIN_MESSAGE,
    NUM_SOURCES,
    SYSTEM_PROMPT,
    answer_question,
    build_messages,
    format_context,
    make_openai_llm,
)
from src.retrieval import RetrievedChunk

DATASET_DIR = Path(__file__).resolve().parent.parent / "pinnacle_capstone_data"
CFG = AppConfig(dataset_dir=DATASET_DIR)


def _chunk(i, title="Attention Is All You Need", page=3, text="Some text.", score=0.5):
    return RetrievedChunk(
        chunk_id=f"c{i}", score=score, title=title, filename="a.pdf", page_number=page + i, text=text, doc_id="d", embedding_model="m"
    )


def _stub_retrieve(monkeypatch, chunks, seen=None):
    def fake(query, mode="x", config=None, top_k=None):
        if seen is not None:
            seen["mode"] = mode
        return chunks

    monkeypatch.setattr("src.rag.retrieve", fake)


def test_format_context_headers_are_citation_labels():
    ctx = format_context([_chunk(0, text="alpha"), _chunk(1, title="GPT-4", text="beta")])
    assert "[1] [Attention Is All You Need, p. 3]\nalpha" in ctx
    assert "[2] [GPT-4, p. 4]\nbeta" in ctx


def test_build_messages_has_system_rules_context_and_question():
    msgs = build_messages("What is RLHF?", [_chunk(0, text="alpha")])
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert ABSTAIN_MARKER in msgs[0]["content"]
    assert "alpha" in msgs[1]["content"] and msgs[1]["content"].endswith("Question: What is RLHF?")


def test_answer_returns_top3_sources_and_passes_mode(monkeypatch):
    seen = {}
    chunks = [_chunk(i, score=0.9 - i * 0.1) for i in range(5)]
    _stub_retrieve(monkeypatch, chunks, seen)
    result = answer_question("q", "hybrid", CFG, llm=lambda m: "Attention is X [Attention Is All You Need, p. 3].")
    assert not result.abstained
    assert seen["mode"] == "hybrid"
    assert result.sources == chunks[:NUM_SOURCES] and len(result.sources) == 3
    assert result.retrieved == chunks
    d = result.to_dict()
    assert d["sources"][0] == {"title": "Attention Is All You Need", "filename": "a.pdf", "page_number": 3, "score": 0.9}


def test_llm_receives_all_retrieved_context(monkeypatch):
    _stub_retrieve(monkeypatch, [_chunk(i, text=f"passage-{i}") for i in range(5)])
    captured = {}

    def llm(messages):
        captured["user"] = messages[1]["content"]
        return "ok"

    answer_question("q", "hybrid", CFG, llm=llm)
    assert all(f"passage-{i}" in captured["user"] for i in range(5))


@pytest.mark.parametrize("reply", [ABSTAIN_MARKER, f"  {ABSTAIN_MARKER}\n", "", "   "])
def test_llm_abstention_is_detected_and_drops_sources(monkeypatch, reply):
    _stub_retrieve(monkeypatch, [_chunk(0)])
    result = answer_question("q", "hybrid", CFG, llm=lambda m: reply)
    assert result.abstained and result.answer == ABSTAIN_MESSAGE
    assert result.sources == [] and len(result.retrieved) == 1


def test_no_retrieval_results_abstains_without_calling_llm(monkeypatch):
    _stub_retrieve(monkeypatch, [])

    def llm(messages):
        raise AssertionError("LLM must not be called with no context")

    result = answer_question("q", "hybrid", CFG, llm=llm)
    assert result.abstained and result.sources == []


def test_default_mode_is_hybrid_reranked(monkeypatch):
    seen = {}
    _stub_retrieve(monkeypatch, [_chunk(0)], seen)
    answer_question("q", config=CFG, llm=lambda m: "ok")
    assert seen["mode"] == "hybrid_reranked"


def test_make_openai_llm_rejects_unknown_provider():
    with pytest.raises(ValueError, match="anthropic"):
        make_openai_llm(AppConfig(dataset_dir=DATASET_DIR, llm_provider="anthropic"))


def test_make_openai_llm_requires_api_key():
    from src.config import ConfigError

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        make_openai_llm(AppConfig(dataset_dir=DATASET_DIR, openai_api_key=None))
