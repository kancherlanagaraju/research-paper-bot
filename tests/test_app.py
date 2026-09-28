"""Tests for the Streamlit app (Milestone 7): pure helpers plus a headless
run of app.py via streamlit's AppTest with a stubbed RAG pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.app_support import (
    FALLBACK_MODE,
    default_mode,
    history_entry,
    load_selected_mode,
    mode_options,
    source_rows,
)
from src.rag import RagAnswer
from src.retrieval import RetrievedChunk

ROOT = Path(__file__).resolve().parent.parent


def _chunk(i):
    return RetrievedChunk(
        chunk_id=f"c{i}", score=0.91234 - i * 0.1, title=f"Paper {i}", filename=f"p{i}.pdf", page_number=i + 1,
        text=f"text {i}", doc_id="d", embedding_model="m",
    )


def _answer(abstained=False):
    return RagAnswer(
        query="q?", answer="the answer [Paper 0, p. 1]" if not abstained else "no evidence", abstained=abstained,
        sources=[] if abstained else [_chunk(0), _chunk(1), _chunk(2)], retrieval_mode="hybrid", model="m",
    )


def _write_results(tmp_path, selected):
    d = tmp_path / "evaluation"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"report": {"selected_mode": selected}}))


def test_load_selected_mode_reads_evaluation_report(tmp_path):
    _write_results(tmp_path, "hybrid")
    assert load_selected_mode(tmp_path) == "hybrid"
    assert default_mode(tmp_path) == "hybrid"


def test_default_mode_falls_back_when_missing_corrupt_or_unknown(tmp_path):
    assert default_mode(tmp_path) == FALLBACK_MODE  # no file
    _write_results(tmp_path, "not_a_mode")
    assert default_mode(tmp_path) == FALLBACK_MODE
    (tmp_path / "evaluation" / "results.json").write_text("{broken")
    assert default_mode(tmp_path) == FALLBACK_MODE
    _write_results_null = tmp_path / "evaluation" / "results.json"
    _write_results_null.write_text(json.dumps({"report": {"selected_mode": None}}))
    assert default_mode(tmp_path) == FALLBACK_MODE


def test_mode_options_flags_only_the_selected_mode():
    opts = mode_options("hybrid")
    assert len(opts) == 4 and "(selected by evaluation)" in opts["hybrid"]
    assert sum("selected by evaluation" in v for v in opts.values()) == 1
    assert not any("selected by evaluation" in v for v in mode_options(None).values())


def test_source_rows_and_history_entry():
    rows = source_rows(_answer())
    assert [r["rank"] for r in rows] == [1, 2, 3]
    assert rows[0] == {"rank": 1, "title": "Paper 0", "filename": "p0.pdf", "page": 1, "score": 0.9123, "text": "text 0"}
    entry = history_entry(_answer())
    assert entry["query"] == "q?" and not entry["abstained"] and entry["mode"] == "hybrid" and len(entry["sources"]) == 3
    assert history_entry(_answer(abstained=True))["sources"] == []


# --------------------------------------------------------------------------- #
# Headless app run
# --------------------------------------------------------------------------- #

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def _run_app(monkeypatch, fake):
    monkeypatch.setattr("src.rag.answer_question", fake)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    return at


def test_app_renders_answer_and_three_sources(monkeypatch):
    seen = {}

    def fake(query, retrieval_mode, config):
        seen.update(query=query, mode=retrieval_mode)
        return _answer()

    at = _run_app(monkeypatch, fake)
    at.selectbox[0].select("hybrid")
    at.chat_input[0].set_value("What is attention?").run()
    assert not at.exception
    assert seen == {"query": "What is attention?", "mode": "hybrid"}
    assert any("the answer" in m.value for m in at.markdown)
    assert len(at.expander) == 3
    assert not at.warning and not at.error


def test_app_shows_insufficient_evidence_warning_without_sources(monkeypatch):
    at = _run_app(monkeypatch, lambda query, retrieval_mode, config: _answer(abstained=True))
    at.chat_input[0].set_value("capital of France?").run()
    assert [w.value for w in at.warning] == ["no evidence"]
    assert len(at.expander) == 0


def test_app_reports_pipeline_errors_in_the_ui(monkeypatch):
    def boom(query, retrieval_mode, config):
        raise RuntimeError("Zilliz unreachable")

    at = _run_app(monkeypatch, boom)
    at.chat_input[0].set_value("anything").run()
    assert not at.exception  # handled, not a stack trace
    assert any("Zilliz unreachable" in e.value for e in at.error)


def test_app_keeps_history_across_questions(monkeypatch):
    at = _run_app(monkeypatch, lambda query, retrieval_mode, config: _answer())
    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("second").run()
    assert [e["query"] for e in at.session_state["history"]] == ["q?", "q?"]
    assert len(at.chat_message) >= 4
