"""Unit tests for src.evaluation (Milestone 6). Retrieval and the LLM are stubbed."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import AppConfig
from src.evaluation import (
    Question,
    aggregate_generation,
    aggregate_retrieval,
    choose_best,
    citation_metrics,
    evaluate_generation,
    evaluate_retrieval,
    hit_and_rr,
    is_relevant,
    load_questions,
    parse_citations,
    parse_judge_reply,
    render_summary_md,
    run_evaluation,
)
from src.retrieval import RetrievedChunk

ROOT = Path(__file__).resolve().parent.parent
CFG = AppConfig(dataset_dir=ROOT / "pinnacle_capstone_data")


def _chunk(cid, text="", filename="a.pdf", title="Paper A", page=1):
    return RetrievedChunk(chunk_id=cid, score=0.5, title=title, filename=filename, page_number=page, text=text, doc_id="d", embedding_model="m")


Q_FACT = Question("q1", "factual", "What?", "ref", ("a.pdf",), ("magic term",))
Q_UNANS = Question("q2", "unanswerable", "Capital of France?")


def test_shipped_question_set_is_valid_and_grounded():
    qs = load_questions(ROOT / "artifacts" / "evaluation" / "questions.json")
    assert len(qs) >= 15 and {q.category for q in qs} == {"factual", "cross-paper", "unanswerable"}
    manifest = json.loads((ROOT / "artifacts" / "ingestion_manifest.json").read_text())
    chunks = [_chunk(c["chunk_id"], c["text"], c["filename"], c["title"], c["page_number"]) for c in manifest["chunks"]]
    for q in qs:
        if q.answerable:
            assert any(is_relevant(c, q) for c in chunks), f"{q.id} has no relevant chunk in the corpus"


def test_load_questions_validation(tmp_path):
    def write(items):
        p = tmp_path / "q.json"
        p.write_text(json.dumps(items))
        return p

    with pytest.raises(ValueError, match="category"):
        load_questions(write([{"id": "a", "category": "weird", "question": "x"}]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_questions(write([{"id": "a", "category": "unanswerable", "question": "x"}] * 2))
    with pytest.raises(ValueError, match="expected_docs"):
        load_questions(write([{"id": "a", "category": "factual", "question": "x"}]))


def test_is_relevant_needs_right_doc_and_term_case_and_whitespace_insensitive():
    assert is_relevant(_chunk("1", "the  MAGIC\nterm here"), Q_FACT)
    assert not is_relevant(_chunk("2", "the magic term", filename="other.pdf"), Q_FACT)
    assert not is_relevant(_chunk("3", "nothing"), Q_FACT)


def test_hit_and_rr():
    assert hit_and_rr([False, False, True]) == (1.0, pytest.approx(1 / 3))
    assert hit_and_rr([True]) == (1.0, 1.0)
    assert hit_and_rr([False, False]) == (0.0, 0.0)
    assert hit_and_rr([]) == (0.0, 0.0)


def test_evaluate_retrieval_skips_unanswerable_and_records_metrics():
    calls = []

    def fake_retrieve(query, mode, config, top_k):
        calls.append((query, mode, top_k))
        return [_chunk("x", "nope"), _chunk("y", "magic term", page=4)]

    rows = evaluate_retrieval([Q_FACT, Q_UNANS], "hybrid", CFG, 5, fake_retrieve)
    assert calls == [("What?", "hybrid", 5)]
    assert len(rows) == 1
    assert rows[0]["hit"] == 1.0 and rows[0]["rr"] == 0.5 and rows[0]["relevance"] == [False, True]
    assert rows[0]["retrieved"] == ["a.pdf#p1", "a.pdf#p4"]
    agg = aggregate_retrieval(rows)
    assert agg["hit_at_k"] == 1.0 and agg["mrr"] == 0.5 and agg["hit_at_k_by_category"] == {"factual": 1.0}


def test_choose_best_uses_mrr_then_hit_then_latency():
    base = {"n": 5, "hit_at_k": 0.8, "mrr": 0.6, "mean_latency_s": 1.0}
    assert choose_best({"a": base, "b": {**base, "mrr": 0.7}}) == "b"
    assert choose_best({"a": base, "b": {**base, "hit_at_k": 0.9}}) == "b"
    assert choose_best({"a": base, "b": {**base, "mean_latency_s": 0.5}}) == "b"
    assert choose_best({}) is None and choose_best({"a": {"n": 0}}) is None


def test_citation_parsing_and_metrics():
    text = "X is true [Paper A, p. 1]. Also Y [Paper A, p. 9] and [Made Up, p. 2]."
    assert parse_citations(text) == [("Paper A", 1), ("Paper A", 9), ("Made Up", 2)]
    retrieved = [_chunk("1", "magic term", page=1), _chunk("2", "other", page=3)]
    m = citation_metrics(text, retrieved, Q_FACT)
    assert m["n_citations"] == 3 and m["citation_valid"] == pytest.approx(1 / 3) and m["citation_hit"] == 1.0
    assert citation_metrics("no citations", retrieved, Q_FACT)["citation_valid"] == 0.0


def test_parse_judge_reply():
    assert parse_judge_reply('Sure: {"correctness": 1, "faithfulness": 0.5}') == {"correctness": 1.0, "faithfulness": 0.5}
    assert parse_judge_reply('{"correctness": 3, "faithfulness": -1}') == {"correctness": 1.0, "faithfulness": 0.0}
    assert parse_judge_reply("garbage") is None and parse_judge_reply('{"correctness": 1}') is None


def _answer(abstained, text="", retrieved=()):
    return SimpleNamespace(abstained=abstained, answer=text, retrieved=list(retrieved))


def test_generation_metrics_abstention_citations_and_judge():
    retrieved = [_chunk("1", "magic term", page=2)]

    def answer_fn(question, retrieval_mode, config):
        if question == "Capital of France?":
            return _answer(True)
        return _answer(False, "It is so [Paper A, p. 2].", retrieved)

    rows = evaluate_generation([Q_FACT, Q_UNANS], "hybrid", CFG, answer_fn, judge=lambda m: '{"correctness": 1, "faithfulness": 1}')
    assert [r["abstention_correct"] for r in rows] == [True, True]
    assert rows[0]["citation_hit"] == 1.0 and rows[0]["correctness"] == 1.0
    agg = aggregate_generation(rows)
    assert agg["abstention_accuracy"] == 1.0 and agg["correct_abstain_rate"] == 1.0
    assert agg["false_abstain_rate"] == 0.0 and agg["faithfulness"] == 1.0


def test_generation_flags_wrong_abstention():
    # A bot that answers everything is wrong on the unanswerable question.
    rows = evaluate_generation([Q_FACT, Q_UNANS], "hybrid", CFG, lambda q, retrieval_mode, config: _answer(False, "x"))
    agg = aggregate_generation(rows)
    assert agg["abstention_accuracy"] == 0.5 and agg["correct_abstain_rate"] == 0.0
    # A bot that abstains on everything is wrong on the answerable question.
    rows = evaluate_generation([Q_FACT, Q_UNANS], "hybrid", CFG, lambda q, retrieval_mode, config: _answer(True))
    agg = aggregate_generation(rows)
    assert agg["abstention_accuracy"] == 0.5 and agg["false_abstain_rate"] == 1.0


def _write_questions(tmp_path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([
        {"id": "q1", "category": "factual", "question": "What?", "expected_docs": ["a.pdf"], "evidence_terms": ["magic"]},
        {"id": "q2", "category": "unanswerable", "question": "Capital?"},
    ]))
    return p


def test_run_evaluation_end_to_end_selects_best_and_writes_reports(tmp_path):
    seen_cfg = {}

    def fake_retrieve(query, mode, config, top_k):
        seen_cfg[mode] = config.hybrid_dense_backend
        good = {"dense_openai": 0, "hybrid": 0, "hybrid_reranked": 0}.get(mode)  # oss ranks it 2nd
        chunks = [_chunk("x", "magic")] if good == 0 else [_chunk("x", "nope"), _chunk("y", "magic")]
        return chunks

    report = run_evaluation(_write_questions(tmp_path), CFG, top_k=5, output_dir=tmp_path / "out", retrieve_fn=fake_retrieve)
    assert report["best_dense_backend"] == "openai"
    assert seen_cfg["hybrid"] == "openai" and seen_cfg["hybrid_reranked"] == "openai"
    assert report["selected_mode"] in {"dense_openai", "hybrid", "hybrid_reranked"}
    assert report["summaries"]["dense_oss"]["mrr"] == 0.5 and report["summaries"]["dense_openai"]["mrr"] == 1.0
    for name in ("results.json", "results.csv", "summary.md"):
        assert (tmp_path / "out" / name).exists()
    md = (tmp_path / "out" / "summary.md").read_text()
    assert "Selected configuration" in md and "B (OpenAI dense)" in md


def test_run_evaluation_survives_a_failing_configuration(tmp_path):
    def fake_retrieve(query, mode, config, top_k):
        if mode == "dense_oss":
            raise RuntimeError("model download blocked")
        return [_chunk("x", "magic")]

    report = run_evaluation(_write_questions(tmp_path), CFG, top_k=5, output_dir=tmp_path / "out", retrieve_fn=fake_retrieve)
    assert "dense_oss" in report["errors"] and "model download blocked" in report["errors"]["dense_oss"]
    assert "dense_oss" not in report["summaries"]
    assert report["best_dense_backend"] == "openai"
    assert "Configurations that failed" in render_summary_md(report)


def test_run_evaluation_with_nothing_working_selects_nothing(tmp_path):
    def boom(query, mode, config, top_k):
        raise RuntimeError("down")

    report = run_evaluation(_write_questions(tmp_path), CFG, top_k=5, output_dir=tmp_path / "out", retrieve_fn=boom)
    assert report["selected_mode"] is None and len(report["errors"]) == 4
    assert "nothing was selected" in (tmp_path / "out" / "summary.md").read_text()
