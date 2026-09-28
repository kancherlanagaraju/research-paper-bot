"""Retrieval + answer evaluation (Milestone 6).

Runs the four retrieval configurations (see ``src.retrieval``) against one
fixed question set under identical conditions (same corpus, chunking and
``top_k``), then picks a final configuration with a documented rule.

Question set (``artifacts/evaluation/questions.json``): each item has
``id``, ``category`` (``factual`` | ``cross-paper`` | ``unanswerable``),
``question``, ``reference_answer``, ``expected_docs`` (filenames) and
``evidence_terms``. A retrieved chunk counts as RELEVANT when it comes from
an expected doc AND contains at least one evidence term (case-insensitive,
whitespace-collapsed). Judging relevance by text rather than chunk ID keeps
the labels valid if chunking changes.

Retrieval metrics (answerable questions only): hit@k (any relevant chunk in
the top k), MRR, mean retrieval latency. Generation metrics (optional, need
an LLM): abstention accuracy, citation validity (every cited [Title, p. N]
was actually retrieved), citation hit (a cited source is relevant), and, if
a judge is supplied, answer correctness + faithfulness (0-1, LLM-judged).

Selection rule (``choose_best``): highest MRR, then hit@k, then lowest mean
latency. For C/D the dense side is the better of A/B by that same rule.

NOTE ON VERIFICATION: metric maths, relevance matching, aggregation, the
selection rule and report writing are unit-tested with stub retrievers and
LLMs (tests/test_evaluation.py). The real run needs a populated Zilliz
collection, model weights and an OpenAI key -- see README.
"""
from __future__ import annotations

import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.config import AppConfig
from src.retrieval import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

CATEGORIES = {"factual", "cross-paper", "unanswerable"}
DEFAULT_MODES = ["dense_oss", "dense_openai", "hybrid", "hybrid_reranked"]
MODE_LABELS = {
    "dense_oss": "A (OSS dense)",
    "dense_openai": "B (OpenAI dense)",
    "hybrid": "C (hybrid)",
    "hybrid_reranked": "D (hybrid + rerank)",
}
_DENSE_MODE_TO_BACKEND = {"dense_oss": "oss", "dense_openai": "openai"}
_CITATION_RE = re.compile(r"\[([^\[\]]+), p\. (\d+)\]")


# --------------------------------------------------------------------------- #
# Question set
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    reference_answer: str = ""
    expected_docs: tuple = ()
    evidence_terms: tuple = ()

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


def load_questions(path: Path | str) -> List[Question]:
    items = json.loads(Path(path).read_text())
    questions: List[Question] = []
    seen = set()
    for item in items:
        q = Question(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            reference_answer=item.get("reference_answer", ""),
            expected_docs=tuple(item.get("expected_docs", [])),
            evidence_terms=tuple(item.get("evidence_terms", [])),
        )
        if q.category not in CATEGORIES:
            raise ValueError(f"Question {q.id}: category must be one of {sorted(CATEGORIES)}, got '{q.category}'.")
        if q.id in seen:
            raise ValueError(f"Duplicate question id '{q.id}'.")
        if q.answerable and not (q.expected_docs and q.evidence_terms):
            raise ValueError(f"Answerable question {q.id} needs expected_docs and evidence_terms.")
        seen.add(q.id)
        questions.append(q)
    return questions


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def is_relevant(chunk: RetrievedChunk, question: Question) -> bool:
    if chunk.filename not in question.expected_docs:
        return False
    text = _norm(chunk.text)
    return any(_norm(term) in text for term in question.evidence_terms)


# --------------------------------------------------------------------------- #
# Retrieval metrics
# --------------------------------------------------------------------------- #


def hit_and_rr(relevance: Sequence[bool]) -> tuple:
    """(hit, reciprocal_rank) for one ranked list of relevance flags."""
    for rank, rel in enumerate(relevance, start=1):
        if rel:
            return 1.0, 1.0 / rank
    return 0.0, 0.0


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval(
    questions: Sequence[Question],
    mode: str,
    config: AppConfig,
    top_k: int,
    retrieve_fn: Callable[..., List[RetrievedChunk]] = retrieve,
) -> List[Dict[str, Any]]:
    """One row per ANSWERABLE question; unanswerable ones have no relevance labels."""
    rows = []
    for q in questions:
        if not q.answerable:
            continue
        start = time.perf_counter()
        chunks = retrieve_fn(q.question, mode=mode, config=config, top_k=top_k)
        latency = time.perf_counter() - start
        relevance = [is_relevant(c, q) for c in chunks]
        hit, rr = hit_and_rr(relevance)
        rows.append(
            {
                "mode": mode,
                "question_id": q.id,
                "category": q.category,
                "hit": hit,
                "rr": rr,
                "retrieval_latency_s": latency,
                "retrieved": [f"{c.filename}#p{c.page_number}" for c in chunks],
                "relevance": relevance,
            }
        )
    return rows


def aggregate_retrieval(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_cat: Dict[str, List[float]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["hit"])
    return {
        "n": len(rows),
        "hit_at_k": _mean([r["hit"] for r in rows]),
        "mrr": _mean([r["rr"] for r in rows]),
        "mean_latency_s": _mean([r["retrieval_latency_s"] for r in rows]),
        "hit_at_k_by_category": {c: _mean(v) for c, v in sorted(by_cat.items())},
    }


def choose_best(summaries: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Highest MRR, then hit@k, then lowest latency. None if nothing ran."""
    valid = {m: s for m, s in summaries.items() if s and s.get("n")}
    if not valid:
        return None
    return min(valid, key=lambda m: (-valid[m]["mrr"], -valid[m]["hit_at_k"], valid[m]["mean_latency_s"]))


# --------------------------------------------------------------------------- #
# Generation metrics
# --------------------------------------------------------------------------- #

JUDGE_PROMPT = """You are grading an answer from a research-paper Q&A bot.
Question: {question}
Reference answer: {reference}

Context passages given to the bot:
{context}

Bot answer: {answer}

Score two things from 0.0 to 1.0:
- correctness: does the answer agree with the reference answer?
- faithfulness: is every claim in the answer supported by the context passages?
Reply with JSON only: {{"correctness": <float>, "faithfulness": <float>}}"""


def parse_citations(answer: str) -> List[tuple]:
    return [(title.strip(), int(page)) for title, page in _CITATION_RE.findall(answer)]


def citation_metrics(answer: str, retrieved: Sequence[RetrievedChunk], question: Question) -> Dict[str, Any]:
    cites = parse_citations(answer)
    if not cites:
        return {"n_citations": 0, "citation_valid": 0.0, "citation_hit": 0.0}
    by_label = {(c.title, c.page_number): c for c in retrieved}
    valid = [by_label[c] for c in cites if c in by_label]
    return {
        "n_citations": len(cites),
        "citation_valid": len(valid) / len(cites),
        "citation_hit": 1.0 if any(is_relevant(c, question) for c in valid) else 0.0,
    }


def parse_judge_reply(reply: str) -> Optional[Dict[str, float]]:
    match = re.search(r"\{.*\}", reply, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return {k: min(1.0, max(0.0, float(data[k]))) for k in ("correctness", "faithfulness")}
    except (ValueError, KeyError, TypeError):
        return None


def evaluate_generation(
    questions: Sequence[Question],
    mode: str,
    config: AppConfig,
    answer_fn: Optional[Callable[..., Any]] = None,
    judge: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> List[Dict[str, Any]]:
    from src.rag import answer_question

    answer_fn = answer_fn or answer_question
    rows = []
    for q in questions:
        start = time.perf_counter()
        result = answer_fn(q.question, retrieval_mode=mode, config=config)
        latency = time.perf_counter() - start
        row: Dict[str, Any] = {
            "mode": mode,
            "question_id": q.id,
            "category": q.category,
            "abstained": result.abstained,
            "abstention_correct": result.abstained == (not q.answerable),
            "end_to_end_latency_s": latency,
            "answer": result.answer,
        }
        if q.answerable and not result.abstained:
            row.update(citation_metrics(result.answer, result.retrieved, q))
            if judge is not None:
                context = "\n\n".join(f"[{c.title}, p. {c.page_number}] {c.text}" for c in result.retrieved)
                prompt = JUDGE_PROMPT.format(
                    question=q.question, reference=q.reference_answer, context=context, answer=result.answer
                )
                scores = parse_judge_reply(judge([{"role": "user", "content": prompt}]))
                if scores:
                    row.update(scores)
        rows.append(row)
    return rows


def aggregate_generation(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def col(name: str, subset=rows) -> List[float]:
        return [float(r[name]) for r in subset if name in r]

    unanswerable = [r for r in rows if r["category"] == "unanswerable"]
    answerable = [r for r in rows if r["category"] != "unanswerable"]
    return {
        "n": len(rows),
        "abstention_accuracy": _mean([float(r["abstention_correct"]) for r in rows]),
        "correct_abstain_rate": _mean([float(r["abstained"]) for r in unanswerable]),
        "false_abstain_rate": _mean([float(r["abstained"]) for r in answerable]),
        "citation_valid": _mean(col("citation_valid")),
        "citation_hit": _mean(col("citation_hit")),
        "correctness": _mean(col("correctness")) if col("correctness") else None,
        "faithfulness": _mean(col("faithfulness")) if col("faithfulness") else None,
        "mean_end_to_end_latency_s": _mean(col("end_to_end_latency_s")),
    }


# --------------------------------------------------------------------------- #
# Orchestration + reporting
# --------------------------------------------------------------------------- #


def _config_for_mode(config: AppConfig, mode: str, best_dense_backend: Optional[str]) -> AppConfig:
    """Hybrid modes use the winning dense backend from A/B, if one was decided."""
    if mode in _DENSE_MODE_TO_BACKEND or not best_dense_backend:
        return config
    return replace(config, hybrid_dense_backend=best_dense_backend)


def run_evaluation(
    question_set_path: Path | str,
    config: Optional[AppConfig] = None,
    modes: Sequence[str] = tuple(DEFAULT_MODES),
    top_k: Optional[int] = None,
    with_generation: bool = False,
    judge: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    output_dir: Optional[Path | str] = None,
    retrieve_fn: Callable[..., List[RetrievedChunk]] = retrieve,
    answer_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    config = config or AppConfig.from_env()
    questions = load_questions(question_set_path)
    k = top_k if top_k is not None else config.rerank_top_k
    out_dir = Path(output_dir) if output_dir else config.artifacts_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    retrieval_rows: List[Dict[str, Any]] = []
    generation_rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}
    best_dense_backend: Optional[str] = None

    # Dense modes first so C/D can use the better embedding model.
    ordered = sorted(modes, key=lambda m: m not in _DENSE_MODE_TO_BACKEND)
    for mode in ordered:
        cfg = _config_for_mode(config, mode, best_dense_backend)
        try:
            rows = evaluate_retrieval(questions, mode, cfg, k, retrieve_fn)
            summary = aggregate_retrieval(rows)
            if with_generation:
                gen = evaluate_generation(questions, mode, cfg, answer_fn, judge)
                generation_rows.extend(gen)
                summary["generation"] = aggregate_generation(gen)
        except Exception as exc:  # noqa: BLE001 -- one broken config must not sink the whole run
            logger.exception("Configuration %s failed", mode)
            errors[mode] = f"{type(exc).__name__}: {exc}"
            continue
        retrieval_rows.extend(rows)
        summaries[mode] = summary
        if mode in _DENSE_MODE_TO_BACKEND:
            dense_summaries = {m: s for m, s in summaries.items() if m in _DENSE_MODE_TO_BACKEND}
            best = choose_best(dense_summaries)
            best_dense_backend = _DENSE_MODE_TO_BACKEND[best] if best else None

    selected = choose_best(summaries)
    report = {
        "top_k": k,
        "n_questions": len(questions),
        "n_answerable": sum(q.answerable for q in questions),
        "modes_run": list(summaries),
        "errors": errors,
        "best_dense_backend": best_dense_backend,
        "selected_mode": selected,
        "summaries": summaries,
    }
    write_reports(out_dir, report, retrieval_rows, generation_rows)
    return report


def _fmt(value: Optional[float], digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_summary_md(report: Dict[str, Any]) -> str:
    k = report["top_k"]
    lines = [
        "# Evaluation summary",
        "",
        f"{report['n_questions']} questions ({report['n_answerable']} answerable), top_k = {k}, "
        "same corpus and chunking for every configuration.",
        "",
        f"## Retrieval (answerable questions, k={k})",
        "",
        f"| Configuration | hit@{k} | MRR | mean latency (s) |",
        "|---|---|---|---|",
    ]
    for mode, s in report["summaries"].items():
        lines.append(f"| {MODE_LABELS.get(mode, mode)} | {_fmt(s['hit_at_k'])} | {_fmt(s['mrr'])} | {_fmt(s['mean_latency_s'])} |")
    gen = {m: s["generation"] for m, s in report["summaries"].items() if "generation" in s}
    if gen:
        lines += [
            "",
            "## Generation",
            "",
            "| Configuration | abstention acc. | false abstain | citation valid | citation hit | correctness | faithfulness | e2e latency (s) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for mode, g in gen.items():
            lines.append(
                f"| {MODE_LABELS.get(mode, mode)} | {_fmt(g['abstention_accuracy'])} | {_fmt(g['false_abstain_rate'])} | "
                f"{_fmt(g['citation_valid'])} | {_fmt(g['citation_hit'])} | {_fmt(g['correctness'])} | "
                f"{_fmt(g['faithfulness'])} | {_fmt(g['mean_end_to_end_latency_s'])} |"
            )
    lines += ["", "## Selected configuration", ""]
    selected = report["selected_mode"]
    if selected:
        s = report["summaries"][selected]
        lines.append(
            f"**{MODE_LABELS.get(selected, selected)}** (`{selected}`): highest MRR ({_fmt(s['mrr'])}) with "
            f"hit@{k} {_fmt(s['hit_at_k'])}. Rule: highest MRR, then hit@k, then lowest latency."
        )
        if report.get("best_dense_backend"):
            lines.append(f"Hybrid configurations used the `{report['best_dense_backend']}` dense backend (the better of A/B).")
    else:
        lines.append("No configuration completed, so nothing was selected.")
    if report["errors"]:
        lines += ["", "## Configurations that failed", ""]
        lines += [f"- `{m}`: {err}" for m, err in report["errors"].items()]
    return "\n".join(lines) + "\n"


def write_reports(out_dir: Path, report: Dict[str, Any], retrieval_rows: List[Dict], generation_rows: List[Dict]) -> None:
    (out_dir / "results.json").write_text(
        json.dumps({"report": report, "retrieval": retrieval_rows, "generation": generation_rows}, indent=2)
    )
    if retrieval_rows:
        with open(out_dir / "results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "question_id", "category", "hit", "rr", "retrieval_latency_s", "retrieved"])
            for r in retrieval_rows:
                writer.writerow([r["mode"], r["question_id"], r["category"], r["hit"], f"{r['rr']:.4f}",
                                 f"{r['retrieval_latency_s']:.4f}", ";".join(r["retrieved"])])
    (out_dir / "summary.md").write_text(render_summary_md(report))
