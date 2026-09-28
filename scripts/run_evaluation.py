#!/usr/bin/env python3
"""Run the evaluation suite over all retrieval configurations.

Retrieval metrics only (needs Zilliz collections + model access):
    python scripts/run_evaluation.py

Also run RAG generation and score abstention / citations (needs OPENAI_API_KEY):
    python scripts/run_evaluation.py --with-generation

Add an LLM judge for answer correctness + faithfulness:
    python scripts/run_evaluation.py --with-generation --judge

Writes results.json, results.csv and summary.md under artifacts/evaluation/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig  # noqa: E402
from src.evaluation import DEFAULT_MODES, render_summary_md, run_evaluation  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=None, help="Question set JSON (default: artifacts/evaluation/questions.json)")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=DEFAULT_MODES)
    parser.add_argument("--top-k", type=int, default=None, help="k for every configuration (default: RERANK_TOP_K)")
    parser.add_argument("--with-generation", action="store_true", help="Also run RAG answers (calls the LLM)")
    parser.add_argument("--judge", action="store_true", help="LLM-judge correctness/faithfulness (implies --with-generation)")
    args = parser.parse_args(argv)

    config = AppConfig.from_env()
    config.configure_logging()
    questions = args.questions or str(config.artifacts_dir / "evaluation" / "questions.json")

    judge = None
    if args.judge:
        from src.rag import make_openai_llm

        judge = make_openai_llm(config)
    report = run_evaluation(
        questions,
        config=config,
        modes=args.modes,
        top_k=args.top_k,
        with_generation=args.with_generation or args.judge,
        judge=judge,
    )
    print(render_summary_md(report))
    print(f"Results written to {config.artifacts_dir / 'evaluation'}")
    return 0 if report["selected_mode"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
