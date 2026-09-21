"""Retrieval + answer evaluation -- NOT YET IMPLEMENTED (Milestone 7).

Planned contents:
* Loader for the ~15-20 question evaluation set (question, expected answer /
  key points, expected source paper+page where feasible, category).
* Metrics: Recall@k / source-hit rate, MRR (when expected sources are
  known), answer correctness, faithfulness, citation correctness,
  abstention behavior on unanswerable questions, retrieval + end-to-end
  latency.
* Runs all four retrieval configurations (see ``src.retrieval``) against the
  same question set and writes raw results to
  ``artifacts/evaluation/results.csv`` (or .json) plus a Markdown summary
  comparing configurations and naming the selected one with justification.
"""
from __future__ import annotations


def run_evaluation(question_set_path: str):
    raise NotImplementedError("Evaluation is implemented in Milestone 7.")
