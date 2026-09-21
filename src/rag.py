"""RAG generation with citations and abstention -- NOT YET IMPLEMENTED (Milestone 6).

Planned contents:
* A prompt template that instructs the LLM to answer only from supplied
  context, say so explicitly when evidence is insufficient, never fabricate
  titles/pages/quotes, distinguish stated facts from cross-paper synthesis,
  and cite as ``[Paper Title, p. N]``.
* ``answer_question(query, retrieval_mode)`` that retrieves context (via
  ``src.retrieval``), calls the configured LLM (``AppConfig.llm_provider`` /
  ``llm_model``), and returns the answer plus the top-3 supporting sources
  (title, filename, page, retrieval score).
"""
from __future__ import annotations


def answer_question(query: str, retrieval_mode: str = "hybrid_reranked"):
    raise NotImplementedError("RAG generation is implemented in Milestone 6.")
