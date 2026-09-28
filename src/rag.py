"""RAG generation with source citations and abstention (Milestone 5).

``answer_question(query, retrieval_mode)`` retrieves context via
``src.retrieval``, asks the configured LLM (``AppConfig.llm_provider`` /
``llm_model``) to answer *only* from that context, and returns the answer
plus the top-3 supporting sources (title, filename, page, retrieval score),
as the brief requires.

Abstention has two layers:
1. No chunks retrieved -> return the abstention message without calling the LLM.
2. The prompt tells the model to reply with exactly ``ABSTAIN_MARKER`` when the
   context does not contain the answer; that reply is detected, replaced by a
   friendly message, and returned with ``abstained=True`` and no sources.

NOTE ON VERIFICATION: the prompt/context building, abstention handling and
orchestration are unit-tested with a stub LLM and stubbed retrieval
(tests/test_rag.py). No real LLM call has been made yet.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.config import AppConfig
from src.retrieval import RetrievedChunk, retrieve

logger = logging.getLogger(__name__)

NUM_SOURCES = 3  # brief: "top 3 will do"
ABSTAIN_MARKER = "INSUFFICIENT_EVIDENCE"
ABSTAIN_MESSAGE = (
    "I couldn't find enough information in the indexed research papers to answer that question."
)

SYSTEM_PROMPT = f"""You are a research assistant answering questions about Generative AI papers.

Rules:
- Answer ONLY from the numbered context passages below. Do not use outside knowledge.
- If the passages do not contain enough evidence to answer, reply with exactly {ABSTAIN_MARKER} and nothing else.
- Never invent paper titles, page numbers, figures, or quotes.
- Cite every claim with the passage's label in the form [Paper Title, p. N], copied exactly from the passage header.
- Clearly separate what a single paper states from any synthesis across several papers (say "Across these papers, ...").
- Be concise and precise."""


@dataclass
class RagAnswer:
    query: str
    answer: str
    abstained: bool
    sources: List[RetrievedChunk] = field(default_factory=list)  # top-3 used; empty on abstention
    retrieved: List[RetrievedChunk] = field(default_factory=list)  # everything given to the LLM
    retrieval_mode: str = ""
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "abstained": self.abstained,
            "sources": [
                {"title": c.title, "filename": c.filename, "page_number": c.page_number, "score": c.score}
                for c in self.sources
            ],
            "retrieval_mode": self.retrieval_mode,
            "model": self.model,
        }


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Numbered passages whose headers are exactly the citation labels."""
    blocks = [
        f"[{i}] [{c.title}, p. {c.page_number}]\n{c.text}" for i, c in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


def build_messages(query: str, chunks: List[RetrievedChunk]) -> List[Dict[str, str]]:
    user = f"Context passages:\n\n{format_context(chunks)}\n\nQuestion: {query}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _is_abstention(text: str) -> bool:
    return not text.strip() or ABSTAIN_MARKER in text


def make_openai_llm(config: AppConfig) -> Callable[[List[Dict[str, str]]], str]:
    """Return ``llm(messages) -> text`` backed by OpenAI chat completions."""
    if config.llm_provider != "openai":
        raise ValueError(f"Unsupported LLM_PROVIDER '{config.llm_provider}'. Only 'openai' is implemented.")
    api_key = config.require_openai_key()
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is not installed. Install it with `pip install openai`.") from exc
    client = OpenAI(api_key=api_key)

    def llm(messages: List[Dict[str, str]]) -> str:
        resp = client.chat.completions.create(model=config.llm_model, messages=messages, temperature=0)
        return resp.choices[0].message.content or ""

    return llm


def answer_question(
    query: str,
    retrieval_mode: str = "hybrid_reranked",
    config: Optional[AppConfig] = None,
    llm: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> RagAnswer:
    """Answer ``query`` from the indexed papers, with citations or abstention.

    ``llm`` is injectable (any ``messages -> text`` callable) for testing.
    """
    config = config or AppConfig.from_env()
    chunks = retrieve(query, mode=retrieval_mode, config=config)
    base = dict(query=query, retrieved=chunks, retrieval_mode=retrieval_mode, model=config.llm_model)

    if not chunks:
        logger.info("No context retrieved for %r; abstaining without an LLM call.", query)
        return RagAnswer(answer=ABSTAIN_MESSAGE, abstained=True, **base)

    llm = llm or make_openai_llm(config)
    text = llm(build_messages(query, chunks)).strip()
    if _is_abstention(text):
        return RagAnswer(answer=ABSTAIN_MESSAGE, abstained=True, **base)
    return RagAnswer(answer=text, abstained=False, sources=chunks[:NUM_SOURCES], **base)


if __name__ == "__main__":
    # Manual live check: `python -m src.rag "your question" [mode]`
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is attention in a transformer model?"
    mode = sys.argv[2] if len(sys.argv) > 2 else "hybrid_reranked"
    result = answer_question(question, retrieval_mode=mode)
    print(result.answer)
    print("\nSources:" if result.sources else "\n(no sources)")
    for c in result.sources:
        print(f"  [{c.score:.4f}] {c.title} ({c.filename}, p.{c.page_number})")
