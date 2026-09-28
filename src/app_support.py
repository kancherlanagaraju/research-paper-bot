"""Framework-free helpers for the Streamlit app (Milestone 7), kept out of
``app.py`` so they can be unit-tested without a browser or Streamlit.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation import DEFAULT_MODES, MODE_LABELS
from src.rag import RagAnswer

FALLBACK_MODE = "hybrid_reranked"  # config D: the richest pipeline, used until an evaluation has picked a winner


def load_selected_mode(artifacts_dir: Path) -> Optional[str]:
    """Mode chosen by the last evaluation run (``results.json``), if any."""
    path = Path(artifacts_dir) / "evaluation" / "results.json"
    try:
        selected = json.loads(path.read_text())["report"]["selected_mode"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return selected if selected in DEFAULT_MODES else None


def default_mode(artifacts_dir: Path) -> str:
    return load_selected_mode(artifacts_dir) or FALLBACK_MODE


def mode_options(selected: Optional[str] = None) -> Dict[str, str]:
    """{mode: label}; the evaluation-selected mode is flagged."""
    return {m: MODE_LABELS[m] + (" (selected by evaluation)" if m == selected else "") for m in DEFAULT_MODES}


def source_rows(answer: RagAnswer) -> List[Dict[str, Any]]:
    """The top-3 supporting passages, ready to render."""
    return [
        {
            "rank": i,
            "title": c.title,
            "filename": c.filename,
            "page": c.page_number,
            "score": round(c.score, 4),
            "text": c.text,
        }
        for i, c in enumerate(answer.sources, start=1)
    ]


def history_entry(answer: RagAnswer) -> Dict[str, Any]:
    return {
        "query": answer.query,
        "answer": answer.answer,
        "abstained": answer.abstained,
        "mode": answer.retrieval_mode,
        "sources": source_rows(answer),
    }
