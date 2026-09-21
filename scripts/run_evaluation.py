#!/usr/bin/env python3
"""Command to run the evaluation suite -- NOT YET IMPLEMENTED (Milestone 7).

Will run all retrieval configurations against the fixed evaluation question
set and write raw results + a Markdown comparison summary under
artifacts/evaluation/.
"""
from __future__ import annotations

import sys


def main() -> int:
    print(
        "Evaluation is implemented in Milestone 7 (after retrieval strategies and "
        "RAG generation exist). Nothing to run yet.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
