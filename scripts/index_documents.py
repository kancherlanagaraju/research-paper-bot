#!/usr/bin/env python3
"""Command to (re)build the document index.

1. Extract + chunk all PDFs under DATASET_DIR (src.ingestion).
2. Embed every chunk with one or both configured embedding backends
   (src.embeddings): "oss" (sentence-transformers) and/or "openai".
3. Upsert the chunks + vectors into the corresponding Zilliz Cloud
   Serverless collection (src.vector_store), one collection per embedding
   model so different vector spaces/dimensions never mix.

A chunk manifest (metadata for every chunk, independent of embeddings) is
always written to artifacts/ingestion_manifest.json, so ingestion can be
inspected even when --skip-embeddings is used or the vector store step
fails.

Usage:
    # Ingestion + manifest only, no embeddings/upsert (works with zero credentials):
    python scripts/index_documents.py --skip-embeddings

    # Full pipeline with the open-source embedding model (default):
    python scripts/index_documents.py

    # Full pipeline with OpenAI embeddings instead:
    python scripts/index_documents.py --embedding-backend openai

    # Both, into their own collections:
    python scripts/index_documents.py --embedding-backend both

    # Intentionally drop and recreate the collection(s) first:
    python scripts/index_documents.py --rebuild
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig, ConfigError  # noqa: E402
from src.embeddings import EmbeddingError, get_backend  # noqa: E402
from src.ingestion import run_ingestion  # noqa: E402

logger = logging.getLogger(__name__)


def _write_manifest(result, out_path: Path) -> dict:
    summary = result.summary()
    manifest = {
        "summary": summary,
        "dataset_dir": result.dataset_dir,
        "ingestion_version": result.ingestion_version,
        "chunk_size_tokens": result.chunk_size_tokens,
        "chunk_overlap_tokens": result.chunk_overlap_tokens,
        "tokenizer_backend": result.tokenizer_backend,
        "documents": [doc.to_dict() for doc in result.documents],
        "chunks": [c.to_dict() for c in result.all_chunks()],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary


def _index_with_backend(kind: str, config: AppConfig, chunks, rebuild: bool, batch_size: int) -> dict:
    from src.vector_store import ensure_collection, get_client, rebuild_collection, upsert_chunks, collection_name_for

    t0 = time.time()
    backend = get_backend(kind, config)
    vectors = backend.embed_passages([c.text for c in chunks], batch_size=batch_size)
    embed_seconds = time.time() - t0

    client = get_client(config)
    collection_name = collection_name_for(config.zilliz_collection_prefix, backend.name)
    if rebuild:
        rebuild_collection(client, collection_name, backend.dimension)
    else:
        ensure_collection(client, collection_name, backend.dimension)

    t1 = time.time()
    written = upsert_chunks(client, collection_name, chunks, vectors, embedding_model=backend.name, batch_size=batch_size)
    upsert_seconds = time.time() - t1

    return {
        "backend": kind,
        "model": backend.name,
        "dimension": backend.dimension,
        "collection": collection_name,
        "chunks_embedded": len(vectors),
        "chunks_written": written,
        "embed_seconds": round(embed_seconds, 2),
        "upsert_seconds": round(upsert_seconds, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the Research Paper Answer Bot index.")
    parser.add_argument(
        "--embedding-backend",
        choices=["oss", "openai", "both"],
        default="oss",
        help="Which embedding model(s) to index with (default: oss).",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only run extraction + chunking and write the manifest; skip embeddings and the vector store entirely.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Intentionally drop and recreate the collection(s) first.")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding/upsert batch size (default: 32).")
    parser.add_argument(
        "--out", default="artifacts/ingestion_manifest.json", help="Where to write the chunk manifest JSON."
    )
    args = parser.parse_args()

    try:
        config = AppConfig.from_env()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    result = run_ingestion(config)
    summary = _write_manifest(result, Path(args.out))
    print(json.dumps(summary, indent=2))

    if summary["documents_failed"] or summary["total_page_failures"]:
        print("\nExtraction issues:", file=sys.stderr)
        for doc in result.documents:
            if doc.file_error:
                print(f"  [FILE FAILED] {doc.filename}: {doc.file_error}", file=sys.stderr)
            for pf in doc.page_failures:
                print(f"  [PAGE FAILED] {doc.filename} p{pf.page_number}: {pf.error}", file=sys.stderr)

    print(f"\nWrote manifest with {summary['total_chunks']} chunks to {args.out}")

    if args.skip_embeddings:
        print("\n--skip-embeddings set: stopping after ingestion. No embeddings computed, nothing indexed.")
        return 0

    chunks = result.all_chunks()
    if not chunks:
        print("\nNo chunks to embed/index.", file=sys.stderr)
        return 1

    backends = ["oss", "openai"] if args.embedding_backend == "both" else [args.embedding_backend]
    exit_code = 0
    for kind in backends:
        print(f"\n--- Embedding + indexing with backend='{kind}' ---")
        try:
            report = _index_with_backend(kind, config, chunks, rebuild=args.rebuild, batch_size=args.batch_size)
            print(json.dumps(report, indent=2))
        except (ConfigError, EmbeddingError) as exc:
            print(f"  SKIPPED ({kind}): {exc}", file=sys.stderr)
            exit_code = 1
        except Exception as exc:  # connection errors, etc. -- keep going for 'both'
            logger.exception("Indexing with backend '%s' failed", kind)
            print(f"  FAILED ({kind}): {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
