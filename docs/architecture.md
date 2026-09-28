# Architecture

> Status: skeleton only. This document will be filled in as each milestone
> lands (see the progress checklist in `README.md`). It will end with a
> diagram and a table mapping every capstone requirement to its
> implementation and evidence, per the project brief.

## Overview (to be completed)

RAG system over the five supplied Generative AI / LLM papers: PDF ingestion
-> chunking -> dual embeddings (open-source + OpenAI) -> Zilliz Cloud
Serverless vector store -> retrieval experiments (dense / hybrid / reranked)
-> RAG generation with citations -> evaluation -> Streamlit app.

## Component map (to be completed)

| Capstone requirement | Implementation | Evidence |
|---|---|---|
| PDF loading & indexing | `src/ingestion.py`, `src/embeddings.py`, `src/vector_store.py`, `scripts/index_documents.py` | 28+11+12 passing tests; live run: 95/95 chunks embedded via OpenAI and manifest written. Zilliz write pending real-cluster verification (see README "Known limitations"). |
| Metadata preservation | `src/ingestion.py: ChunkRecord`, `src/vector_store.py` schema | Verified: title/filename/page/chunk_id/embedding_model/ingestion_version all present on every stored row. |
| Embedding comparison | `src/embeddings.py` (`OSSEmbeddingBackend`, `OpenAIEmbeddingBackend`) | OpenAI backend live-verified (real API, 1536-dim). OSS backend implemented + unit-tested; live HF download blocked in this sandbox, pending verification. |
| Retrieval strategy comparison | `src/retrieval.py` | Dense cosine (configs A/B) implemented + unit-tested; hybrid (C) and reranked (D) implemented + unit-tested; not yet run live |
| RAG pipeline + LLM | `src/rag.py` (`answer_question`, OpenAI chat, temperature 0) | Implemented; unit-tested with stub LLM/retrieval (tests/test_rag.py). No live LLM call yet. |
| Source citation (top 3) | `src/rag.py` (`RagAnswer.sources`, inline `[Title, p. N]` citations, abstention) | Implemented; unit-tested. |
| Evaluation | `src/evaluation.py`, `artifacts/evaluation/` | TBD (Milestone 6) |
| Streamlit app | `app.py` | TBD (Milestone 7) |

## Data flow diagram (to be completed)

Will be added once the vector store and retrieval milestones are implemented.

## Milestone 1 notes (current)

Ingestion (`src/ingestion.py`) is implemented and unit-tested. See
`README.md` for exact commands and results. Key decisions:

- Chunking is done **per page** (chunks never span a page boundary), so
  every chunk's page citation is exact. Short pages simply produce a single
  (sub-800-token) chunk.
- Chunk IDs are deterministic and derived from `(filename, ingestion
  version, page number, in-page chunk index)` -- not from chunk text -- so
  re-running ingestion reproduces identical IDs (idempotent upserts later).
- Chunk "tokens" default to a dependency-free, offline word/punctuation
  tokenizer (`TOKENIZER_BACKEND=approx_word`) rather than `tiktoken`, so
  chunk boundaries never depend on network availability at ingestion time.
  `TOKENIZER_BACKEND=tiktoken` is available as an explicit opt-in for exact
  OpenAI token counts.

## Milestone 2 notes

Embeddings (`src/embeddings.py`) and the Zilliz Cloud Serverless vector
store (`src/vector_store.py`) are implemented and unit-tested. Key
decisions:

- One collection per embedding model
  (`{ZILLIZ_COLLECTION_PREFIX}__{slugified_model_name}`), so different
  models/dimensions never collide.
- `COSINE` metric on the vector index directly -- no manual normalization
  needed for cosine retrieval (Milestone 3).
- `upsert` keyed by `chunk_id` (idempotent); `--rebuild` explicitly drops
  and recreates a collection.
- OpenAI embeddings were live-verified end-to-end against the real API.
  Open-source embeddings (sentence-transformers) and the live Zilliz
  connection could not be verified from the sandboxed development
  environment (outbound network to Hugging Face's model-weight CDN and to
  `*.cloud.zilliz.com` was blocked there) -- see README "Known limitations"
  for the exact hosts and what's pending.
