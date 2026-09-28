# Architecture

A Retrieval-Augmented Generation system over five Generative AI / LLM papers
(Attention Is All You Need, GPT-4, InstructGPT, Gemini, Mistral 7B). The
capstone brief asks not just for a RAG system but for a comparison of
approaches at each stage, so embeddings, retrieval strategy and the final
pipeline are all treated as experiments.

> Verification status: every component is unit-tested against fakes or
> stubs. Nothing beyond the OpenAI embedding call has been run against live
> services (Zilliz, model downloads, an LLM). See README "Live verification".

## Data flow

```mermaid
flowchart LR
    PDF[5 PDFs] --> ING[ingestion<br/>PyMuPDF, per-page chunks]
    ING --> MAN[(ingestion_manifest.json)]
    ING --> EMB{embeddings}
    EMB -->|OSS bge-small| ZO[(Zilliz collection A)]
    EMB -->|OpenAI 3-small| ZB[(Zilliz collection B)]

    Q[Question] --> RET{retrieval mode}
    RET -->|A| DA[dense cosine, OSS]
    RET -->|B| DB[dense cosine, OpenAI]
    RET -->|C| HY[dense + BM25<br/>RRF / weighted fusion]
    RET -->|D| RR[hybrid candidates<br/>cross-encoder rerank]
    ZO -.-> DA
    ZB -.-> DB
    ZO -.-> HY
    HY --> RR
    DA & DB & HY & RR --> CTX[retrieved chunks]
    CTX --> GEN[rag: grounded prompt<br/>gpt-4o-mini]
    GEN --> OUT[answer + inline citations<br/>+ top-3 sources<br/>or abstention]
    OUT --> APP[Streamlit app]

    EV[evaluation<br/>21 questions] -.->|hit@k, MRR, latency<br/>abstention, citations| RET
    EV -.->|selected mode| APP
```

The same flow as text:

1. **Ingest** (`src/ingestion.py`): extract each PDF page, chunk per page
   (never across a page boundary), attach title / filename / page / chunk ID.
2. **Embed + store** (`src/embeddings.py`, `src/vector_store.py`): embed every
   chunk with each backend and upsert into one Zilliz collection per model.
3. **Retrieve** (`src/retrieval.py`): four configurations, below.
4. **Generate** (`src/rag.py`): a grounded prompt over the retrieved chunks;
   the model cites `[Title, p. N]` or replies with an abstention marker.
5. **Evaluate** (`src/evaluation.py`): run configurations A-D over the same
   questions, compare, select one.
6. **Serve** (`app.py`): Streamlit UI using the evaluation-selected mode.

## Retrieval configurations

| Config | Mode | How it works |
|---|---|---|
| A | `dense_oss` | Query embedded with `BAAI/bge-small-en-v1.5`; cosine search in Zilliz. |
| B | `dense_openai` | Query embedded with `text-embedding-3-small`; cosine search in Zilliz. |
| C | `hybrid` | Dense search (backend = `HYBRID_DENSE_BACKEND`) plus BM25 over the same collection's chunk text; each contributes `HYBRID_CANDIDATE_K` hits; fused by RRF (k=60) or min-max-weighted scores (`FUSION_METHOD`). |
| D | `hybrid_reranked` | Config C's candidates rescored by cross-encoder `ms-marco-MiniLM-L-6-v2`, cut to `RERANK_TOP_K`. |

All four run under identical conditions in evaluation (same corpus,
chunking and k). The BM25 corpus is read from the same Zilliz collection the
dense side searches, so both rankers share chunk IDs and fusion can join on
them.

## Requirement traceability

| Capstone requirement | Implementation | Evidence |
|---|---|---|
| PDF loading & indexing | `src/ingestion.py`, `src/embeddings.py`, `src/vector_store.py`, `scripts/index_documents.py` | Unit-tested; OpenAI embedding of all 95 chunks live-verified. The Zilliz write is unverified against a real cluster. |
| Metadata preservation | `src/ingestion.py: ChunkRecord`, `src/vector_store.py` schema | title / filename / page / chunk_id / embedding_model / ingestion_version stored on every row (tests + manifest). |
| Embedding comparison | `src/embeddings.py` (OSS + OpenAI), configs A vs B in `src/evaluation.py` | OpenAI backend live-verified (1536-dim). OSS backend unit-tested only (weights never downloaded). Comparison results pending a real evaluation run. |
| Retrieval strategy comparison | `src/retrieval.py` (configs A-D) | Unit-tested. Comparison results pending a real evaluation run. |
| RAG pipeline + LLM | `src/rag.py` (`answer_question`, OpenAI chat, temperature 0) | Unit-tested with stub LLM/retrieval (`tests/test_rag.py`). No live LLM call yet. |
| Source citation (top 3) | `src/rag.py` (`RagAnswer.sources`, inline `[Title, p. N]`), `app.py` expanders | Unit-tested and headless-app-tested. |
| Test on sample queries | `artifacts/evaluation/questions.json`, `scripts/run_evaluation.py`, `DEMO.md` | 21-question set grounded against the ingested text; harness unit-tested. Live run pending. |
| Stretch goal: Streamlit app | `app.py`, `src/app_support.py` | Headless `AppTest` runs with a stub pipeline (`tests/test_app.py`). |

## Evaluation design

- **Question set:** 21 questions -- 16 factual (all five papers), 2
  cross-paper, 3 unanswerable (used to test abstention).
- **Relevance labels:** a retrieved chunk is relevant if it comes from an
  expected paper and contains an evidence phrase. Text-based labels stay
  valid if chunking changes. Every answerable question was checked to have
  at least one matching chunk in the ingested text.
- **Retrieval metrics** (answerable questions): hit@k, MRR, mean latency.
- **Generation metrics** (`--with-generation`): abstention accuracy, false
  abstain rate, citation validity (cited label was actually retrieved),
  citation hit (a cited source is relevant); with `--judge`, LLM-scored
  correctness and faithfulness.
- **Selection rule:** highest MRR, then hit@k, then lowest latency. C and D
  use the better dense backend of A/B.
- **Limits:** 18 answerable questions is small, so close scores may be noise;
  relevance is phrase-based, not human-labelled; the LLM judge is a rough guide.

## Known design limits

- One chunk per page (every page in this corpus is under the 800-token
  threshold), so retrieval granularity is a page.
- Abstention depends on the model following the marker instruction; weak
  retrieval can still yield a confident answer. Measured, not guaranteed, by
  the evaluation's abstention metrics.
- The app answers each question independently; history is display-only.
- BM25 is built in memory at first use from the collection; fine for 95
  chunks, not designed for large corpora.

## Milestone 1 notes -- ingestion

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

## Milestone 3 notes -- dense retrieval

`retrieve(query, mode, config, top_k)` is the single entry point used by the
RAG pipeline, the evaluation harness and the app. Dense modes embed the
query with the matching backend and run `client.search` (COSINE) against
that backend's own collection, returning `RetrievedChunk` objects carrying
everything a citation needs.

## Milestone 4 notes -- hybrid and reranked retrieval

- BM25 (`rank-bm25`, Okapi) over lowercase alphanumeric tokens; chunks with
  no query-term overlap are dropped from the sparse list.
- Fusion is a documented choice (`FUSION_METHOD`): RRF uses ranks only, so no
  score normalisation across cosine and BM25 is needed; `weighted` min-max
  normalises each list first. Ties break deterministically.
- The cross-encoder replaces the retrieval score with its own relevance
  score; the candidate pool is `max(HYBRID_CANDIDATE_K, top_k)`.

## Milestone 5 notes -- generation

- Passage headers in the prompt are exactly the citation labels, so the model
  copies rather than guesses titles and pages.
- Two abstention layers: no retrieved chunks (no LLM call), or the model
  replying `INSUFFICIENT_EVIDENCE`. Abstentions return no sources.
- `sources` is the top 3 chunks; the model sees all retrieved chunks.

## Milestone 6 notes -- evaluation

See "Evaluation design" above. A configuration that fails (for example, model
weights that cannot be downloaded) is recorded in the report and skipped
rather than aborting the run.

## Milestone 7 notes -- app

The app defaults to the mode in `artifacts/evaluation/results.json`
(`selected_mode`), falling back to hybrid + rerank. Pipeline errors such as a
missing key are shown in the page rather than as a stack trace.
