# Research Paper Answer Bot

A Retrieval-Augmented Generation (RAG) system over five seminal Generative
AI / LLM research papers, built as a capstone project (Analytics Vidhya
GenAI Pinnacle Program). See `Capstone Project - Generative AI Pinnacle.docx.pdf`
for the original brief.

## Progress checklist

- [x] **Milestone 1 -- Repository & dataset assessment, PDF extraction, metadata-preserving chunking, unit tests.**
- [x] **Milestone 2 -- Embeddings (open-source + OpenAI) and Zilliz Cloud Serverless vector store, idempotent upsert, rebuild command.**
- [x] **Milestone 3 -- Dense cosine retrieval, per embedding model (`dense_oss` / `dense_openai`), unit-tested and exercised against the populated live collections in the latest evaluation.**
- [x] **Milestone 4 -- Hybrid (dense + BM25, RRF/weighted fusion) retrieval (`hybrid`) and cross-encoder reranking (`hybrid_reranked`), unit-tested and exercised in the latest evaluation.**
- [x] **Milestone 5 -- RAG generation (`src/rag.py`) with top-3 sources and abstention, unit-tested with a stub LLM and evaluated with live generation and judging.**
- [x] **Milestone 6 -- Evaluation harness (`src/evaluation.py`, `scripts/run_evaluation.py`): 21-question set, retrieval and generation metrics for configs A-D, LLM-judge metrics, automatic selection. Latest results are in `artifacts/evaluation/`.**
- [x] **Milestone 7 -- Streamlit app (`app.py`): question box, answer with citations, top-3 source expanders, retrieval-mode selector, "insufficient evidence" warning, session history. Tested headless with a stubbed pipeline; run the app against your services before presenting it live.**
- [x] **Milestone 8 -- Documentation (README, `docs/architecture.md` with data-flow diagram and requirement traceability, `.env.example`, `DEMO.md`), 106 unit tests plus opt-in live smoke tests.**

**Verification status:** the latest evaluation completed all four retrieval
modes with live services and included live generation and LLM judging; no
configuration errors were recorded. The selected mode is `hybrid`, using
OpenAI embeddings for its dense side. The Streamlit UI itself has only been
tested headlessly with a stubbed pipeline. Citation quality is imperfect:
the selected mode's citation-validity and citation-hit scores are both
0.611, so inspect cited passages during the demo.

Where to look: `docs/architecture.md` (diagram, retrieval configs,
requirement traceability), `DEMO.md` (demo script), `.env.example` (all settings).

## Dataset

Five extractable, non-scanned, English-language PDFs under
`pinnacle_capstone_data/` (no duplicates, no damaged files, no empty pages):

| File | Detected title | Pages | Size |
|---|---|---|---|
| `attention_paper.pdf` | Attention Is All You Need | 15 | 2.2 MB |
| `gemini_paper.pdf` | Gemini: A Family of Highly Capable Multimodal Models | 40 | 14.8 MB |
| `gpt4.pdf` | GPT-4 Technical Report | 14 | 0.65 MB |
| `instructgpt.pdf` | Training language models to follow instructions with human feedback | 20 | 0.81 MB |
| `mistral_paper.pdf` | Mistral 7B | 6 | 3.3 MB |

## Setup

```bash
cd research-paper-bot
python3 -m venv .venv && source .venv/bin/activate

# Full pinned set (heavier: sentence-transformers, pymilvus, streamlit, ...):
pip install -r requirements.txt

# Ingestion + most unit tests need far less:
pip install pymupdf python-dotenv pytest tiktoken rank-bm25 numpy

cp .env.example .env   # then fill in credentials
```

Ingestion and the unit tests need no credentials. Indexing and retrieval
need `ZILLIZ_URI` / `ZILLIZ_TOKEN`; OpenAI embeddings, generation and the
LLM judge need `OPENAI_API_KEY`. Each is checked only when used.

## Commands

Ingestion only (no embeddings, no vector store -- works with zero credentials):

```bash
python scripts/index_documents.py --skip-embeddings
```

Last run: 5/5 documents ingested successfully, 0 file failures, 0 page
failures, 95 pages -> 95 chunks (one chunk per page at the default 800-token
threshold; every page in this dataset is shorter than that).

Full pipeline -- ingest, embed, and upsert into Zilliz Cloud Serverless:

```bash
python scripts/index_documents.py                          # open-source embeddings (default)
python scripts/index_documents.py --embedding-backend openai
python scripts/index_documents.py --embedding-backend both  # both, into separate collections
python scripts/index_documents.py --rebuild                 # intentionally drop + recreate first
```

Ask a question from the command line, or launch the app:

```bash
python -m src.rag "How many attention heads does the base Transformer use?" [mode]
streamlit run app.py
```

Compare the four retrieval configurations and pick one:

```bash
python scripts/run_evaluation.py                              # retrieval metrics only
python scripts/run_evaluation.py --with-generation --judge    # also abstention, citations, LLM-judged quality
```

Results land in `artifacts/evaluation/` (`results.json`, `results.csv`,
`summary.md`). The app then defaults to the winning configuration.

Run the unit tests:

```bash
python -m pytest -v
```

**106 unit tests pass** (28 ingestion, 11 embeddings, 12 vector store, 22
retrieval, 12 RAG, 13 evaluation, 8 app), primarily against fakes or stubs.
Live smoke tests are skipped by default and run with
`RUN_LIVE_TESTS=1 python -m pytest -m live -v`.

Inspect things directly without the full CLI:

```bash
python -m src.ingestion                      # ingestion summary for the configured dataset
python -m src.embeddings                     # embeds one sample query with each configured backend
python -m src.retrieval "your question here" # top-3 hits for all four retrieval modes
```

Retrieval modes: `dense_oss` (A), `dense_openai` (B), `hybrid` (C),
`hybrid_reranked` (D). See `docs/architecture.md`.

## Configuration

All settings live in `src/config.py` (`AppConfig`), loaded from environment
variables / a local `.env` file. See `.env.example` for the full list with
comments on which milestone needs each one. Nothing required by a
not-yet-implemented milestone is validated eagerly -- e.g. `OPENAI_API_KEY`
and `ZILLIZ_URI`/`ZILLIZ_TOKEN` are only checked when a component that
actually uses them is called (`AppConfig.require_openai_key()` /
`AppConfig.require_zilliz_credentials()`), with an actionable error message.

Key settings (see `.env.example` for all of them):

| Variable | Default | Meaning |
|---|---|---|
| `HYBRID_DENSE_BACKEND` | `openai` | Dense side of hybrid modes: `oss` or `openai`; `openai` was the better dense backend in the latest evaluation. |
| `FUSION_METHOD` | `rrf` | `rrf` or `weighted`; weights via `FUSION_WEIGHT_DENSE` / `FUSION_WEIGHT_SPARSE`. |
| `HYBRID_CANDIDATE_K` / `RERANK_TOP_K` | `20` / `5` | Candidates per ranker before fusion or reranking / final results for mode D. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for mode D. |
| `LLM_MODEL` | `gpt-4o-mini` | Generation model (`LLM_PROVIDER=openai` only). |
| `DATASET_DIR` | `./pinnacle_capstone_data` | Where PDFs are discovered (recursive). |
| `CHUNK_SIZE_TOKENS` | `800` | Target chunk size. |
| `CHUNK_OVERLAP_TOKENS` | `120` | Overlap between consecutive chunks on the same page. |
| `INGESTION_VERSION` | `v1` | Baked into every chunk ID; bump to force new IDs on a rebuild. |
| `TOKENIZER_BACKEND` | `approx_word` | `approx_word` (offline, deterministic) or `tiktoken` (exact GPT tokens, needs network on first use). |

## Live evaluation and limitations

The latest `artifacts/evaluation/summary.md` records a successful evaluation
of all four retrieval modes over 21 questions (18 answerable), with live
generation and LLM judging and no mode errors. Retrieval results were:

| Configuration | hit@5 | MRR | Mean retrieval latency |
|---|---:|---:|---:|
| A (OSS dense) | 0.944 | 0.792 | 3.172 s |
| B (OpenAI dense) | 0.944 | 0.889 | 0.573 s |
| C (hybrid) | 0.944 | 0.944 | 0.452 s |
| D (hybrid + rerank) | 0.944 | 0.852 | 0.750 s |

The selection rule chooses the highest MRR, then hit@k, then lowest latency.
It selected **C (`hybrid`)**, with the hybrid dense backend set to **OpenAI**.
Generation abstention accuracy was 1.000 for each mode. For the selected
mode, citation validity and citation hit were both 0.611: citations are not
fully reliable and should be checked against the displayed source passages.
The evaluation set is small and uses phrase-based relevance labels; LLM-judge
scores are approximate rather than human-verified.

To reproduce the evaluation with your own configured services:

```bash
python scripts/run_evaluation.py --with-generation --judge
```

The initial development environment had network restrictions that prevented
some live checks. The later successful evaluation verifies the live
retrieval, generation, and judge paths represented by its results. The
Streamlit application has only been tested headlessly with a stubbed
pipeline, so launch it and verify it against your services before presenting.

## Key design decisions (Milestone 2)

- **One Zilliz collection per embedding model**, named
  `{ZILLIZ_COLLECTION_PREFIX}__{slugified_model_name}` (e.g.
  `res_bot__baai_bge_small_en_v1_5` vs. `res_bot__text_embedding_3_small`).
  Different models never share a collection, so dimension mismatches are
  impossible by construction, and the two embedding approaches (Milestone
  3/4 comparison) stay cleanly separated.
- **Vector index metric is `COSINE`** directly (Milvus/Zilliz support it
  natively), so retrieval configuration A ("dense cosine similarity") needs
  no manual vector normalization step at query time. `AUTOINDEX` is used for
  the index type, which Zilliz Cloud Serverless manages for you.
- **Upsert, not insert**, keyed by the same deterministic `chunk_id` from
  Milestone 1. Re-running indexing on unchanged files overwrites the same
  rows rather than duplicating them -- verified in `tests/test_vector_store.py`
  against a fake client. The latest live evaluation retrieved from populated
  Zilliz collections, but did not separately exercise indexing/upsert or
  rebuild against the live cluster.
- **`--rebuild` is opt-in and explicit** (drops + recreates the collection);
  the default behavior only creates a collection if it doesn't already
  exist, and otherwise upserts into what's there.

## Key design decisions (Milestone 1)

- **Chunking never spans a page boundary.** Each page is windowed
  independently into ~800-token chunks with 120-token overlap. This keeps
  every chunk's page citation exact and simple, at the cost of some
  shorter-than-800-token chunks on short pages (e.g. abstracts). This is a
  reasonable default, not a hard requirement from the brief -- flag if you'd
  prefer chunks to flow across page breaks instead.
- **Default tokenizer is offline and word-based (`approx_word`), not
  `tiktoken`.** `tiktoken` needs to download its vocabulary file from
  `openaipublic.blob.core.windows.net` on first use; in this sandboxed
  environment that download failed (`SSLError`), and relying on it by
  default would make chunk boundaries depend on whichever machine happens
  to have network access at ingestion time -- breaking reproducibility.
  `TOKENIZER_BACKEND=tiktoken` is available as an explicit, fail-loud opt-in
  for exact OpenAI token counts once network access is confirmed. Because
  `approx_word` counts whitespace-delimited words rather than GPT subword
  tokens, actual chunks run a bit larger in real-token terms than "800"
  suggests (words undercount tokens by roughly 30%); this is documented,
  not hidden.
- **Chunk IDs are deterministic and content-independent**:
  `sha256(filename | ingestion_version | page | in-page index)`. Re-running
  ingestion on unchanged files reproduces identical IDs every time (verified
  in tests), which is what will make the Milestone 2 vector-store write an
  idempotent upsert rather than an insert. A separate `content_hash` field
  captures the chunk's actual text, for future change detection.
- **Title extraction** uses the largest-font horizontal text on page 1,
  excluding arXiv's rotated watermark text (which is often a larger font
  than the real title). Verified correct on all 5 sample papers; falls back
  to a filename-derived title if the heuristic finds nothing.

## Decisions already made (with your approval)

1. Per-page chunking (never spans a page boundary) -- confirmed.
2. `approx_word` as the default tokenizer (offline/reproducible) -- confirmed.
3. Proceed with Milestone 2 by building against a mocked vector-store client
   here -- confirmed. The later live evaluation verified retrieval from
   populated Zilliz collections; direct indexing/upsert and rebuild against
   the live cluster remain unverified.

## Live smoke tests

```bash
python scripts/index_documents.py --embedding-backend both   # ingest, embed, upsert both collections
python -m src.retrieval "What is attention in a transformer model?"
RUN_LIVE_TESTS=1 python -m pytest -m live -v                 # all four modes + a real RAG answer + abstention
python scripts/run_evaluation.py --with-generation --judge   # the real comparison and selection
streamlit run app.py
```

The evaluation artifacts document a successful live comparison. Run the
opt-in smoke-test command above as a separate check when setting up another
environment; its tests are skipped by default.

## Project structure

```
src/config.py         Central configuration
src/ingestion.py      PDF extraction + per-page chunking
src/embeddings.py     Open-source + OpenAI embedding backends
src/vector_store.py   Zilliz Cloud Serverless client, idempotent upsert, rebuild
src/retrieval.py      Dense cosine, hybrid BM25/RRF, cross-encoder reranked retrieval
src/rag.py            Grounded generation with top-3 citations + abstention
src/evaluation.py     Metrics, configuration comparison, selection
src/app_support.py    Framework-free helpers for the app
app.py                Streamlit app
scripts/index_documents.py   Ingest -> embed -> upsert CLI
scripts/run_evaluation.py    Evaluation CLI
tests/                106 unit tests + tests/test_live.py (opt-in live smoke tests)
artifacts/            ingestion_manifest.json; evaluation question set and latest results
docs/architecture.md  Data-flow diagram, retrieval configs, requirement traceability
DEMO.md               Demo walkthrough and sample questions
.env.example          Every setting, documented
```
