# Research Paper Answer Bot

A Retrieval-Augmented Generation (RAG) system over five seminal Generative
AI / LLM research papers, built as a capstone project (Analytics Vidhya
GenAI Pinnacle Program). See `Capstone Project - Generative AI Pinnacle.docx.pdf`
for the original brief.

## Progress checklist

- [x] **Milestone 1 -- Repository & dataset assessment, PDF extraction, metadata-preserving chunking, unit tests.**
- [x] **Milestone 2 -- Embeddings (open-source + OpenAI) and Zilliz Cloud Serverless vector store, idempotent upsert, rebuild command.**
- [x] **Milestone 3 -- Dense cosine retrieval, per embedding model (`dense_oss` / `dense_openai`), unit-tested. Not yet run against a real populated collection -- see "Open item" below.**
- [x] **Milestone 4 -- Hybrid (dense + BM25, RRF/weighted fusion) retrieval (`hybrid`) and cross-encoder reranking (`hybrid_reranked`), unit-tested. Not yet run against a real populated collection or with real model weights -- see "Open item" below.**
- [x] **Milestone 5 -- RAG generation (`src/rag.py`) with top-3 source citations and abstention, unit-tested with a stub LLM. No real LLM call made yet.**
- [x] **Milestone 6 -- Evaluation harness (`src/evaluation.py`, `scripts/run_evaluation.py`): 21-question set, hit@k / MRR / latency for configs A-D, optional generation + LLM-judge metrics, automatic selection. Unit-tested with stubs; NOT yet run on real data, so no results or final selection exist yet.**
- [x] **Milestone 7 -- Streamlit app (`app.py`): question box, answer with citations, top-3 source expanders, retrieval-mode selector, "insufficient evidence" warning, session history. Tested headless with a stubbed pipeline; not yet used against live services.**
- [x] **Milestone 8 -- Documentation (README, `docs/architecture.md` with data-flow diagram and requirement traceability, `.env.example`, `DEMO.md`), 106 unit tests plus opt-in live smoke tests.**

**Verification status:** all seven pipeline milestones are code-complete and
unit-tested, but Milestones 3-7 are tested against fakes/stubs only. Nothing
beyond the OpenAI embedding call has been run against live services (Zilliz,
open-source model weights, an LLM), and no evaluation has been run, so there
are no results and no selected configuration yet. See "Live verification"
below for the exact steps to close that gap.

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
retrieval, 12 RAG, 13 evaluation, 8 app), all against fakes or stubs. Live
smoke tests are skipped by default and run with
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
| `HYBRID_DENSE_BACKEND` | `oss` | Dense side of hybrid modes: `oss` or `openai`. Set to the evaluation winner. |
| `FUSION_METHOD` | `rrf` | `rrf` or `weighted`; weights via `FUSION_WEIGHT_DENSE` / `FUSION_WEIGHT_SPARSE`. |
| `HYBRID_CANDIDATE_K` / `RERANK_TOP_K` | `20` / `5` | Candidates per ranker before fusion or reranking / final results for mode D. |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for mode D. |
| `LLM_MODEL` | `gpt-4o-mini` | Generation model (`LLM_PROVIDER=openai` only). |
| `DATASET_DIR` | `./pinnacle_capstone_data` | Where PDFs are discovered (recursive). |
| `CHUNK_SIZE_TOKENS` | `800` | Target chunk size. |
| `CHUNK_OVERLAP_TOKENS` | `120` | Overlap between consecutive chunks on the same page. |
| `INGESTION_VERSION` | `v1` | Baked into every chunk ID; bump to force new IDs on a rebuild. |
| `TOKENIZER_BACKEND` | `approx_word` | `approx_word` (offline, deterministic) or `tiktoken` (exact GPT tokens, needs network on first use). |

## Known limitations (sandbox network blocks found during development)

Development happens inside a sandboxed environment whose outbound network
access is restricted to an allowlist of domains. This surfaced three
specific blocks, all at the network/TLS layer (not credential problems):

| Host | Needed for | Status here |
|---|---|---|
| `api.openai.com` | OpenAI embeddings, future LLM calls | **Reachable.** Live-tested: real API key verified (200 OK), real embeddings computed for a sample and for the full 95-chunk corpus (`text-embedding-3-small`, 1536-dim). |
| `huggingface.co` (API/metadata) | Resolving open-source model info | Reachable. |
| `*.cdn.hf.co` / `cas-server.xethub.hf.co` | Downloading actual model weights (e.g. `BAAI/bge-small-en-v1.5`) | **Blocked** (SSL handshake fails). The OSS embedding backend's code is complete and unit-tested with a stubbed model, but the real multi-hundred-MB weight download could not be completed here. |
| `*.cloud.zilliz.com` | The Zilliz Cloud Serverless cluster itself | **Blocked** (SSL handshake fails). Schema/index construction was verified directly against real `pymilvus` classes (no network needed for that); upsert/rebuild orchestration is unit-tested against an in-memory fake client. The actual write to your cluster has not been verified end-to-end. |
| `openaipublic.blob.core.windows.net` | `tiktoken`'s vocabulary file | Blocked (this is why `TOKENIZER_BACKEND` defaults to the offline `approx_word` tokenizer -- see Milestone 1 notes). |

**What this means practically:** `python scripts/index_documents.py` (with
`--embedding-backend openai`) ran the full pipeline through embedding
successfully here, then failed at the Zilliz connection step with a clean,
caught error (not a crash) -- confirmed by an actual run. To finish
verifying Milestone 2 end-to-end, run that same command yourself (or from
any machine with normal internet access to Hugging Face and Zilliz Cloud)
and share the output; I'll fix anything that comes up.

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
  against a fake client (real-cluster verification still pending, see
  "Known limitations").
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
   here, with real-cluster verification deferred to an environment with
   network access to Zilliz -- confirmed.

## Live verification (the remaining open item)

Everything below the OpenAI embedding call has only been exercised against
fakes. To verify for real, from a machine with normal internet access:

```bash
python scripts/index_documents.py --embedding-backend both   # ingest, embed, upsert both collections
python -m src.retrieval "What is attention in a transformer model?"
RUN_LIVE_TESTS=1 python -m pytest -m live -v                 # all four modes + a real RAG answer + abstention
python scripts/run_evaluation.py --with-generation --judge   # the real comparison and selection
streamlit run app.py
```

What could still fail, and where: the Zilliz write and the BM25 corpus fetch
(`client.query` with a `chunk_id != ""` filter and pagination) have only run
against a fake client; the open-source embedding and cross-encoder weights
have never been downloaded; the LLM has never been called, so citation
format and abstention behaviour are untested against a real model.

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
artifacts/            ingestion_manifest.json; evaluation/questions.json (results appear after a run)
docs/architecture.md  Data-flow diagram, retrieval configs, requirement traceability
DEMO.md               Demo walkthrough and sample questions
.env.example          Every setting, documented
```
