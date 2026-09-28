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
- [ ] Milestone 5 -- RAG generation with citations and abstention.
- [ ] Milestone 6 -- Evaluation across all retrieval configurations + final selection.
- [ ] Milestone 7 -- Streamlit application (stretch goal).
- [ ] Milestone 8 -- Full documentation, complete test suite, demo prep.

Milestones 5-8 are stubbed (see `src/*.py` docstrings) but not implemented.
Milestones 3-4 are unit-tested against fakes/stubs only; nothing beyond the
OpenAI embedding call has been verified against live services.

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

# Everything needed for Milestone 1 (ingestion + tests):
pip install pymupdf==1.28.2 python-dotenv==1.2.3 pytest==9.1.1 tiktoken==0.14.0

# Or the full pinned set for later milestones (heavier: sentence-transformers,
# pymilvus, streamlit, ...):
pip install -r requirements.txt

cp .env.example .env   # fill in credentials as later milestones need them
```

No credentials are required for Milestone 1. `.env.example` documents every
variable the full project will eventually need (Zilliz, OpenAI, retrieval,
LLM) so nothing is a surprise later.

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

Run the unit tests:

```bash
python -m pytest -v
```

Last run (confirmed on the user's own machine, real Zilliz connection):
**61 passed, 0 failed** (28 ingestion + 11 embeddings + 12 vector store + 10
retrieval).

Inspect things directly without the full CLI:

```bash
python -m src.ingestion                      # ingestion summary for the configured dataset
python -m src.embeddings                     # embeds one sample query with each configured backend
python -m src.retrieval "your question here" # top-3 hits for dense_oss and dense_openai
```

## Configuration

All settings live in `src/config.py` (`AppConfig`), loaded from environment
variables / a local `.env` file. See `.env.example` for the full list with
comments on which milestone needs each one. Nothing required by a
not-yet-implemented milestone is validated eagerly -- e.g. `OPENAI_API_KEY`
and `ZILLIZ_URI`/`ZILLIZ_TOKEN` are only checked when a component that
actually uses them is called (`AppConfig.require_openai_key()` /
`AppConfig.require_zilliz_credentials()`), with an actionable error message.

Key ingestion settings:

| Variable | Default | Meaning |
|---|---|---|
| `DATASET_DIR` | `./pinnacle_capstone_data` | Where PDFs are discovered (recursive). |
| `CHUNK_SIZE_TOKENS` | `800` | Target chunk size. |
| `CHUNK_OVERLAP_TOKENS` | `120` | Overlap between consecutive chunks on the same page. |
| `INGESTION_VERSION` | `v1` | Baked into every chunk ID; bump to force new IDs on a rebuild. |
| `TOKENIZER_BACKEND` | `approx_word` | `approx_word` (offline, deterministic) or `tiktoken` (exact GPT tokens, needs network on first use). |

## Known limitations (found while building Milestone 2)

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

## Open item before Milestone 4

**Real end-to-end verification of indexing + retrieval.** Run the full
pipeline, then try a real query against it:

```bash
python scripts/index_documents.py                 # or --embedding-backend both
python -m src.retrieval "What is attention in a transformer model?"
```

The second command prints the top-3 hits (score, title, page, text preview)
for both `dense_oss` and `dense_openai` modes. Share the output -- that's
the one thing that couldn't be verified from the sandboxed environment this
was developed in (see "Known limitations" above). Everything else needed
for Milestones 2-3 is done and unit-tested (61 tests passing).

## Project structure

```
src/config.py        Central configuration (implemented)
src/ingestion.py      PDF extraction + chunking (implemented)
src/embeddings.py     Open-source + OpenAI embedding backends (implemented)
src/vector_store.py   Zilliz Cloud Serverless client, idempotent upsert, rebuild (implemented)
src/retrieval.py      Dense cosine, hybrid BM25/RRF, and cross-encoder reranked retrieval (implemented)
src/rag.py            Generation with citations + abstention (stub -- Milestone 5)
src/evaluation.py     Evaluation harness (stub -- Milestone 6)
scripts/index_documents.py   Full ingest -> embed -> upsert CLI (implemented)
scripts/run_evaluation.py    Evaluation CLI (stub)
app.py                Streamlit app (stub -- Milestone 7)
tests/test_ingestion.py      28 unit tests (Milestone 1)
tests/test_embeddings.py     11 unit tests (Milestone 2)
tests/test_vector_store.py   12 unit tests (Milestone 2)
tests/test_retrieval.py      10 unit tests (Milestone 3)
artifacts/evaluation/        Evaluation outputs land here (empty for now)
docs/architecture.md         Architecture + requirement-traceability table (skeleton)
```
