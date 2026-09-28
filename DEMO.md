# Demo guide

A 10-minute walkthrough for a mentor review or final demo.

## Before you start (once)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill ZILLIZ_URI, ZILLIZ_TOKEN, OPENAI_API_KEY
python scripts/index_documents.py --embedding-backend both
```

Confirm the live paths work first (never demo untested):

```bash
RUN_LIVE_TESTS=1 python -m pytest -m live -v
python scripts/run_evaluation.py --with-generation --judge   # writes artifacts/evaluation/summary.md
```

Then set `HYBRID_DENSE_BACKEND` in `.env` to the winner named in
`summary.md` if it isn't the default (`oss`).

## Run order

1. **Problem and data (1 min).** Five seminal papers, 95 pages. Show `README.md` dataset table.
2. **Pipeline (2 min).** Show the data-flow diagram in `docs/architecture.md`. Point out per-page chunks with title / filename / page metadata, and one Zilliz collection per embedding model.
3. **Experiments (3 min).** Open `artifacts/evaluation/summary.md`: configs A-D compared on hit@k, MRR and latency, the selection rule, and which config won and why. State the limits (18 answerable questions; phrase-based relevance).
4. **Live app (3 min).** `streamlit run app.py`, then ask the questions below.
5. **Tests and honesty (1 min).** `python -m pytest`; say what was verified live and what wasn't.

## Questions to ask in the app

| Question | What it shows |
|---|---|
| How many attention heads does the base Transformer use? | Precise fact, cited to the Attention paper. |
| What attention mechanisms does Mistral 7B use to speed up inference? | Second paper, multi-fact answer. |
| What are the steps of the RLHF method used to train InstructGPT? | Multi-step answer from several passages. |
| Which of the papers discuss hallucination or truthfulness? | Cross-paper synthesis, sources from more than one paper. |
| What is the capital of France? | Abstention: warning, no sources. |
| What does the DeepSeek-R1 paper report about reinforcement learning? | Abstention on a plausible-sounding paper that isn't indexed. |

For each answer, open the three source expanders and check the cited page
against the passage. Switch the sidebar strategy (A-D) on one question to
show how retrieval changes the sources.

## If something fails during the demo

- **Config error banner:** a key in `.env` is missing; the message names it.
- **Empty or odd answers:** the collection may be empty; re-run the indexing command.
- **Slow first question in mode D:** the cross-encoder loads on first use.
