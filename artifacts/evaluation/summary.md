# Evaluation summary

21 questions (18 answerable), top_k = 5, same corpus and chunking for every configuration.

## Retrieval (answerable questions, k=5)

| Configuration | hit@5 | MRR | mean latency (s) |
|---|---|---|---|
| A (OSS dense) | 0.944 | 0.792 | 2.021 |
| C (hybrid) | 0.944 | 0.815 | 1.733 |
| D (hybrid + rerank) | 0.944 | 0.852 | 2.363 |

## Selected configuration

**D (hybrid + rerank)** (`hybrid_reranked`): highest MRR (0.852) with hit@5 0.944. Rule: highest MRR, then hit@k, then lowest latency.
Hybrid configurations used the `oss` dense backend (the better of A/B).
