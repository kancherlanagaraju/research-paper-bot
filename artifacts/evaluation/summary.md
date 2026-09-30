# Evaluation summary

21 questions (18 answerable), top_k = 5, same corpus and chunking for every configuration.

## Retrieval (answerable questions, k=5)

| Configuration | hit@5 | MRR | mean latency (s) |
|---|---|---|---|
| A (OSS dense) | 0.944 | 0.792 | 3.172 |
| B (OpenAI dense) | 0.944 | 0.889 | 0.573 |
| C (hybrid) | 0.944 | 0.944 | 0.452 |
| D (hybrid + rerank) | 0.944 | 0.852 | 0.750 |

## Generation

| Configuration | abstention acc. | false abstain | citation valid | citation hit | correctness | faithfulness | e2e latency (s) |
|---|---|---|---|---|---|---|---|
| A (OSS dense) | 1.000 | 0.000 | 0.556 | 0.444 | 1.000 | 1.000 | 3.331 |
| B (OpenAI dense) | 1.000 | 0.000 | 0.722 | 0.667 | 0.944 | 0.961 | 1.989 |
| C (hybrid) | 1.000 | 0.000 | 0.611 | 0.611 | 0.989 | 0.994 | 1.868 |
| D (hybrid + rerank) | 1.000 | 0.000 | 0.583 | 0.500 | 1.000 | 1.000 | 2.331 |

## Selected configuration

**C (hybrid)** (`hybrid`): highest MRR (0.944) with hit@5 0.944. Rule: highest MRR, then hit@k, then lowest latency.
Hybrid configurations used the `openai` dense backend (the better of A/B).
