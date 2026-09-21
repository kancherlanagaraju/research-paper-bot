"""Retrieval strategies.

Milestone 3 (implemented): dense cosine retrieval, one path per embedding
backend, so configuration A (open-source dense + cosine) and configuration
B (OpenAI dense + cosine) can be run against the same query set and
compared later under identical conditions (same dataset, chunking, and
top_k) per the brief.

Milestone 4 (not yet implemented): configuration C (best dense model + BM25
sparse hybrid, fused via a documented method -- RRF or weighted, both
configurable via ``AppConfig.fusion_method`` / ``fusion_weight_dense`` /
``fusion_weight_sparse``) and configuration D (hybrid retrieval of the top
``AppConfig.hybrid_candidate_k`` candidates, reranked down to
``AppConfig.rerank_top_k`` with ``AppConfig.reranker_model``).

NOTE ON VERIFICATION: `dense_search`'s orchestration (the exact `client.search`
call it makes) is unit-tested against a fake client
(tests/test_retrieval.py). Actually retrieving from a populated Zilliz
collection depends on Milestone 2's indexing having been run for real first
-- see README "Known limitations" for how that's been verified so far.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.config import AppConfig
from src.embeddings import get_backend
from src.vector_store import collection_name_for, get_client

logger = logging.getLogger(__name__)

# Every chunk metadata field we want back alongside each hit, so a result
# can be cited (title, filename, page) without a second lookup.
OUTPUT_FIELDS = [
    "doc_id",
    "title",
    "filename",
    "relative_path",
    "page_number",
    "page_end",
    "chunk_index",
    "chunk_index_in_page",
    "text",
    "token_count",
    "char_count",
    "content_hash",
    "tokenizer_backend",
    "embedding_model",
    "ingestion_version",
]


@dataclass(frozen=True)
class RetrievedChunk:
    """One search hit, flattened for easy use by rag.py / evaluation.py /
    the Streamlit app -- all the fields a citation needs, plus the raw
    similarity score.
    """

    chunk_id: str
    score: float  # COSINE metric: higher = more similar
    title: str
    filename: str
    page_number: int
    text: str
    doc_id: str
    embedding_model: str

    @classmethod
    def from_hit(cls, hit: Dict[str, Any]) -> "RetrievedChunk":
        # pymilvus MilvusClient.search() hits look like:
        # {"id": <primary key>, "distance": <score>, "entity": {<output fields>}}
        entity = hit.get("entity", {}) if "entity" in hit else hit
        return cls(
            chunk_id=hit.get("id") or entity.get("chunk_id", ""),
            score=float(hit.get("distance", 0.0)),
            title=entity.get("title", ""),
            filename=entity.get("filename", ""),
            page_number=int(entity.get("page_number", 0) or 0),
            text=entity.get("text", ""),
            doc_id=entity.get("doc_id", ""),
            embedding_model=entity.get("embedding_model", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "title": self.title,
            "filename": self.filename,
            "page_number": self.page_number,
            "text": self.text,
            "doc_id": self.doc_id,
            "embedding_model": self.embedding_model,
        }


def dense_search(client: Any, collection_name: str, query_vector: List[float], top_k: int) -> List[RetrievedChunk]:
    """Plain dense-cosine search against one collection.

    ``client`` is injected (a real `pymilvus.MilvusClient`, or a fake with
    the same `.search()` shape) so this is unit-testable without a live
    cluster.
    """
    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        anns_field="vector",
        limit=top_k,
        search_params={"metric_type": "COSINE"},
        output_fields=OUTPUT_FIELDS,
    )
    hits = results[0] if results else []
    return [RetrievedChunk.from_hit(hit) for hit in hits]


def retrieve_dense(
    query: str, embedding_kind: str, config: AppConfig, top_k: Optional[int] = None
) -> List[RetrievedChunk]:
    """Configuration A ("oss") or B ("openai"): embed ``query`` with the
    given backend and run a plain dense cosine search against that
    backend's own collection (see `src.vector_store.collection_name_for`).
    """
    backend = get_backend(embedding_kind, config)
    query_vector = backend.embed_query(query)
    collection_name = collection_name_for(config.zilliz_collection_prefix, backend.name)
    client = get_client(config)
    k = top_k if top_k is not None else config.top_k_dense
    logger.info("Dense search: backend=%s collection=%s top_k=%d", embedding_kind, collection_name, k)
    return dense_search(client, collection_name, query_vector, top_k=k)


_MODE_TO_EMBEDDING_KIND = {
    "dense_oss": "oss",
    "dense_openai": "openai",
}


def retrieve(query: str, mode: str = "dense_oss", config: Optional[AppConfig] = None, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    """Mode dispatcher used by rag.py / evaluation.py / the Streamlit app.

    Supported now: "dense_oss", "dense_openai" (Milestone 3).
    Not yet supported: "hybrid", "hybrid_reranked" (Milestone 4).
    """
    config = config or AppConfig.from_env()
    if mode in _MODE_TO_EMBEDDING_KIND:
        return retrieve_dense(query, _MODE_TO_EMBEDDING_KIND[mode], config, top_k)
    raise NotImplementedError(
        f"Retrieval mode '{mode}' is not implemented yet. Available now: "
        f"{sorted(_MODE_TO_EMBEDDING_KIND)}. Hybrid/reranked modes land in Milestone 4."
    )


if __name__ == "__main__":
    # Manual live check: `python -m src.retrieval "your question here"`
    import sys

    cfg = AppConfig.from_env()
    question = " ".join(sys.argv[1:]) or "What is attention in a transformer model?"
    for mode in ("dense_oss", "dense_openai"):
        print(f"\n--- mode={mode} ---")
        try:
            for chunk in retrieve(question, mode=mode, config=cfg, top_k=3):
                print(f"  [{chunk.score:.4f}] {chunk.title} (p.{chunk.page_number}) -- {chunk.text[:100]!r}")
        except Exception as exc:  # noqa: BLE001 -- manual diagnostic script
            print(f"  SKIPPED: {exc}")
