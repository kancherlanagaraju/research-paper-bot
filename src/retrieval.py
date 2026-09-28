"""Retrieval strategies.

Milestone 3 (implemented): dense cosine retrieval, one path per embedding
backend, so configuration A (open-source dense + cosine) and configuration
B (OpenAI dense + cosine) can be run against the same query set and
compared later under identical conditions (same dataset, chunking, and
top_k) per the brief.

Milestone 4 (implemented): configuration C is dense
retrieval (backend chosen by ``AppConfig.hybrid_dense_backend``) plus a BM25
sparse ranking over the same collection's chunk text, fused via a documented
method -- RRF or weighted (min-max normalised), configurable via
``AppConfig.fusion_method`` / ``fusion_weight_dense`` / ``fusion_weight_sparse``.
Each ranker contributes its top ``AppConfig.hybrid_candidate_k`` candidates.
Configuration D reranks those hybrid candidates down to
``AppConfig.rerank_top_k`` with the cross-encoder ``AppConfig.reranker_model``.

NOTE ON VERIFICATION: `dense_search`'s orchestration (the exact `client.search`
call it makes) is unit-tested against a fake client
(tests/test_retrieval.py). Actually retrieving from a populated Zilliz
collection depends on Milestone 2's indexing having been run for real first
-- see README "Known limitations" for how that's been verified so far.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.config import AppConfig
from src.embeddings import get_backend
from src.vector_store import PRIMARY_KEY_FIELD, collection_name_for, get_client

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


# --------------------------------------------------------------------------- #
# Sparse (BM25) retrieval + fusion -- Milestone 4, configuration C
# --------------------------------------------------------------------------- #

RRF_K = 60  # standard RRF damping constant (Cormack et al., 2009)
_QUERY_PAGE_SIZE = 1000
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_for_bm25(text: str) -> List[str]:
    """Lowercase alphanumeric tokens; deterministic and offline."""
    return _TOKEN_RE.findall(text.lower())


def fetch_all_chunks(client: Any, collection_name: str) -> List[RetrievedChunk]:
    """Page every chunk out of a collection (score=0.0) to build the BM25 corpus.

    Reading the corpus from the same collection the dense side searches
    guarantees both rankers share chunk IDs, so fusion can join on them.
    """
    chunks: List[RetrievedChunk] = []
    offset = 0
    while True:
        rows = client.query(
            collection_name=collection_name,
            filter=f'{PRIMARY_KEY_FIELD} != ""',
            output_fields=[PRIMARY_KEY_FIELD] + OUTPUT_FIELDS,
            limit=_QUERY_PAGE_SIZE,
            offset=offset,
        )
        if not rows:
            break
        chunks.extend(RetrievedChunk.from_hit({"id": r.get(PRIMARY_KEY_FIELD, ""), "entity": r}) for r in rows)
        if len(rows) < _QUERY_PAGE_SIZE:
            break
        offset += len(rows)
    return chunks


class BM25Index:
    """In-memory BM25 (Okapi) index over a fixed list of chunks."""

    def __init__(self, chunks: Sequence[RetrievedChunk]):
        from rank_bm25 import BM25Okapi

        self.chunks: List[RetrievedChunk] = list(chunks)
        self._bm25 = BM25Okapi([tokenize_for_bm25(c.text) for c in self.chunks]) if self.chunks else None

    def search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        tokens = tokenize_for_bm25(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))
        # Zero score = no query term matched; not a real sparse hit.
        return [replace(self.chunks[i], score=float(scores[i])) for i in ranked[:top_k] if scores[i] > 0]


_BM25_CACHE: Dict[str, BM25Index] = {}


def get_bm25_index(client: Any, collection_name: str) -> BM25Index:
    """Build (once per process, per collection) the BM25 index."""
    if collection_name not in _BM25_CACHE:
        chunks = fetch_all_chunks(client, collection_name)
        logger.info("Built BM25 index over %d chunks from %s", len(chunks), collection_name)
        _BM25_CACHE[collection_name] = BM25Index(chunks)
    return _BM25_CACHE[collection_name]


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def fuse_results(
    dense: List[RetrievedChunk],
    sparse: List[RetrievedChunk],
    method: str = "rrf",
    weight_dense: float = 0.5,
    weight_sparse: float = 0.5,
    top_k: int = 5,
    rrf_k: int = RRF_K,
) -> List[RetrievedChunk]:
    """Fuse two ranked lists (best first) into one, joined on ``chunk_id``.

    * ``rrf``: score = sum_r w_r / (rrf_k + rank_r), rank starting at 1. Uses
      ranks only, so it needs no score normalisation across cosine and BM25.
    * ``weighted``: each list's scores are min-max normalised to [0, 1], then
      score = w_dense * dense_norm + w_sparse * sparse_norm.

    A chunk missing from one list contributes 0 from that list. Ties break by
    first appearance (dense list first) so output is deterministic.
    """
    if method not in {"rrf", "weighted"}:
        raise ValueError(f"Unknown fusion method '{method}'. Use 'rrf' or 'weighted'.")

    fused: Dict[str, float] = {}
    by_id: Dict[str, RetrievedChunk] = {}
    for chunks, weight in ((dense, weight_dense), (sparse, weight_sparse)):
        if method == "rrf":
            contribs = [weight / (rrf_k + rank) for rank in range(1, len(chunks) + 1)]
        else:
            contribs = [weight * n for n in _minmax([c.score for c in chunks])]
        for chunk, contrib in zip(chunks, contribs):
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + contrib
            by_id.setdefault(chunk.chunk_id, chunk)

    order = {cid: i for i, cid in enumerate(by_id)}
    ranked = sorted(fused, key=lambda cid: (-fused[cid], order[cid]))
    return [replace(by_id[cid], score=fused[cid]) for cid in ranked[:top_k]]


def retrieve_hybrid(query: str, config: AppConfig, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    """Configuration C: dense + BM25, fused (see :func:`fuse_results`)."""
    backend = get_backend(config.hybrid_dense_backend, config)
    query_vector = backend.embed_query(query)
    collection_name = collection_name_for(config.zilliz_collection_prefix, backend.name)
    client = get_client(config)
    k = top_k if top_k is not None else config.top_k_dense
    candidates = max(config.hybrid_candidate_k, k)
    logger.info(
        "Hybrid search: dense_backend=%s collection=%s candidates=%d fusion=%s top_k=%d",
        config.hybrid_dense_backend, collection_name, candidates, config.fusion_method, k,
    )
    dense = dense_search(client, collection_name, query_vector, top_k=candidates)
    sparse = get_bm25_index(client, collection_name).search(query, top_k=candidates)
    return fuse_results(
        dense,
        sparse,
        method=config.fusion_method,
        weight_dense=config.fusion_weight_dense,
        weight_sparse=config.fusion_weight_sparse,
        top_k=k,
    )


# --------------------------------------------------------------------------- #
# Cross-encoder reranking -- Milestone 4, configuration D
# --------------------------------------------------------------------------- #

_RERANKER_CACHE: Dict[str, Any] = {}


def get_reranker(model_name: str) -> Any:
    """Load (once per process) a sentence-transformers CrossEncoder."""
    if model_name not in _RERANKER_CACHE:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Install it with `pip install sentence-transformers`."
            ) from exc
        logger.info("Loading cross-encoder reranker %s", model_name)
        _RERANKER_CACHE[model_name] = CrossEncoder(model_name)
    return _RERANKER_CACHE[model_name]


def rerank(query: str, chunks: List[RetrievedChunk], model: Any, top_k: int) -> List[RetrievedChunk]:
    """Re-score ``chunks`` against ``query`` with a cross-encoder and keep ``top_k``.

    ``model`` needs only ``.predict(list_of_(query, text)_pairs) -> scores``
    (higher = more relevant), so tests can inject a stub. The returned
    chunks carry the cross-encoder score; ties keep the incoming order.
    """
    if not chunks:
        return []
    scores = model.predict([(query, c.text) for c in chunks])
    ranked = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), i))
    return [replace(chunks[i], score=float(scores[i])) for i in ranked[:top_k]]


def retrieve_hybrid_reranked(query: str, config: AppConfig, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    """Configuration D: hybrid retrieval of ``hybrid_candidate_k`` candidates,
    cross-encoder reranked down to ``top_k`` (default ``rerank_top_k``).
    """
    k = top_k if top_k is not None else config.rerank_top_k
    candidates = retrieve_hybrid(query, config, top_k=max(config.hybrid_candidate_k, k))
    logger.info("Reranking %d candidates with %s -> top %d", len(candidates), config.reranker_model, k)
    return rerank(query, candidates, get_reranker(config.reranker_model), top_k=k)


_MODE_TO_EMBEDDING_KIND = {
    "dense_oss": "oss",
    "dense_openai": "openai",
}


def retrieve(query: str, mode: str = "dense_oss", config: Optional[AppConfig] = None, top_k: Optional[int] = None) -> List[RetrievedChunk]:
    """Mode dispatcher used by rag.py / evaluation.py / the Streamlit app.

    Modes: "dense_oss" (config A), "dense_openai" (config B), "hybrid"
    (config C), "hybrid_reranked" (config D).
    """
    config = config or AppConfig.from_env()
    if mode in _MODE_TO_EMBEDDING_KIND:
        return retrieve_dense(query, _MODE_TO_EMBEDDING_KIND[mode], config, top_k)
    if mode == "hybrid":
        return retrieve_hybrid(query, config, top_k)
    if mode == "hybrid_reranked":
        return retrieve_hybrid_reranked(query, config, top_k)
    raise ValueError(
        f"Unknown retrieval mode '{mode}'. Available: "
        f"{sorted([*_MODE_TO_EMBEDDING_KIND, 'hybrid', 'hybrid_reranked'])}."
    )


if __name__ == "__main__":
    # Manual live check: `python -m src.retrieval "your question here"`
    import sys

    cfg = AppConfig.from_env()
    question = " ".join(sys.argv[1:]) or "What is attention in a transformer model?"
    for mode in ("dense_oss", "dense_openai", "hybrid", "hybrid_reranked"):
        print(f"\n--- mode={mode} ---")
        try:
            for chunk in retrieve(question, mode=mode, config=cfg, top_k=3):
                print(f"  [{chunk.score:.4f}] {chunk.title} (p.{chunk.page_number}) -- {chunk.text[:100]!r}")
        except Exception as exc:  # noqa: BLE001 -- manual diagnostic script
            print(f"  SKIPPED: {exc}")
