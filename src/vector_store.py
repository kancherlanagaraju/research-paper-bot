"""Zilliz Cloud Serverless vector store client, via `pymilvus`.

Responsibilities:

* One collection per embedding model (never per dimension alone -- two
  different 384-dim models never share a collection either), named from
  ``AppConfig.zilliz_collection_prefix`` plus a slug of the model name. This
  satisfies "use separate collections when embedding dimensions differ" by
  construction, since different models always get different collections.
* An idempotent ``upsert_chunks``: every row is keyed by the deterministic
  ``ChunkRecord.chunk_id`` from ``src.ingestion``, so re-running indexing on
  unchanged source files overwrites the same rows instead of duplicating
  them (Milvus/Zilliz ``upsert`` = delete-then-insert by primary key).
* An explicit ``rebuild_collection`` that intentionally drops and recreates
  a collection -- the "provide a command that rebuilds an index
  intentionally" requirement.
* Every function that talks to the network takes an already-constructed
  ``client`` as its first argument, so orchestration logic (which rows go in
  which batch, whether a collection needed creating, etc.) can be unit
  tested against an in-memory fake client, independent of real
  connectivity.

NOTE ON VERIFICATION: outbound access to `*.cloud.zilliz.com` was blocked at
the network layer in the sandboxed environment this was written in (see
README "Known limitations") -- schema/index-parameter construction was
verified directly against the real `pymilvus` classes (no network needed for
that), and upsert/rebuild orchestration is unit-tested against a fake client
(tests/test_vector_store.py). The actual connection to your Zilliz Cloud
Serverless cluster must be verified by running scripts/index_documents.py
from an environment with network access to it.
"""
from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any, Dict, List, Sequence

from src.config import AppConfig
from src.ingestion import ChunkRecord

logger = logging.getLogger(__name__)

# Field length limits (Milvus VARCHAR requires an explicit max_length).
_ID_MAX_LEN = 160
_SHORT_STR_MAX_LEN = 512
_FILENAME_MAX_LEN = 256
_TEXT_MAX_LEN = 8192
_HASH_MAX_LEN = 64
_TAG_MAX_LEN = 64

VECTOR_FIELD = "vector"
PRIMARY_KEY_FIELD = "chunk_id"


def _slugify_identifier(value: str) -> str:
    """Make ``value`` a valid Milvus collection-name component:
    ``^[a-zA-Z_][a-zA-Z0-9_]*$``, reasonably short.
    """
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not slug or not (slug[0].isalpha() or slug[0] == "_"):
        slug = f"m_{slug}"
    return slug


def collection_name_for(collection_prefix: str, embedding_model_name: str) -> str:
    """Deterministic collection name for a given embedding model.

    Different models -> different names, always -- so collections are never
    accidentally shared across incompatible embedding dimensions/spaces.
    """
    prefix_slug = _slugify_identifier(collection_prefix)
    model_slug = _slugify_identifier(embedding_model_name)
    name = f"{prefix_slug}__{model_slug}"
    return name[:255]


def get_client(config: AppConfig):
    """Build a connected `pymilvus.MilvusClient` from Zilliz credentials."""
    uri, token = config.require_zilliz_credentials()
    try:
        from pymilvus import MilvusClient
    except ImportError as exc:
        raise RuntimeError("pymilvus is not installed. Install it with `pip install pymilvus`.") from exc
    return MilvusClient(uri=uri, token=token)


def _build_schema(dim: int):
    from pymilvus import DataType, MilvusClient

    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(PRIMARY_KEY_FIELD, DataType.VARCHAR, max_length=_ID_MAX_LEN, is_primary=True)
    schema.add_field(VECTOR_FIELD, DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=_ID_MAX_LEN)
    schema.add_field("title", DataType.VARCHAR, max_length=_SHORT_STR_MAX_LEN)
    schema.add_field("filename", DataType.VARCHAR, max_length=_FILENAME_MAX_LEN)
    schema.add_field("relative_path", DataType.VARCHAR, max_length=_SHORT_STR_MAX_LEN)
    schema.add_field("page_number", DataType.INT64)
    schema.add_field("page_end", DataType.INT64)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("chunk_index_in_page", DataType.INT64)
    schema.add_field("text", DataType.VARCHAR, max_length=_TEXT_MAX_LEN)
    schema.add_field("token_count", DataType.INT64)
    schema.add_field("char_count", DataType.INT64)
    schema.add_field("content_hash", DataType.VARCHAR, max_length=_HASH_MAX_LEN)
    schema.add_field("tokenizer_backend", DataType.VARCHAR, max_length=_TAG_MAX_LEN)
    schema.add_field("embedding_model", DataType.VARCHAR, max_length=_SHORT_STR_MAX_LEN)
    schema.add_field("ingestion_version", DataType.VARCHAR, max_length=_TAG_MAX_LEN)
    return schema


def _build_index_params():
    from pymilvus import MilvusClient

    index_params = MilvusClient.prepare_index_params()
    # COSINE directly, so retrieval configuration A ("dense cosine similarity")
    # needs no manual vector normalization step.
    index_params.add_index(field_name=VECTOR_FIELD, index_type="AUTOINDEX", metric_type="COSINE")
    return index_params


def ensure_collection(client: Any, collection_name: str, dim: int) -> bool:
    """Create ``collection_name`` (with index) if it doesn't already exist.

    Returns True if it was created, False if it already existed (a no-op).
    """
    if client.has_collection(collection_name):
        logger.info("Collection '%s' already exists; leaving it as-is.", collection_name)
        return False
    logger.info("Creating collection '%s' (dim=%d)...", collection_name, dim)
    client.create_collection(
        collection_name=collection_name, schema=_build_schema(dim), index_params=_build_index_params()
    )
    return True


def rebuild_collection(client: Any, collection_name: str, dim: int) -> None:
    """Intentionally drop and recreate ``collection_name``."""
    if client.has_collection(collection_name):
        logger.warning("Dropping existing collection '%s' (--rebuild requested).", collection_name)
        client.drop_collection(collection_name)
    client.create_collection(
        collection_name=collection_name, schema=_build_schema(dim), index_params=_build_index_params()
    )
    logger.info("Rebuilt collection '%s' (dim=%d).", collection_name, dim)


def chunk_to_row(chunk: ChunkRecord, vector: Sequence[float], embedding_model: str) -> Dict:
    """One chunk + its vector -> a row dict ready for `client.upsert`.

    ``embedding_model`` is stamped onto the row explicitly (chunks from
    src.ingestion carry ``embedding_model=None`` until they're actually
    embedded).
    """
    stamped = replace(chunk, embedding_model=embedding_model)
    row = stamped.to_dict()
    row[VECTOR_FIELD] = list(vector)
    return row


def upsert_chunks(
    client: Any,
    collection_name: str,
    chunks: List[ChunkRecord],
    vectors: List[Sequence[float]],
    embedding_model: str,
    batch_size: int = 100,
) -> int:
    """Upsert ``chunks``/``vectors`` (same order, same length) keyed by
    ``chunk_id``. Idempotent: re-running with the same chunk IDs overwrites
    rather than duplicates. Returns the number of rows written.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must be the same length")

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]
        rows = [chunk_to_row(c, v, embedding_model) for c, v in zip(batch_chunks, batch_vectors)]
        client.upsert(collection_name=collection_name, data=rows)
        total += len(rows)
    logger.info("Upserted %d row(s) into '%s'.", total, collection_name)
    return total


def collection_stats(client: Any, collection_name: str) -> Dict:
    """Row count and basic info for a collection (best-effort)."""
    return client.get_collection_stats(collection_name)
