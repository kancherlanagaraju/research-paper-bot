"""Unit tests for src.vector_store.

Schema/index-parameter construction is tested against the REAL `pymilvus`
classes (no network involved in building a schema object). Everything that
would otherwise need a live cluster (create_collection, upsert, drop) is
tested against `_FakeMilvusClient`, an in-memory stand-in that implements
just the subset of the MilvusClient interface this module calls.

Live connectivity to Zilliz Cloud Serverless itself could not be verified
from the sandboxed environment this was developed in (see README "Known
limitations") and must be verified by running scripts/index_documents.py
against your real cluster.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion import ChunkRecord
from src.vector_store import (
    PRIMARY_KEY_FIELD,
    VECTOR_FIELD,
    _build_index_params,
    _build_schema,
    chunk_to_row,
    collection_name_for,
    ensure_collection,
    rebuild_collection,
    upsert_chunks,
)


def _make_chunk(chunk_id="c1", text="hello world", embedding_model=None) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id="doc1",
        title="Some Paper",
        filename="some.pdf",
        relative_path="some.pdf",
        page_number=1,
        page_end=1,
        chunk_index=0,
        chunk_index_in_page=0,
        text=text,
        token_count=2,
        char_count=len(text),
        content_hash="deadbeef",
        tokenizer_backend="approx_word",
        embedding_model=embedding_model,
        ingestion_version="v1",
    )


# --------------------------------------------------------------------------- #
# Fake Milvus client (in-memory)
# --------------------------------------------------------------------------- #


class _FakeMilvusClient:
    def __init__(self):
        self.collections = {}  # name -> {"rows": {chunk_id: row}}
        self.create_calls = []
        self.drop_calls = []
        self.upsert_calls = []

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_collection(self, collection_name, schema, index_params):
        self.create_calls.append(collection_name)
        self.collections[collection_name] = {"rows": {}, "schema": schema, "index_params": index_params}

    def drop_collection(self, collection_name):
        self.drop_calls.append(collection_name)
        self.collections.pop(collection_name, None)

    def upsert(self, collection_name, data):
        self.upsert_calls.append((collection_name, len(data)))
        rows = self.collections[collection_name]["rows"]
        for row in data:
            rows[row[PRIMARY_KEY_FIELD]] = row

    def get_collection_stats(self, collection_name):
        return {"row_count": len(self.collections[collection_name]["rows"])}


# --------------------------------------------------------------------------- #
# collection_name_for
# --------------------------------------------------------------------------- #


def test_collection_name_differs_by_embedding_model():
    a = collection_name_for("res-bot", "BAAI/bge-small-en-v1.5")
    b = collection_name_for("res-bot", "text-embedding-3-small")
    assert a != b


def test_collection_name_is_valid_milvus_identifier():
    name = collection_name_for("res-bot", "BAAI/bge-small-en-v1.5")
    assert name[0].isalpha() or name[0] == "_"
    assert all(ch.isalnum() or ch == "_" for ch in name)


def test_collection_name_is_deterministic():
    a = collection_name_for("res-bot", "BAAI/bge-small-en-v1.5")
    b = collection_name_for("res-bot", "BAAI/bge-small-en-v1.5")
    assert a == b


# --------------------------------------------------------------------------- #
# Schema / index params (real pymilvus objects, no network)
# --------------------------------------------------------------------------- #


def test_build_schema_has_expected_fields_and_primary_key():
    schema = _build_schema(dim=384)
    field_names = [f.name for f in schema.fields]
    assert PRIMARY_KEY_FIELD in field_names
    assert VECTOR_FIELD in field_names
    assert "title" in field_names and "page_number" in field_names and "embedding_model" in field_names

    primary_fields = [f for f in schema.fields if f.is_primary]
    assert len(primary_fields) == 1
    assert primary_fields[0].name == PRIMARY_KEY_FIELD


def test_build_schema_vector_field_has_requested_dimension():
    schema = _build_schema(dim=1536)
    vector_field = next(f for f in schema.fields if f.name == VECTOR_FIELD)
    assert vector_field.params["dim"] == 1536


def test_build_index_params_uses_cosine_metric():
    # Different pymilvus/platform combinations expose the built IndexParams
    # object's internals differently (a list of dicts on some installs, a
    # non-subscriptable `IndexParam` object on others), so rather than
    # introspecting that object's shape, verify OUR code's intent: that
    # `add_index` was called with the field/metric we require.
    from unittest.mock import MagicMock

    from pymilvus import MilvusClient

    fake_index_params = MagicMock()
    original = MilvusClient.prepare_index_params
    MilvusClient.prepare_index_params = staticmethod(lambda: fake_index_params)
    try:
        result = _build_index_params()
    finally:
        MilvusClient.prepare_index_params = original

    assert result is fake_index_params
    fake_index_params.add_index.assert_called_once_with(
        field_name=VECTOR_FIELD, index_type="AUTOINDEX", metric_type="COSINE"
    )


# --------------------------------------------------------------------------- #
# chunk_to_row
# --------------------------------------------------------------------------- #


def test_chunk_to_row_stamps_embedding_model_without_mutating_original():
    chunk = _make_chunk(embedding_model=None)
    row = chunk_to_row(chunk, [0.1, 0.2, 0.3], embedding_model="BAAI/bge-small-en-v1.5")
    assert row["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert row[VECTOR_FIELD] == [0.1, 0.2, 0.3]
    assert chunk.embedding_model is None  # original ChunkRecord is frozen/untouched


# --------------------------------------------------------------------------- #
# ensure_collection / rebuild_collection / upsert_chunks (fake client)
# --------------------------------------------------------------------------- #


def test_ensure_collection_creates_once_and_is_a_noop_after():
    client = _FakeMilvusClient()
    created = ensure_collection(client, "my_collection", dim=384)
    assert created is True
    assert client.create_calls == ["my_collection"]

    created_again = ensure_collection(client, "my_collection", dim=384)
    assert created_again is False
    assert client.create_calls == ["my_collection"]  # not called a second time


def test_rebuild_collection_drops_then_recreates():
    client = _FakeMilvusClient()
    ensure_collection(client, "my_collection", dim=384)
    client.upsert("my_collection", [{"chunk_id": "x"}])
    assert client.get_collection_stats("my_collection")["row_count"] == 1

    rebuild_collection(client, "my_collection", dim=384)
    assert client.drop_calls == ["my_collection"]
    assert client.get_collection_stats("my_collection")["row_count"] == 0  # fresh, empty collection


def test_upsert_chunks_is_idempotent_by_chunk_id():
    client = _FakeMilvusClient()
    ensure_collection(client, "my_collection", dim=3)
    chunks = [_make_chunk(chunk_id="a"), _make_chunk(chunk_id="b")]
    vectors = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]

    written1 = upsert_chunks(client, "my_collection", chunks, vectors, embedding_model="m1")
    assert written1 == 2
    assert client.get_collection_stats("my_collection")["row_count"] == 2

    # Re-running with the SAME chunk_ids must overwrite, not duplicate.
    written2 = upsert_chunks(client, "my_collection", chunks, vectors, embedding_model="m1")
    assert written2 == 2
    assert client.get_collection_stats("my_collection")["row_count"] == 2


def test_upsert_chunks_batches_correctly():
    client = _FakeMilvusClient()
    ensure_collection(client, "my_collection", dim=3)
    chunks = [_make_chunk(chunk_id=f"c{i}") for i in range(5)]
    vectors = [[0.0, 0.0, 0.0] for _ in chunks]

    upsert_chunks(client, "my_collection", chunks, vectors, embedding_model="m1", batch_size=2)
    # 5 rows at batch_size=2 -> 3 upsert() calls (2, 2, 1)
    assert [n for _, n in client.upsert_calls] == [2, 2, 1]


def test_upsert_chunks_rejects_mismatched_lengths():
    client = _FakeMilvusClient()
    ensure_collection(client, "my_collection", dim=3)
    with pytest.raises(ValueError):
        upsert_chunks(client, "my_collection", [_make_chunk()], [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]], embedding_model="m1")
