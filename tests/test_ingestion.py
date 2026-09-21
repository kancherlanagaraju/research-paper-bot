"""Unit tests for src.ingestion (Milestone 1).

Covers: whitespace normalization, the offline tokenizer, chunk-window
math, deterministic chunk IDs, and an end-to-end run against the real
sample dataset (5 PDFs under pinnacle_capstone_data/), including an
idempotency check (re-running produces identical chunk IDs).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import AppConfig, ConfigError
from src.ingestion import (
    ChunkRecord,
    Tokenizer,
    _chunk_token_index_spans,
    _make_chunk_id,
    _make_doc_id,
    chunk_page_text,
    discover_pdfs,
    normalize_whitespace,
    run_ingestion,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "pinnacle_capstone_data"


# --------------------------------------------------------------------------- #
# normalize_whitespace
# --------------------------------------------------------------------------- #


def test_normalize_whitespace_collapses_spaces_and_tabs():
    assert normalize_whitespace("a   b\t\tc") == "a b c"


def test_normalize_whitespace_strips_line_edges_and_collapses_blank_runs():
    raw = "  Title  \n\n\n\n  Body line  \n   \nMore body  "
    normalized = normalize_whitespace(raw)
    assert "\n\n\n" not in normalized
    assert normalized.startswith("Title")
    assert normalized.endswith("More body")


def test_normalize_whitespace_handles_nbsp():
    assert normalize_whitespace("a  b") == "a b"


# --------------------------------------------------------------------------- #
# Tokenizer (approx_word backend -- default, offline)
# --------------------------------------------------------------------------- #


def test_tokenizer_rejects_unknown_backend():
    with pytest.raises(ValueError):
        Tokenizer(backend="not-a-real-backend")


def test_approx_word_tokenizer_count_matches_word_count():
    tok = Tokenizer(backend="approx_word")
    assert tok.count("one two three") == 3
    assert tok.count("") == 0


def test_approx_word_tokenizer_encode_decode_is_stable():
    tok = Tokenizer(backend="approx_word")
    text = "Attention is all you need."
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)
    # approx_word decode rejoins with single spaces; content/order is preserved.
    assert decoded.split() == text.split()


# --------------------------------------------------------------------------- #
# Chunk window math
# --------------------------------------------------------------------------- #


def test_chunk_spans_basic_windows_with_overlap():
    # 10 tokens, window 4, overlap 1 -> [0,4) [3,7) [6,10)
    spans = _chunk_token_index_spans(n_tokens=10, chunk_size=4, overlap=1, min_tail=1)
    assert spans[0] == (0, 4)
    assert spans[1] == (3, 7)
    assert spans[-1][1] == 10  # last window always reaches the end


def test_chunk_spans_merges_small_trailing_window():
    # 9 tokens, window 4, overlap 1 -> naive windows: (0,4)(3,7)(6,9) -> tail len 3
    # with min_tail=5 the tail should be merged into the previous window.
    spans = _chunk_token_index_spans(n_tokens=9, chunk_size=4, overlap=1, min_tail=5)
    assert spans[-1][1] == 9
    assert len(spans) == 2  # merged down from 3 windows


def test_chunk_spans_empty_text_yields_no_chunks():
    assert _chunk_token_index_spans(n_tokens=0, chunk_size=800, overlap=120, min_tail=30) == []


def test_chunk_page_text_respects_max_chunk_size():
    tok = Tokenizer(backend="approx_word")
    text = " ".join(f"word{i}" for i in range(1000))
    chunks = chunk_page_text(text, tok, chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    for c in chunks:
        assert c["token_count"] <= 100


def test_chunk_page_text_rejects_overlap_gte_chunk_size():
    tok = Tokenizer(backend="approx_word")
    with pytest.raises(ValueError):
        chunk_page_text("some text here", tok, chunk_size_tokens=10, overlap_tokens=10)


def test_chunk_page_text_short_page_is_single_chunk():
    tok = Tokenizer(backend="approx_word")
    text = "A short abstract with only a few words."
    chunks = chunk_page_text(text, tok, chunk_size_tokens=800, overlap_tokens=120)
    assert len(chunks) == 1
    assert chunks[0]["text"].split() == text.split()


# --------------------------------------------------------------------------- #
# Deterministic IDs
# --------------------------------------------------------------------------- #


def test_chunk_id_is_deterministic_across_calls():
    id1 = _make_chunk_id("attention_paper.pdf", "v1", page_number=3, chunk_index_in_page=0)
    id2 = _make_chunk_id("attention_paper.pdf", "v1", page_number=3, chunk_index_in_page=0)
    assert id1 == id2


def test_chunk_id_changes_with_page_or_index():
    base = _make_chunk_id("attention_paper.pdf", "v1", page_number=3, chunk_index_in_page=0)
    other_page = _make_chunk_id("attention_paper.pdf", "v1", page_number=4, chunk_index_in_page=0)
    other_index = _make_chunk_id("attention_paper.pdf", "v1", page_number=3, chunk_index_in_page=1)
    other_version = _make_chunk_id("attention_paper.pdf", "v2", page_number=3, chunk_index_in_page=0)
    assert len({base, other_page, other_index, other_version}) == 4


def test_chunk_id_does_not_depend_on_chunk_text():
    # By design: IDs are derived from position, not content, so re-ingesting
    # a file whose extracted text shifts slightly (e.g. a PyMuPDF version
    # bump) still produces the same ID for "page 3, first chunk".
    id1 = _make_chunk_id("gpt4.pdf", "v1", page_number=1, chunk_index_in_page=0)
    id2 = _make_chunk_id("gpt4.pdf", "v1", page_number=1, chunk_index_in_page=0)
    assert id1 == id2  # text is not even a parameter -- nothing to vary


def test_doc_id_is_a_filesystem_and_url_safe_slug():
    doc_id = _make_doc_id("Some Paper (v2).pdf")
    assert doc_id == doc_id.lower()
    assert " " not in doc_id
    assert "(" not in doc_id


# --------------------------------------------------------------------------- #
# AppConfig validation
# --------------------------------------------------------------------------- #


def test_config_rejects_overlap_greater_or_equal_to_chunk_size():
    cfg = AppConfig(dataset_dir=DATASET_DIR, chunk_size_tokens=100, chunk_overlap_tokens=100)
    with pytest.raises(ConfigError):
        cfg.validate_ingestion()


def test_config_rejects_missing_dataset_dir(tmp_path):
    missing = tmp_path / "does-not-exist"
    cfg = AppConfig(dataset_dir=missing)
    with pytest.raises(ConfigError):
        cfg.validate_ingestion()


def test_config_require_openai_key_raises_actionable_error():
    cfg = AppConfig(dataset_dir=DATASET_DIR, openai_api_key=None)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        cfg.require_openai_key()


def test_config_require_zilliz_credentials_raises_actionable_error():
    cfg = AppConfig(dataset_dir=DATASET_DIR, zilliz_uri=None, zilliz_token=None)
    with pytest.raises(ConfigError, match="ZILLIZ_URI"):
        cfg.require_zilliz_credentials()


# --------------------------------------------------------------------------- #
# discover_pdfs
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="sample dataset not present")
def test_discover_pdfs_finds_all_sample_papers():
    pdfs = discover_pdfs(DATASET_DIR)
    names = sorted(p.name for p in pdfs)
    assert names == [
        "attention_paper.pdf",
        "gemini_paper.pdf",
        "gpt4.pdf",
        "instructgpt.pdf",
        "mistral_paper.pdf",
    ]


def test_discover_pdfs_raises_for_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_pdfs(tmp_path / "nope")


# --------------------------------------------------------------------------- #
# End-to-end ingestion against the real sample dataset
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def ingestion_config() -> AppConfig:
    return AppConfig(dataset_dir=DATASET_DIR)


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="sample dataset not present")
def test_run_ingestion_processes_all_papers_with_no_failures(ingestion_config):
    result = run_ingestion(ingestion_config)
    summary = result.summary()

    assert summary["documents_discovered"] == 5
    assert summary["documents_succeeded"] == 5
    assert summary["documents_failed"] == 0
    assert summary["total_page_failures"] == 0  # all 5 papers are text-based, none scanned
    assert summary["total_chunks"] > 0


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="sample dataset not present")
def test_run_ingestion_preserves_required_metadata_on_every_chunk(ingestion_config):
    result = run_ingestion(ingestion_config)
    chunks = result.all_chunks()
    assert chunks, "expected at least one chunk"

    for chunk in chunks:
        assert isinstance(chunk, ChunkRecord)
        assert chunk.title  # non-empty
        assert chunk.filename.endswith(".pdf")
        assert chunk.page_number >= 1
        assert chunk.chunk_id
        assert chunk.ingestion_version == "v1"
        assert chunk.text.strip() != ""
        assert chunk.token_count > 0
        d = chunk.to_dict()
        assert d["chunk_id"] == chunk.chunk_id  # to_dict round-trips cleanly


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="sample dataset not present")
def test_run_ingestion_titles_are_correctly_detected(ingestion_config):
    result = run_ingestion(ingestion_config)
    titles_by_file = {doc.filename: doc.title for doc in result.documents}

    assert titles_by_file["attention_paper.pdf"] == "Attention Is All You Need"
    assert titles_by_file["gpt4.pdf"] == "GPT-4 Technical Report"
    assert titles_by_file["mistral_paper.pdf"] == "Mistral 7B"
    assert "Gemini" in titles_by_file["gemini_paper.pdf"]
    assert "instruct" in titles_by_file["instructgpt.pdf"].lower()


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="sample dataset not present")
def test_run_ingestion_chunk_ids_are_unique(ingestion_config):
    result = run_ingestion(ingestion_config)
    chunk_ids = [c.chunk_id for c in result.all_chunks()]
    assert len(chunk_ids) == len(set(chunk_ids))


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="sample dataset not present")
def test_run_ingestion_is_idempotent_across_repeated_runs(ingestion_config):
    """Rerunning ingestion on unchanged files must reproduce identical chunk
    IDs (and the same count), which is what lets a future vector-store
    upsert avoid creating duplicates.
    """
    result1 = run_ingestion(ingestion_config)
    result2 = run_ingestion(ingestion_config)

    ids1 = [c.chunk_id for c in result1.all_chunks()]
    ids2 = [c.chunk_id for c in result2.all_chunks()]
    assert ids1 == ids2  # same order, same IDs


@pytest.mark.skipif(not DATASET_DIR.exists(), reason="sample dataset not present")
def test_run_ingestion_page_numbers_stay_within_document_bounds(ingestion_config):
    result = run_ingestion(ingestion_config)
    pages_by_file = {doc.filename: doc.num_pages for doc in result.documents}
    for chunk in result.all_chunks():
        assert 1 <= chunk.page_number <= pages_by_file[chunk.filename]
