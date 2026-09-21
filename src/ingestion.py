"""PDF extraction and metadata-preserving chunking.

Pipeline for this milestone:

1. :func:`discover_pdfs` recursively finds PDFs under the dataset directory.
2. :func:`extract_pdf_pages` reads each PDF page-by-page with PyMuPDF, so
   page citations stay accurate, normalizes whitespace, and records any
   page/file that fails to extract instead of silently skipping it.
3. :func:`chunk_page_text` splits each page's text into ~800-token chunks
   with 120-token overlap (both configurable), using a small, dependency-free
   tokenizer by default so the pipeline is reproducible without a network
   call (see :class:`Tokenizer`).
4. :func:`process_pdf` / :func:`run_ingestion` assemble :class:`ChunkRecord`
   objects that carry paper title, filename, page number, and a
   deterministic chunk ID -- everything downstream retrieval/citation code
   needs.

Design notes relevant to later milestones:

* Chunk IDs are derived from (filename, ingestion version, page number,
  chunk-within-page index) -- NOT from the chunk text. Re-running ingestion
  on unchanged source files therefore reproduces identical IDs, which is
  what makes an upsert-based vector-store write idempotent (Milestone 2).
* Each chunk also carries a ``content_hash`` of its own text, so a future
  re-ingestion can detect "this chunk's source text changed" and decide to
  update the stored record even though the ID stayed the same.
* Chunking is done independently per page (never merged across a page
  boundary). This keeps every chunk's page citation exact and simple, at
  the cost of some short chunks on short pages. This is a deliberate
  simplicity/accuracy trade-off for this capstone -- see README.
"""
from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pymupdf

from src.config import AppConfig

logger = logging.getLogger(__name__)

_ARXIV_WATERMARK_RE = re.compile(r"arxiv", re.IGNORECASE)
_WHITESPACE_RUN_RE = re.compile(r"[ \t ]+")
_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")
_WORD_TOKEN_RE = re.compile(r"\S+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


# =========================================================================
# Data model
# =========================================================================


@dataclass(frozen=True)
class PageFailure:
    """A page (or whole file) that could not be extracted."""

    filename: str
    page_number: Optional[int]  # None when the whole file failed to open
    error: str


@dataclass(frozen=True)
class PageRecord:
    page_number: int  # 1-indexed
    text: str
    char_count: int
    token_count: int


@dataclass(frozen=True)
class ChunkRecord:
    """One retrievable unit, with the metadata required by the capstone brief:
    paper title, filename, page number, chunk ID, embedding model, and
    ingestion version.
    """

    chunk_id: str
    doc_id: str
    title: str
    filename: str
    relative_path: str
    page_number: int
    page_end: int  # equal to page_number today; reserved for cross-page chunks
    chunk_index: int  # sequential index across the whole document
    chunk_index_in_page: int
    text: str
    token_count: int
    char_count: int
    content_hash: str
    tokenizer_backend: str
    embedding_model: Optional[str]
    ingestion_version: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocumentIngestionResult:
    filename: str
    relative_path: str
    title: str
    num_pages: int
    chunks: List[ChunkRecord] = field(default_factory=list)
    page_failures: List[PageFailure] = field(default_factory=list)
    file_error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.file_error is None

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "relative_path": self.relative_path,
            "title": self.title,
            "num_pages": self.num_pages,
            "num_chunks": len(self.chunks),
            "page_failures": [asdict(f) for f in self.page_failures],
            "file_error": self.file_error,
        }


@dataclass
class IngestionRunResult:
    documents: List[DocumentIngestionResult]
    dataset_dir: str
    started_at: str
    finished_at: str
    ingestion_version: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    tokenizer_backend: str

    def all_chunks(self) -> List[ChunkRecord]:
        return [c for doc in self.documents for c in doc.chunks]

    def summary(self) -> dict:
        chunks = self.all_chunks()
        return {
            "documents_discovered": len(self.documents),
            "documents_succeeded": sum(1 for d in self.documents if d.succeeded),
            "documents_failed": sum(1 for d in self.documents if not d.succeeded),
            "total_pages": sum(d.num_pages for d in self.documents),
            "total_page_failures": sum(len(d.page_failures) for d in self.documents),
            "total_chunks": len(chunks),
            "avg_chunk_tokens": round(sum(c.token_count for c in chunks) / len(chunks), 1) if chunks else 0.0,
        }


# =========================================================================
# Tokenizer (chunk sizing only -- not used for embeddings)
# =========================================================================


class Tokenizer:
    """Deterministic text tokenizer used only to size and slice chunks.

    Two backends:

    * ``"approx_word"`` (default): whitespace/punctuation-delimited tokens.
      No network access, no model download -- fully deterministic across
      machines, which matters because the SAME chunk boundaries must be
      produced every time ingestion runs (reproducibility, idempotency).
    * ``"tiktoken"``: exact OpenAI ``cl100k_base`` subword tokens. More
      accurate relative to the "~800 tokens" target in the brief, but
      requires downloading the vocabulary file on first use. Opt-in only,
      and fails loudly (rather than silently falling back) if unavailable,
      so chunking behavior never silently depends on network conditions.
    """

    def __init__(self, backend: str = "approx_word", encoding_name: str = "cl100k_base"):
        if backend not in {"approx_word", "tiktoken"}:
            raise ValueError(f"Unknown tokenizer backend '{backend}'. Use 'approx_word' or 'tiktoken'.")
        self.backend = backend
        self.encoding_name = encoding_name
        self._enc = None
        if backend == "tiktoken":
            try:
                import tiktoken  # optional dependency
            except ImportError as exc:
                raise RuntimeError(
                    "TOKENIZER_BACKEND=tiktoken requires the 'tiktoken' package. Install it with "
                    "`pip install tiktoken` or set TOKENIZER_BACKEND=approx_word."
                ) from exc
            try:
                self._enc = tiktoken.get_encoding(encoding_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load tiktoken encoding '{encoding_name}'. This usually means there is "
                    "no network access to download the vocabulary file on first use. Set "
                    "TOKENIZER_BACKEND=approx_word to use the offline tokenizer instead."
                ) from exc

    def encode(self, text: str) -> List:
        """Return an opaque, sliceable token sequence for ``text``."""
        if self.backend == "tiktoken":
            return self._enc.encode(text)
        return _WORD_TOKEN_RE.findall(text)

    def decode(self, tokens: Sequence) -> str:
        """Reconstruct text from a (possibly sliced) token sequence."""
        if self.backend == "tiktoken":
            return self._enc.decode(list(tokens))
        return " ".join(tokens)

    def count(self, text: str) -> int:
        return len(self.encode(text))


# =========================================================================
# Text normalization
# =========================================================================


def normalize_whitespace(text: str) -> str:
    """Collapse incidental whitespace while preserving paragraph structure.

    - Unicode-normalizes (NFKC) so ligatures/odd PDF glyphs behave consistently.
    - Collapses runs of spaces/tabs/nbsp to a single space.
    - Strips trailing/leading whitespace on each line.
    - Collapses 3+ consecutive blank lines down to a single blank line.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ")
    text = _WHITESPACE_RUN_RE.sub(" ", text)
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_LINE_RUN_RE.sub("\n\n", text)
    return text.strip()


def _filename_to_title(stem: str) -> str:
    words = re.split(r"[_\-]+", stem)
    return " ".join(w.capitalize() if w.islower() else w for w in words if w).strip() or stem


def _derive_title(doc: "pymupdf.Document", fallback: str) -> str:
    """Best-effort title extraction: the largest-font horizontal text on
    page 1, excluding arXiv's rotated watermark (which often uses a larger
    font than the actual title). Falls back to a filename-derived title if
    this heuristic finds nothing usable.
    """
    try:
        page = doc[0]
        raw = page.get_text("dict")
        spans: List[Tuple[float, str]] = []
        for block in raw.get("blocks", []):
            for line in block.get("lines", []):
                direction = line.get("dir", (1.0, 0.0))
                is_horizontal = abs(direction[0]) > 0.99
                if not is_horizontal:
                    continue
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if not txt or _ARXIV_WATERMARK_RE.search(txt):
                        continue
                    spans.append((round(span.get("size", 0.0), 1), txt))
        if not spans:
            return fallback
        max_size = max(size for size, _ in spans)
        title_spans = [txt for size, txt in spans if size >= max_size - 0.5][:10]
        title = re.sub(r"\s+", " ", " ".join(title_spans)).strip()
        return title or fallback
    except Exception as exc:  # pragma: no cover -- defensive, PyMuPDF internals
        logger.warning("Title extraction failed, using fallback: %s", exc)
        return fallback


def _make_doc_id(filename: str) -> str:
    return _SLUG_RE.sub("-", Path(filename).stem.lower()).strip("-")


def _make_chunk_id(filename: str, ingestion_version: str, page_number: int, chunk_index_in_page: int) -> str:
    """Deterministic, content-independent chunk ID.

    Depends only on (filename, ingestion version, page, in-page index), so
    re-running ingestion on the same source files reproduces identical IDs
    -- required for idempotent upserts into the vector store.
    """
    slug = _make_doc_id(filename)
    digest_input = f"{filename}|{ingestion_version}|p{page_number:04d}|c{chunk_index_in_page:04d}"
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:10]
    return f"{slug}-p{page_number:04d}-c{chunk_index_in_page:04d}-{digest}"


# =========================================================================
# PDF discovery + page extraction
# =========================================================================


def discover_pdfs(dataset_dir: Path) -> List[Path]:
    """Recursively find PDFs under ``dataset_dir`` (case-insensitive extension)."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    pdfs = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf"]
    return sorted(pdfs)


def extract_pdf_pages(
    pdf_path: Path, tokenizer: Tokenizer
) -> Tuple[str, List[PageRecord], List[PageFailure], Optional[str], int]:
    """Extract normalized text for every page of ``pdf_path``.

    Returns ``(title, pages, page_failures, file_error, total_page_count)``.
    ``file_error`` is set (and the other lists empty) if the file could not
    be opened at all. A page with no extractable text (e.g. a scanned image
    page with no OCR layer) is recorded as a :class:`PageFailure` rather
    than silently dropped or silently producing an empty chunk.
    """
    filename = pdf_path.name
    fallback_title = _filename_to_title(pdf_path.stem)

    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        error = f"Failed to open PDF: {exc}"
        logger.error("%s: %s", filename, error)
        return fallback_title, [], [PageFailure(filename=filename, page_number=None, error=error)], error, 0

    total_page_count = doc.page_count
    title = _derive_title(doc, fallback_title)
    pages: List[PageRecord] = []
    failures: List[PageFailure] = []

    for i in range(total_page_count):
        page_number = i + 1
        try:
            raw_text = doc[i].get_text("text")
            text = normalize_whitespace(raw_text)
            if not text:
                failures.append(
                    PageFailure(
                        filename=filename,
                        page_number=page_number,
                        error="No extractable text (likely a scanned/image page with no text layer).",
                    )
                )
                continue
            pages.append(
                PageRecord(
                    page_number=page_number,
                    text=text,
                    char_count=len(text),
                    token_count=tokenizer.count(text),
                )
            )
        except Exception as exc:
            failures.append(PageFailure(filename=filename, page_number=page_number, error=str(exc)))
            logger.error("%s page %d: extraction failed: %s", filename, page_number, exc)

    doc.close()
    return title, pages, failures, None, total_page_count


# =========================================================================
# Chunking
# =========================================================================


def _chunk_token_index_spans(n_tokens: int, chunk_size: int, overlap: int, min_tail: int) -> List[Tuple[int, int]]:
    """Compute (start, end) token-index windows covering ``n_tokens`` with
    the given size/overlap, merging a too-small trailing window into the
    previous one so chunks don't end in a near-empty sliver.
    """
    if n_tokens <= 0:
        return []
    spans: List[Tuple[int, int]] = []
    start = 0
    while start < n_tokens:
        end = min(start + chunk_size, n_tokens)
        spans.append((start, end))
        if end == n_tokens:
            break
        start = end - overlap
    if len(spans) > 1:
        last_start, last_end = spans[-1]
        if (last_end - last_start) < min_tail:
            prev_start, _prev_end = spans[-2]
            spans[-2] = (prev_start, last_end)
            spans.pop()
    return spans


def chunk_page_text(
    text: str,
    tokenizer: Tokenizer,
    chunk_size_tokens: int,
    overlap_tokens: int,
    min_tail_tokens: Optional[int] = None,
) -> List[Dict]:
    """Split one page's text into overlapping token-bounded chunks.

    Returns a list of dicts: ``{"text", "token_count", "start_token", "end_token"}``.
    A trailing sliver smaller than ``min_tail_tokens`` (default:
    ``max(30, overlap_tokens // 2)``) is merged into the previous chunk
    instead of being kept as its own tiny chunk.
    """
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")

    tokens = tokenizer.encode(text)
    min_tail = min_tail_tokens if min_tail_tokens is not None else max(30, overlap_tokens // 2)
    spans = _chunk_token_index_spans(len(tokens), chunk_size_tokens, overlap_tokens, min_tail)

    chunks = []
    for start, end in spans:
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(
            {
                "text": chunk_text,
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end,
            }
        )
    return chunks


# =========================================================================
# Per-document and full-run orchestration
# =========================================================================


def process_pdf(pdf_path: Path, dataset_dir: Path, config: AppConfig, tokenizer: Tokenizer) -> DocumentIngestionResult:
    """Extract + chunk a single PDF into a :class:`DocumentIngestionResult`."""
    filename = pdf_path.name
    try:
        relative_path = str(pdf_path.relative_to(dataset_dir))
    except ValueError:
        relative_path = filename

    title, pages, page_failures, file_error, total_page_count = extract_pdf_pages(pdf_path, tokenizer)
    doc_id = _make_doc_id(filename)
    chunks: List[ChunkRecord] = []
    global_index = 0

    if file_error is None:
        for page in pages:
            page_chunks = chunk_page_text(page.text, tokenizer, config.chunk_size_tokens, config.chunk_overlap_tokens)
            for i, chunk in enumerate(page_chunks):
                chunk_id = _make_chunk_id(filename, config.ingestion_version, page.page_number, i)
                content_hash = hashlib.sha256(chunk["text"].encode("utf-8")).hexdigest()[:16]
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        title=title,
                        filename=filename,
                        relative_path=relative_path,
                        page_number=page.page_number,
                        page_end=page.page_number,
                        chunk_index=global_index,
                        chunk_index_in_page=i,
                        text=chunk["text"],
                        token_count=chunk["token_count"],
                        char_count=len(chunk["text"]),
                        content_hash=content_hash,
                        tokenizer_backend=tokenizer.backend,
                        embedding_model=None,  # populated by the embeddings step (later milestone)
                        ingestion_version=config.ingestion_version,
                    )
                )
                global_index += 1

    return DocumentIngestionResult(
        filename=filename,
        relative_path=relative_path,
        title=title,
        num_pages=total_page_count,
        chunks=chunks,
        page_failures=page_failures,
        file_error=file_error,
    )


def run_ingestion(config: AppConfig) -> IngestionRunResult:
    """Run the full ingestion pipeline over ``config.dataset_dir``."""
    config.configure_logging()
    started_at = datetime.now(timezone.utc).isoformat()
    tokenizer = Tokenizer(backend=config.tokenizer_backend, encoding_name=config.tokenizer_encoding_name)

    pdf_paths = discover_pdfs(config.dataset_dir)
    logger.info("Discovered %d PDF(s) under %s", len(pdf_paths), config.dataset_dir)

    documents = [process_pdf(p, config.dataset_dir, config, tokenizer) for p in pdf_paths]
    finished_at = datetime.now(timezone.utc).isoformat()

    result = IngestionRunResult(
        documents=documents,
        dataset_dir=str(config.dataset_dir),
        started_at=started_at,
        finished_at=finished_at,
        ingestion_version=config.ingestion_version,
        chunk_size_tokens=config.chunk_size_tokens,
        chunk_overlap_tokens=config.chunk_overlap_tokens,
        tokenizer_backend=tokenizer.backend,
    )

    for doc in documents:
        if not doc.succeeded:
            logger.error("FAILED to ingest %s: %s", doc.filename, doc.file_error)
        elif doc.page_failures:
            logger.warning("%s: %d page(s) failed to extract", doc.filename, len(doc.page_failures))

    logger.info("Ingestion summary: %s", result.summary())
    return result


if __name__ == "__main__":
    # Quick manual check: `python -m src.ingestion`
    cfg = AppConfig.from_env()
    run_result = run_ingestion(cfg)
    print(run_result.summary())
