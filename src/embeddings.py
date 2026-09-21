"""Embedding model backends: open-source (sentence-transformers) and
commercial (OpenAI).

Two backends implement the same small interface so retrieval and evaluation
code (later milestones) can swap between them without caring which is
active:

* :class:`OSSEmbeddingBackend` -- ``sentence-transformers``, model name from
  ``AppConfig.embedding_model_oss`` (default ``BAAI/bge-small-en-v1.5``).
  Loads the model lazily (first call), so importing this module never
  triggers a network call or a multi-hundred-MB download.
* :class:`OpenAIEmbeddingBackend` -- OpenAI's embeddings API, model name
  from ``AppConfig.embedding_model_openai`` (default
  ``text-embedding-3-small``). Requires ``AppConfig.require_openai_key()``.

Both expose ``embed_passages(texts)`` (for indexing chunk text) and
``embed_query(text)`` (for a user's question at retrieval time), because
some open-source models (BGE in particular) recommend a different
instruction prefix for queries than for passages -- getting this right now
means the retrieval milestone doesn't have to special-case it later.

NOTE ON VERIFICATION: the OpenAI backend was smoke-tested live against the
real API (see README "Known limitations"). The OSS backend's *wiring* is
unit-tested with a stubbed ``SentenceTransformer`` (tests/test_embeddings.py);
actually downloading ``BAAI/bge-small-en-v1.5`` from Hugging Face could not
be completed from this sandboxed environment (its model-weight CDN host was
blocked at the network layer here) -- run `python -m src.embeddings` on a
machine with normal internet access to verify the live download once.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from src.config import AppConfig, ConfigError

logger = logging.getLogger(__name__)

# Models known to want a special instruction prefix on the QUERY side only
# (passages are embedded as-is). Matched by case-insensitive substring.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIX_BY_SUBSTRING = {
    "bge": _BGE_QUERY_PREFIX,
}

# Known output dimensions, so `.dimension` can be answered without an API
# call for OpenAI models (useful for planning a Zilliz collection schema
# before spending a request). Falls back to a live probe if unknown.
_KNOWN_OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend fails to load or to embed text."""


class EmbeddingBackend:
    """Common interface implemented by every embedding backend."""

    name: str  # the underlying model identifier, e.g. "BAAI/bge-small-en-v1.5"
    kind: str  # "oss" | "openai" -- used to pick a Zilliz collection

    def embed_passages(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError


# =========================================================================
# Open-source backend (sentence-transformers)
# =========================================================================


class OSSEmbeddingBackend(EmbeddingBackend):
    kind = "oss"

    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None
        self._query_prefix = next(
            (prefix for key, prefix in _QUERY_PREFIX_BY_SUBSTRING.items() if key in model_name.lower()),
            "",
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers is not installed. Install it with "
                "`pip install sentence-transformers` (see requirements.txt)."
            ) from exc
        try:
            logger.info("Loading open-source embedding model '%s' (first use only)...", self.name)
            self._model = SentenceTransformer(self.name)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load embedding model '{self.name}' from Hugging Face. This usually means "
                "there is no network access to Hugging Face's model storage. Verify network access, "
                "or pre-download the model on a machine that has it and point HF cache dirs at it."
            ) from exc

    def embed_passages(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        self._ensure_loaded()
        vectors = self._model.encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        self._ensure_loaded()
        vector = self._model.encode([self._query_prefix + text], normalize_embeddings=True)[0]
        return vector.tolist()

    @property
    def dimension(self) -> int:
        self._ensure_loaded()
        return int(self._model.get_sentence_embedding_dimension())


# =========================================================================
# Commercial backend (OpenAI)
# =========================================================================


class OpenAIEmbeddingBackend(EmbeddingBackend):
    kind = "openai"

    def __init__(self, model_name: str, api_key: str):
        self.name = model_name
        self._api_key = api_key
        self._client = None
        self._dimension_cache: Optional[int] = _KNOWN_OPENAI_DIMENSIONS.get(model_name)

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError(
                "openai is not installed. Install it with `pip install openai` (see requirements.txt)."
            ) from exc
        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        client = self._ensure_client()
        try:
            response = client.embeddings.create(model=self.name, input=texts)
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embeddings request failed for model '{self.name}': {exc}") from exc
        vectors = [item.embedding for item in response.data]
        if vectors and self._dimension_cache is None:
            self._dimension_cache = len(vectors[0])
        return vectors

    def embed_passages(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            vectors.extend(self._embed_batch(texts[i : i + batch_size]))
        return vectors

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    @property
    def dimension(self) -> int:
        if self._dimension_cache is None:
            self._embed_batch(["dimension probe"])
        return self._dimension_cache  # type: ignore[return-value]


# =========================================================================
# Factory
# =========================================================================


def get_backend(kind: str, config: AppConfig) -> EmbeddingBackend:
    """Build the requested backend ("oss" or "openai") from ``config``."""
    if kind == "oss":
        return OSSEmbeddingBackend(config.embedding_model_oss)
    if kind == "openai":
        api_key = config.require_openai_key()
        return OpenAIEmbeddingBackend(config.embedding_model_openai, api_key)
    raise ValueError(f"Unknown embedding backend kind '{kind}'. Use 'oss' or 'openai'.")


if __name__ == "__main__":
    # Manual live check: `python -m src.embeddings`
    cfg = AppConfig.from_env()
    for kind in ("oss", "openai"):
        try:
            backend = get_backend(kind, cfg)
            vec = backend.embed_query("What is attention in a transformer model?")
            print(f"{kind}: model={backend.name} dim={len(vec)} sample={vec[:5]}")
        except (EmbeddingError, ConfigError) as exc:
            print(f"{kind}: SKIPPED ({exc})")
