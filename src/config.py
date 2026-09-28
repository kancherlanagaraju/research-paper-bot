"""Centralized configuration for the Research Paper Answer Bot.

Settings are loaded from environment variables (via a local ``.env`` file if
present) with sensible defaults. Fields needed only by later pipeline stages
(embeddings, Zilliz vector store, reranking, LLM generation) are declared
here too, so the full configuration surface is visible in one place -- but
they are validated lazily, only when a component that actually needs them is
used. This means ingestion works with zero credentials configured.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Repository root (this file lives in <root>/src/config.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_DEFAULT_DATASET_DIR = PROJECT_ROOT / "pinnacle_capstone_data"
_DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid.

    Messages are written to be actionable: they say which variable is
    missing/invalid and where to set it.
    """


def _load_dotenv_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    else:
        # Still allow variables exported directly into the shell environment.
        load_dotenv(override=False)


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _get_optional_str(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value else None


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}='{raw}' is not a valid integer.") from exc


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name}='{raw}' is not a valid float.") from exc


@dataclass
class AppConfig:
    """Fully-resolved application configuration.

    Build with :meth:`AppConfig.from_env` rather than constructing directly,
    so environment variables and ``.env`` values are applied consistently.
    """

    # --- Paths --------------------------------------------------------- #
    project_root: Path = PROJECT_ROOT
    dataset_dir: Path = _DEFAULT_DATASET_DIR
    artifacts_dir: Path = _DEFAULT_ARTIFACTS_DIR

    # --- Ingestion / chunking ------------------------------------------- #
    ingestion_version: str = "v1"
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 120
    # "approx_word": offline, deterministic, no network call (default).
    # "tiktoken": exact OpenAI cl100k_base tokens; requires network access
    # on first use to download the vocabulary file.
    tokenizer_backend: str = "approx_word"
    tokenizer_encoding_name: str = "cl100k_base"

    # --- Vector store: Zilliz Cloud Serverless -------------------------- #
    zilliz_uri: Optional[str] = None
    zilliz_token: Optional[str] = None
    zilliz_collection_prefix: str = "research_paper_bot"

    # --- Embedding models ------------------------------------------------ #
    embedding_model_oss: str = "BAAI/bge-small-en-v1.5"
    embedding_model_openai: str = "text-embedding-3-small"
    openai_api_key: Optional[str] = None

    # --- Retrieval --------------------------------------------------------- #
    top_k_dense: int = 5
    hybrid_candidate_k: int = 20
    rerank_top_k: int = 5
    hybrid_dense_backend: str = "oss"  # dense side of hybrid: "oss" | "openai"
    fusion_method: str = "rrf"  # "rrf" | "weighted"
    fusion_weight_dense: float = 0.5
    fusion_weight_sparse: float = 0.5
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM / generation ------------------------------------------------- #
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # --- Misc -------------------------------------------------------------- #
    log_level: str = "INFO"

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls) -> "AppConfig":
        _load_dotenv_file()
        cfg = cls(
            dataset_dir=Path(_get_str("DATASET_DIR", str(_DEFAULT_DATASET_DIR))).expanduser(),
            artifacts_dir=Path(_get_str("ARTIFACTS_DIR", str(_DEFAULT_ARTIFACTS_DIR))).expanduser(),
            ingestion_version=_get_str("INGESTION_VERSION", "v1"),
            chunk_size_tokens=_get_int("CHUNK_SIZE_TOKENS", 800),
            chunk_overlap_tokens=_get_int("CHUNK_OVERLAP_TOKENS", 120),
            tokenizer_backend=_get_str("TOKENIZER_BACKEND", "approx_word"),
            tokenizer_encoding_name=_get_str("TOKENIZER_ENCODING_NAME", "cl100k_base"),
            zilliz_uri=_get_optional_str("ZILLIZ_URI"),
            zilliz_token=_get_optional_str("ZILLIZ_TOKEN"),
            zilliz_collection_prefix=_get_str("ZILLIZ_COLLECTION_PREFIX", "research_paper_bot"),
            embedding_model_oss=_get_str("EMBEDDING_MODEL_OSS", "BAAI/bge-small-en-v1.5"),
            embedding_model_openai=_get_str("EMBEDDING_MODEL_OPENAI", "text-embedding-3-small"),
            openai_api_key=_get_optional_str("OPENAI_API_KEY"),
            top_k_dense=_get_int("TOP_K_DENSE", 5),
            hybrid_candidate_k=_get_int("HYBRID_CANDIDATE_K", 20),
            rerank_top_k=_get_int("RERANK_TOP_K", 5),
            hybrid_dense_backend=_get_str("HYBRID_DENSE_BACKEND", "oss"),
            fusion_method=_get_str("FUSION_METHOD", "rrf"),
            fusion_weight_dense=_get_float("FUSION_WEIGHT_DENSE", 0.5),
            fusion_weight_sparse=_get_float("FUSION_WEIGHT_SPARSE", 0.5),
            reranker_model=_get_str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            llm_provider=_get_str("LLM_PROVIDER", "openai"),
            llm_model=_get_str("LLM_MODEL", "gpt-4o-mini"),
            log_level=_get_str("LOG_LEVEL", "INFO"),
        )
        cfg.validate_ingestion()
        return cfg

    # ------------------------------------------------------------------ #
    # Validation. Only the ingestion settings are validated eagerly
    # (in from_env). Everything else is validated on demand via the
    # require_* helpers below, with an actionable error message, instead of
    # failing at import time.
    # ------------------------------------------------------------------ #
    def validate_ingestion(self) -> None:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ConfigError(
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS "
                f"(got overlap={self.chunk_overlap_tokens}, size={self.chunk_size_tokens})."
            )
        if self.chunk_size_tokens <= 0 or self.chunk_overlap_tokens < 0:
            raise ConfigError("CHUNK_SIZE_TOKENS must be positive and CHUNK_OVERLAP_TOKENS must be non-negative.")
        if self.tokenizer_backend not in {"approx_word", "tiktoken"}:
            raise ConfigError(
                f"TOKENIZER_BACKEND must be 'approx_word' or 'tiktoken', got '{self.tokenizer_backend}'."
            )
        if not self.dataset_dir.exists():
            raise ConfigError(
                f"Dataset directory not found: {self.dataset_dir}. Set DATASET_DIR in your .env file "
                "(see .env.example) or place PDFs under the default 'pinnacle_capstone_data' folder."
            )

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Add it to your .env file (see .env.example) before "
                "running commercial-embedding or LLM-generation steps."
            )
        return self.openai_api_key

    def require_zilliz_credentials(self) -> Tuple[str, str]:
        missing = [name for name, value in (("ZILLIZ_URI", self.zilliz_uri), ("ZILLIZ_TOKEN", self.zilliz_token)) if not value]
        if missing:
            raise ConfigError(
                f"Missing Zilliz Cloud Serverless credential(s): {', '.join(missing)}. "
                "Create a free cluster at https://cloud.zilliz.com and add them to your .env file "
                "(see .env.example) before running vector-store steps."
            )
        return self.zilliz_uri, self.zilliz_token  # type: ignore[return-value]

    def configure_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
