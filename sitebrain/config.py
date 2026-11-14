"""Runtime settings, read from environment variables so nothing secret lives in code."""

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # "claude" calls the Anthropic API; "mock" is a deterministic offline stand-in for tests/evals.
    provider: str = field(default_factory=lambda: _env("SITEBRAIN_PROVIDER", "claude"))
    model: str = field(default_factory=lambda: _env("SITEBRAIN_MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: _env("SITEBRAIN_EFFORT", "medium"))
    # "fastembed" = real local embeddings (BAAI/bge-small-en-v1.5); "hash" = tiny offline embedder for tests.
    embedder: str = field(default_factory=lambda: _env("SITEBRAIN_EMBEDDER", "fastembed"))
    docs_dir: Path = field(default_factory=lambda: Path(_env("SITEBRAIN_DOCS", str(ROOT / "data" / "docs"))))
    db_dir: Path = field(default_factory=lambda: Path(_env("SITEBRAIN_DB", str(ROOT / "data" / "lancedb"))))
    top_k: int = field(default_factory=lambda: int(_env("SITEBRAIN_TOP_K", "4")))
    # "hybrid" (BM25 + vector, fused by RRF), "vector" or "keyword".
    search_mode: str = field(default_factory=lambda: _env("SITEBRAIN_SEARCH_MODE", "hybrid"))
    # Curated knowledge: an OKF v0.2 bundle searched alongside the raw documents.
    okf_dir: Path = field(default_factory=lambda: Path(_env("SITEBRAIN_OKF", str(ROOT / "data" / "okf"))))
    # Which sources are indexed: "both" (default), "docs" (raw documents only) or "okf" (bundle only).
    sources: str = field(default_factory=lambda: _env("SITEBRAIN_SOURCES", "both"))
    # How many linked concepts the researcher may add by following OKF links (0 disables).
    okf_link_budget: int = field(default_factory=lambda: int(_env("SITEBRAIN_OKF_LINKS", "2")))
    # Write approved RFIs back into the bundle as draft "Open RFI" concepts.
    okf_writeback: bool = field(default_factory=lambda: _env("SITEBRAIN_OKF_WRITEBACK", "0") in ("1", "true", "on"))
    max_research_attempts: int = field(default_factory=lambda: int(_env("SITEBRAIN_MAX_ATTEMPTS", "2")))
    # Visual page retrieval (ColPali family). Off by default: it needs the [visual] extra and a model download.
    visual: bool = field(default_factory=lambda: _env("SITEBRAIN_VISUAL", "0") in ("1", "true", "on"))
    # "auto" = colqwen2.5 on GPU, colSmol-256M on CPU; or any colpali-engine model id; or "clip" (baseline).
    visual_model: str = field(default_factory=lambda: _env("SITEBRAIN_VISUAL_MODEL", "auto"))
    visual_device: str = field(default_factory=lambda: _env("SITEBRAIN_VISUAL_DEVICE", "auto"))
    visual_batch: int = field(default_factory=lambda: int(_env("SITEBRAIN_VISUAL_BATCH", "1")))
    pages_dir: Path = field(default_factory=lambda: Path(_env("SITEBRAIN_PAGES", str(ROOT / "data" / "pages"))))
    # API service: SQLite file for LangGraph checkpoints (paused RFIs survive restarts).
    checkpoint_db: Path = field(default_factory=lambda: Path(_env("SITEBRAIN_CHECKPOINT_DB", str(ROOT / "data" / "checkpoints.sqlite"))))
    # Answer cache: Redis when REDIS_URL is set, otherwise in-process. TTL in seconds; 0 disables caching.
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))
    cache_ttl: int = field(default_factory=lambda: int(_env("SITEBRAIN_CACHE_TTL", "3600")))


def get_settings() -> Settings:
    return Settings()
