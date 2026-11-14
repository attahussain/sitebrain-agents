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
    max_research_attempts: int = field(default_factory=lambda: int(_env("SITEBRAIN_MAX_ATTEMPTS", "2")))
    # API service: SQLite file for LangGraph checkpoints (paused RFIs survive restarts).
    checkpoint_db: Path = field(default_factory=lambda: Path(_env("SITEBRAIN_CHECKPOINT_DB", str(ROOT / "data" / "checkpoints.sqlite"))))
    # Answer cache: Redis when REDIS_URL is set, otherwise in-process. TTL in seconds; 0 disables caching.
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", ""))
    cache_ttl: int = field(default_factory=lambda: int(_env("SITEBRAIN_CACHE_TTL", "3600")))


def get_settings() -> Settings:
    return Settings()
