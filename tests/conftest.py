import pytest

from sitebrain.config import ROOT


@pytest.fixture()
def offline_settings(tmp_path, monkeypatch):
    """Mock LLM + hashing embedder + a throwaway LanceDB: no API key or model download needed."""
    monkeypatch.setenv("SITEBRAIN_PROVIDER", "mock")
    monkeypatch.setenv("SITEBRAIN_EMBEDDER", "hash")
    monkeypatch.setenv("SITEBRAIN_DOCS", str(ROOT / "data" / "docs"))
    monkeypatch.setenv("SITEBRAIN_DB", str(tmp_path / "lancedb"))
    from sitebrain.config import get_settings
    from sitebrain.store import DocStore

    settings = get_settings()
    DocStore(settings).ingest()
    return settings
