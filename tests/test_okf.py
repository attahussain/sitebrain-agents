import shutil
import uuid
from datetime import datetime, timezone

from langgraph.types import Command

from sitebrain.config import ROOT, get_settings
from sitebrain.graph import build_graph
from sitebrain.okf import load_bundle, resolve_link, searchable, split_frontmatter, trust_tier
from sitebrain.store import DocStore

BUNDLE = ROOT / "data" / "okf"


def test_bundle_follows_the_spec():
    concepts = load_bundle(BUNDLE)
    assert "index" not in concepts and "log" not in concepts  # reserved filenames are not concepts
    rfi = concepts["decisions/rfi-021-retarder-discharge-time"]
    assert rfi.type == "RFI Response" and rfi.trust == "human-reviewed"
    assert concepts["lessons/level-15-cover-failure"].trust == "unverified"
    assert concepts["specs/hot-weather-concreting"].links[0] == "decisions/rfi-021-retarder-discharge-time"
    assert "specs/hot-weather-concreting-rev0" not in {c.concept_id for c in searchable(concepts)}


def test_trust_tiers_and_link_resolution():
    assert trust_tier(None) == "unverified"
    assert trust_tier({"by": "human:qa", "at": "2026-01-01T00:00:00Z"}) == "human-reviewed"  # bare mapping
    assert trust_tier([{"by": "process:nightly"}]) == "machine-confirmed"
    assert resolve_link("specs/a", "/decisions/b.md") == "decisions/b"
    assert resolve_link("specs/a", "../lessons/c.md") == "lessons/c"
    assert resolve_link("specs/a", "https://example.com/x.md") is None


def test_stale_after_is_an_absolute_instant():
    later = datetime(2027, 5, 1, tzinfo=timezone.utc)
    assert load_bundle(BUNDLE, now=later)["specs/hot-weather-concreting"].stale
    assert not load_bundle(BUNDLE)["specs/hot-weather-concreting"].stale


def test_search_follows_links_and_never_returns_deprecated(offline_settings):
    store = DocStore(offline_settings)
    hits = store.search("hot-weather concreting discharge time limit")
    ids = [h["chunk_id"] for h in hits]
    assert "okf:specs/hot-weather-concreting" in ids
    assert "okf:decisions/rfi-021-retarder-discharge-time" in ids  # reached by keyword or by the link
    for query in ("tender 35 C superseded Rev 0", "hot-weather concreting Rev 0"):
        assert all("rev0" not in h["chunk_id"] for h in store.search(query))
    curated = next(h for h in hits if h["kind"] == "okf")
    assert curated["trust"] == "human-reviewed"


def test_docs_only_mode_has_no_curated_rows(offline_settings, monkeypatch, tmp_path):
    monkeypatch.setenv("SITEBRAIN_SOURCES", "docs")
    monkeypatch.setenv("SITEBRAIN_DB", str(tmp_path / "docs_only"))
    store = DocStore(get_settings())
    store.ingest()
    assert all(h["kind"] == "doc" for h in store.search("discharge time retarder"))


def test_approved_rfi_is_written_back_as_a_draft_concept(offline_settings, monkeypatch, tmp_path):
    bundle = tmp_path / "okf"
    shutil.copytree(BUNDLE, bundle)
    monkeypatch.setenv("SITEBRAIN_OKF", str(bundle))
    monkeypatch.setenv("SITEBRAIN_OKF_WRITEBACK", "1")
    graph, _ = build_graph(settings=get_settings())
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    graph.invoke({"question": "Draft an RFI about the Level 15 column cover failure"}, config)
    out = graph.invoke(Command(resume={"approved": True, "note": "ok", "reviewer": "QA Manager"}), config)

    written = list((bundle / "rfis").glob("*.md"))
    assert len(written) == 1 and any("knowledge bundle" in t for t in out["trace"])
    meta, body = split_frontmatter(written[0].read_text(encoding="utf-8"))
    assert meta["type"] == "Open RFI" and meta["status"] == "draft"
    assert meta["approved_for_release"]["by"] == "human:qa-manager"
    assert "verified" not in meta  # releasing an RFI is not the same as verifying its answer
    assert load_bundle(bundle)[written[0].relative_to(bundle).with_suffix("").as_posix()].trust == "unverified"
