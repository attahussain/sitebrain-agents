"""Visual retrieval plumbing, tested with a tiny fake embedder (no torch, no model download).

Model quality is measured separately by evals/compare_visual.py; these tests cover indexing,
multi-vector (MaxSim) search, routing and how page images reach the writer.
"""

import hashlib
import re
import uuid

import lancedb
import numpy as np
import pytest

from sitebrain.agents import _content
from sitebrain.config import ROOT
from sitebrain.graph import build_graph
from sitebrain.visual import PageStore

# What each page "shows", standing in for what a real model reads from the image.
PAGE_TEXT = {
    "S-215": "structural drawing level 15 column layout c15-07 enlarged revision c",
    "S-214": "structural drawing level 14 column layout c14-11 enlarged revision a",
    "BBS-15": "bar bending schedule level 15 columns diameter links",
    "LOG-P2": "site logistics plan tower crane tc-2 gate 2 pump",
}


class FakePageEmbedder:
    name = "fake"
    dim = 64

    def _vec(self, token: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        v[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] = 1.0
        return v

    def _tokens(self, text: str) -> np.ndarray:
        return np.stack([self._vec(t) for t in re.findall(r"[a-z0-9-]+", text.lower())])

    def embed_pages(self, paths):
        return [self._tokens(PAGE_TEXT[p.stem]) for p in paths]

    def embed_query(self, text):
        return self._tokens(text)


@pytest.fixture()
def page_store(tmp_path, offline_settings):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    for page_id in PAGE_TEXT:  # real images from the repo, under ids the fake embedder knows
        (pages_dir / f"{page_id}.png").write_bytes((ROOT / "data" / "pages" / f"{page_id}.png").read_bytes())
    store = PageStore(offline_settings, embedder=FakePageEmbedder(), db=lancedb.connect(str(tmp_path / "pagedb")))
    store.index(pages_dir)
    return store


def test_maxsim_search_ranks_the_matching_page_first(page_store):
    hits = page_store.search("Which drawing shows the level 15 column layout?", k=2)
    assert hits[0]["page_id"] == "S-215"
    assert hits[0]["chunk_id"] == "page:S-215"


def test_visual_question_routes_to_pages_and_cites_them(offline_settings, page_store):
    graph, _ = build_graph(settings=offline_settings, page_store=page_store)
    out = graph.invoke({"question": "Which drawing shows the level 15 column layout?"},
                       {"configurable": {"thread_id": str(uuid.uuid4())}})
    assert out["search_pages"] is True
    assert [p["page_id"] for p in out["pages"]][0] == "S-215"
    assert out["citations"] == ["page:S-215"]
    assert out["critique"]["grounded"] is True  # page citations count as valid sources


def test_text_question_does_not_search_pages(offline_settings, page_store):
    graph, _ = build_graph(settings=offline_settings, page_store=page_store)
    out = graph.invoke({"question": "What is the minimum concrete cover for columns?"},
                       {"configurable": {"thread_id": str(uuid.uuid4())}})
    assert out["search_pages"] is False and out["pages"] == []


def test_page_images_are_sent_as_image_blocks():
    page = {"page_id": "S-215", "chunk_id": "page:S-215", "path": str(ROOT / "data" / "pages" / "S-215.png")}
    blocks = _content("Question: which drawing?", [page])
    kinds = [b["type"] for b in blocks]
    assert kinds == ["text", "image", "text"]
    assert blocks[1]["source"]["media_type"] == "image/png" and len(blocks[1]["source"]["data"]) > 1000
    assert _content("plain", []) == "plain"
