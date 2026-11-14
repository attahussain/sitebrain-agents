"""Document ingestion and vector search on LanceDB."""

import hashlib
import math
import re
from pathlib import Path

import lancedb
from lancedb.index import FTS

from sitebrain.config import Settings
from sitebrain.okf import load_bundle, searchable, to_row

TABLE = "site_docs"
_HASH_DIM = 256


class Embedder:
    """fastembed (BAAI/bge-small-en-v1.5, local ONNX, no API cost) or a hashing embedder for offline tests."""

    def __init__(self, kind: str):
        self.kind = kind
        self._model = None
        if kind == "fastembed":
            from fastembed import TextEmbedding

            self._model = TextEmbedding("BAAI/bge-small-en-v1.5")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is not None:
            return [v.tolist() for v in self._model.embed(texts)]
        return [_hash_embed(t) for t in texts]


def _hash_embed(text: str) -> list[float]:
    vec = [0.0] * _HASH_DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % _HASH_DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def chunk_markdown(doc_id: str, text: str, max_chars: int = 700) -> list[dict]:
    """One chunk per `##` section, prefixed with "<doc title> - <section>".

    Section-level chunks keep unrelated topics apart (a scaffold rule must not share an embedding with
    heat-stress rules), and the prefix carries context the section body alone does not.
    Oversized sections are packed by paragraph up to ~max_chars.
    """
    lines = text.splitlines()
    title = next((ln.lstrip("# ").strip() for ln in lines if ln.startswith("# ")), doc_id)
    sections: list[tuple[str, list[str]]] = [("", [])]
    for ln in lines:
        if ln.startswith("## "):
            sections.append((ln[3:].strip(), []))
        elif not ln.startswith("# "):
            sections[-1][1].append(ln)

    chunks: list[str] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        prefix = f"{title} - {heading}" if heading else title
        buf = ""
        for para in (p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()):
            if buf and len(buf) + len(para) > max_chars:
                chunks.append(f"{prefix}\n{buf}")
                buf = ""
            buf = f"{buf}\n\n{para}" if buf else para
        if buf:
            chunks.append(f"{prefix}\n{buf}")
    return [
        {"chunk_id": f"{doc_id}#{i}", "doc_id": doc_id, "title": title, "text": c}
        for i, c in enumerate(chunks)
    ]


class DocStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = Embedder(settings.embedder)
        settings.db_dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(settings.db_dir))
        self._concepts = None

    @property
    def concepts(self):
        if self._concepts is None:
            self._concepts = load_bundle(self.settings.okf_dir)
        return self._concepts

    def ingest(self, docs_dir: Path | None = None) -> int:
        docs_dir = docs_dir or self.settings.docs_dir
        rows: list[dict] = []
        if self.settings.sources in ("docs", "both"):
            for path in sorted(docs_dir.glob("*.md")):
                for row in chunk_markdown(path.stem, path.read_text(encoding="utf-8")):
                    rows.append({**row, "kind": "doc", "trust": "", "status": ""})
        if self.settings.sources in ("okf", "both"):
            self._concepts = load_bundle(self.settings.okf_dir)
            rows.extend(to_row(c) for c in searchable(self._concepts))
        vectors = self.embedder.embed([r["text"] for r in rows])
        for row, vec in zip(rows, vectors):
            row["vector"] = vec
        table = self.db.create_table(TABLE, data=rows, mode="overwrite")
        # BM25 inverted index over chunk text, used by keyword and hybrid search.
        table.create_index("text", config=FTS(), replace=True)
        return len(rows)

    def search(self, query: str, k: int | None = None, mode: str | None = None) -> list[dict]:
        """Retrieve the top-k chunks. mode: "vector", "keyword" (BM25) or "hybrid" (both, fused by RRF)."""
        k = k or self.settings.top_k
        mode = mode or self.settings.search_mode
        if mode == "vector":
            hits = self._vector(query, k)
        elif mode == "keyword":
            hits = self._keyword(query, k)
        elif mode == "hybrid":
            pool = max(k * 3, 10)  # fuse deeper candidate lists than we return
            vector_hits, keyword_hits = self._vector(query, pool), self._keyword(query, pool)
            if has_identifier(query) and keyword_hits:
                hits = lexical_first(keyword_hits, vector_hits, k)
            else:
                hits = reciprocal_rank_fusion([vector_hits, keyword_hits], k)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
        return self.follow_links(hits)

    def follow_links(self, hits: list[dict]) -> list[dict]:
        """Add concepts linked from retrieved OKF concepts, up to the link budget.

        A requirement links to the decision that amends it, and a spec links to its lesson learned;
        those neighbours often hold the exception or context a similarity search does not rank.
        """
        budget = self.settings.okf_link_budget
        if self.settings.sources == "docs" or budget <= 0:
            return hits
        present = {h["chunk_id"] for h in hits}
        added: list[dict] = []
        for hit in hits:
            concept = self.concepts.get(hit["doc_id"][4:]) if hit.get("kind") == "okf" else None
            for target in concept.links if concept else []:
                linked = self.concepts.get(target)
                if linked is None or linked.status == "deprecated" or f"okf:{target}" in present:
                    continue
                present.add(f"okf:{target}")
                added.append({**to_row(linked), "score": 0.0, "sources": ["link"]})
                if len(added) >= budget:
                    return hits + added
        return hits + added

    def _vector(self, query: str, k: int) -> list[dict]:
        table = self.db.open_table(TABLE)
        hits = table.search(self.embedder.embed([query])[0]).metric("cosine").limit(k).to_list()
        return [_row(h, score=1.0 - float(h["_distance"]), source="vector") for h in hits]

    def _keyword(self, query: str, k: int) -> list[dict]:
        table = self.db.open_table(TABLE)
        # Strip characters the full-text query parser treats as operators.
        cleaned = re.sub(r"[^\w\s-]", " ", query)
        hits = table.search(cleaned, query_type="fts").limit(k).to_list()
        return [_row(h, score=float(h["_score"]), source="keyword") for h in hits]


def _row(hit: dict, score: float, source: str) -> dict:
    return {
        "chunk_id": hit["chunk_id"],
        "doc_id": hit["doc_id"],
        "title": hit["title"],
        "text": hit["text"],
        "kind": hit.get("kind", "doc"),
        "trust": hit.get("trust", ""),
        "score": round(score, 4),
        "sources": [source],
    }


# Codes such as INS-1007, TC-2, C15-01, or long numbers such as 03300: letters/digits joined by
# hyphens with at least one digit, or 4+ digit runs.
_IDENTIFIER = re.compile(r"\b(?=[\w-]*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b|\b\d{4,}\b")


def has_identifier(query: str) -> bool:
    return bool(_IDENTIFIER.search(query))


def lexical_first(keyword_hits: list[dict], vector_hits: list[dict], k: int) -> list[dict]:
    """Keyword ranking order, back-filled with vector hits not already present.

    Used for identifier queries. An embedding of "INS-1007" says little about which of many
    look-alike reports holds it, while BM25 matches it exactly. Weighted RRF cannot express
    "keyword wins": adjacent top ranks differ by ~1/3600 (1/61 vs 1/62), so even a small vector
    vote reorders them.
    """
    seen = {h["chunk_id"] for h in keyword_hits[:k]}
    backfill = [h for h in vector_hits if h["chunk_id"] not in seen]
    return (keyword_hits[:k] + backfill)[:k]


def reciprocal_rank_fusion(rankings: list[list[dict]], k: int, c: int = 60) -> list[dict]:
    """Merge ranked lists by RRF: score(d) = sum over lists of 1 / (c + rank).

    RRF needs no score normalisation, which matters here because cosine similarity and BM25
    scores live on unrelated scales.
    """
    fused: dict[str, dict] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            entry = fused.setdefault(hit["chunk_id"], {**hit, "score": 0.0, "sources": []})
            entry["score"] += 1.0 / (c + rank)
            entry["sources"] = sorted(set(entry["sources"]) | set(hit["sources"]))
    ranked = sorted(fused.values(), key=lambda h: h["score"], reverse=True)[:k]
    for h in ranked:
        h["score"] = round(h["score"], 5)
    return ranked
