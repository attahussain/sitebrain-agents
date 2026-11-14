"""Document ingestion and vector search on LanceDB."""

import hashlib
import math
import re
from pathlib import Path

import lancedb

from sitebrain.config import Settings

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

    def ingest(self, docs_dir: Path | None = None) -> int:
        docs_dir = docs_dir or self.settings.docs_dir
        rows: list[dict] = []
        for path in sorted(docs_dir.glob("*.md")):
            rows.extend(chunk_markdown(path.stem, path.read_text(encoding="utf-8")))
        vectors = self.embedder.embed([r["text"] for r in rows])
        for row, vec in zip(rows, vectors):
            row["vector"] = vec
        self.db.create_table(TABLE, data=rows, mode="overwrite")
        return len(rows)

    def search(self, query: str, k: int | None = None) -> list[dict]:
        table = self.db.open_table(TABLE)
        qvec = self.embedder.embed([query])[0]
        hits = table.search(qvec).metric("cosine").limit(k or self.settings.top_k).to_list()
        return [
            {
                "chunk_id": h["chunk_id"],
                "doc_id": h["doc_id"],
                "title": h["title"],
                "text": h["text"],
                "score": round(1.0 - float(h["_distance"]), 4),
            }
            for h in hits
        ]
