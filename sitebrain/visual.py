"""Visual page retrieval: search document pages as images (drawings, schedules, forms, plans).

ColPali-family models embed a page screenshot directly, as one vector per image patch, and score a
query by late interaction (MaxSim: each query token takes its best-matching patch, then sum). No
OCR or layout parsing is involved, so title blocks, table cells and ticked boxes stay searchable.

Defaults, picked from the hardware (SITEBRAIN_VISUAL_MODEL=auto):
  GPU with >= 12 GB VRAM   vidore/colqwen2.5-v0.2   ~3B params, bf16, most accurate
  smaller GPU, or CPU      vidore/colSmol-256M      fast on GPU, practical on a laptop CPU
Override with SITEBRAIN_VISUAL_MODEL / SITEBRAIN_VISUAL_DEVICE.

Requires the optional dependencies:  pip install -e ".[visual]"
"""

import time
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa

from sitebrain.config import Settings

PAGE_TABLE = "site_pages"
GPU_DEFAULT = "vidore/colqwen2.5-v0.2"
CPU_DEFAULT = "vidore/colSmol-256M"


class PageEmbedder(Protocol):
    name: str

    def embed_pages(self, paths: list[Path]) -> list[np.ndarray]: ...  # one (patches, dim) array per page
    def embed_query(self, text: str) -> np.ndarray: ...  # (tokens, dim)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class ColPaliEmbedder:
    """Any colpali-engine late-interaction model (ColQwen2.5, ColPali, ColSmol/Idefics3)."""

    def __init__(self, model_name: str, device: str, batch_size: int = 1):
        import torch
        from colpali_engine import models

        lowered = model_name.lower()
        if "colqwen2.5" in lowered or "colqwen2_5" in lowered:
            model_cls, proc_cls = models.ColQwen2_5, models.ColQwen2_5_Processor
        elif "colqwen2" in lowered:
            model_cls, proc_cls = models.ColQwen2, models.ColQwen2Processor
        elif "colsmol" in lowered or "idefics3" in lowered:
            model_cls, proc_cls = models.ColIdefics3, models.ColIdefics3Processor
        elif "colpali" in lowered:
            model_cls, proc_cls = models.ColPali, models.ColPaliProcessor
        else:
            raise ValueError(f"Unsupported visual model: {model_name}")

        self.name = model_name
        self.device = device
        self._torch = torch
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = model_cls.from_pretrained(model_name, torch_dtype=dtype, device_map=device).eval()
        self.processor = proc_cls.from_pretrained(model_name)
        # Pages per forward pass. 1 is the safe default: on an 8 GB card a batch of 4 full-size pages
        # with ColQwen2.5 overflows VRAM into shared memory and indexing slows ~40x (see docs).
        self.batch_size = batch_size

    def _run(self, batch) -> list[np.ndarray]:
        with self._torch.no_grad():
            out = self.model(**batch.to(self.model.device))
        # Normalise per vector so LanceDB's cosine MaxSim matches the model's dot-product scoring.
        arrays = [e.float().cpu().numpy() for e in out]
        return [a / np.linalg.norm(a, axis=1, keepdims=True) for a in arrays]

    def embed_pages(self, paths: list[Path]) -> list[np.ndarray]:
        from PIL import Image

        vectors: list[np.ndarray] = []
        for i in range(0, len(paths), self.batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[i : i + self.batch_size]]
            vectors.extend(self._run(self.processor.process_images(images)))
        return vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self._run(self.processor.process_queries([text]))[0]


class ClipEmbedder:
    """Single-vector baseline (CLIP ViT-B/32 via fastembed): one embedding per page and per query."""

    name = "clip-ViT-B-32"

    def __init__(self):
        from fastembed import ImageEmbedding, TextEmbedding

        self._img = ImageEmbedding("Qdrant/clip-ViT-B-32-vision")
        self._txt = TextEmbedding("Qdrant/clip-ViT-B-32-text")

    def embed_pages(self, paths: list[Path]) -> list[np.ndarray]:
        return [np.asarray(v, dtype="float32")[None, :] for v in self._img.embed([str(p) for p in paths])]

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(next(iter(self._txt.embed([text]))), dtype="float32")[None, :]


LARGE_MODEL_MIN_VRAM_GB = 12


def auto_model(device: str) -> str:
    """ColQwen2.5 only when the GPU has room for it.

    Measured on an 8 GB RTX 4070: ColQwen2.5 weights alone push VRAM to ~7.9 GB, the driver spills
    into shared memory and indexing takes ~11 s/page, versus 0.4 s/page for ColSmol on the same card
    (accuracy 92% vs 83% hit@1, both 100% hit@3). See docs/evaluation.md.
    """
    if device != "cuda":
        return CPU_DEFAULT
    import torch

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return GPU_DEFAULT if vram_gb >= LARGE_MODEL_MIN_VRAM_GB else CPU_DEFAULT


def make_page_embedder(settings: Settings) -> PageEmbedder:
    if settings.visual_model == "clip":
        return ClipEmbedder()
    device = resolve_device(settings.visual_device)
    model = auto_model(device) if settings.visual_model == "auto" else settings.visual_model
    return ColPaliEmbedder(model, device, batch_size=settings.visual_batch)


class PageStore:
    """Page images and their multi-vector embeddings in LanceDB; search scores by MaxSim."""

    def __init__(self, settings: Settings, embedder: PageEmbedder | None = None, db=None):
        import lancedb

        self.settings = settings
        self.embedder = embedder or make_page_embedder(settings)
        self.db = db or lancedb.connect(str(settings.db_dir))
        self.last_index_seconds = 0.0

    def index(self, pages_dir: Path | None = None) -> int:
        paths = sorted((pages_dir or self.settings.pages_dir).glob("*.png"))
        start = time.perf_counter()
        vectors = self.embedder.embed_pages(paths)
        self.last_index_seconds = time.perf_counter() - start
        dim = vectors[0].shape[1]
        schema = pa.schema([
            pa.field("page_id", pa.string()),
            pa.field("path", pa.string()),
            pa.field("vector", pa.list_(pa.list_(pa.float32(), dim))),
        ])
        rows = [{"page_id": p.stem, "path": str(p), "vector": v.astype("float32").tolist()} for p, v in zip(paths, vectors)]
        self.db.create_table(PAGE_TABLE, data=rows, schema=schema, mode="overwrite")
        return len(rows)

    def search(self, query: str, k: int = 2) -> list[dict]:
        table = self.db.open_table(PAGE_TABLE)
        qv = self.embedder.embed_query(query).astype("float32").tolist()
        hits = table.search(qv).metric("cosine").limit(k).to_list()
        return [{"page_id": h["page_id"], "chunk_id": f"page:{h['page_id']}", "path": h["path"],
                 "score": round(-float(h["_distance"]), 4)} for h in hits]
