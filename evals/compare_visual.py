"""Visual retrieval benchmark: CLIP (single vector) vs ColPali-family models on CPU and GPU.

Indexes data/pages/*.png with each configuration and scores evals/visual_queries.jsonl, whose answers
exist only inside the page images. Reports hit@1, hit@3, MRR, indexing time per page and query latency.

Usage:  python evals/compare_visual.py                    # every configuration this machine can run
        python evals/compare_visual.py --only cpu         # just the CPU configurations
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitebrain.config import get_settings  # noqa: E402
from sitebrain.visual import CPU_DEFAULT, GPU_DEFAULT, ClipEmbedder, ColPaliEmbedder, PageStore  # noqa: E402

QUERIES = Path(__file__).with_name("visual_queries.jsonl")


def configurations(only: str | None, batch: int = 1, model: str | None = None):
    import torch

    configs = [("clip (cpu)", "cpu", lambda: ClipEmbedder()),
               (f"{CPU_DEFAULT} (cpu)", "cpu", lambda: ColPaliEmbedder(CPU_DEFAULT, "cpu", batch))]
    if torch.cuda.is_available():
        configs += [(f"{CPU_DEFAULT} (gpu)", "gpu", lambda: ColPaliEmbedder(CPU_DEFAULT, "cuda", batch)),
                    (f"{GPU_DEFAULT} (gpu)", "gpu", lambda: ColPaliEmbedder(GPU_DEFAULT, "cuda", batch))]
    return [c for c in configs if (only is None or c[1] == only) and (model is None or model in c[0])]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["cpu", "gpu"])
    parser.add_argument("--batch", type=int, default=1, help="pages per forward pass for ColPali models")
    parser.add_argument("--model", help="only run configurations whose label contains this text")
    args = parser.parse_args()

    import lancedb

    settings = get_settings()
    cases = [json.loads(ln) for ln in QUERIES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    report = []
    for label, _, make in configurations(args.only, args.batch, args.model):
        embedder = make()
        store = PageStore(settings, embedder=embedder, db=lancedb.connect(tempfile.mkdtemp()))
        n_pages = store.index()
        per_page = store.last_index_seconds / n_pages
        hit1 = hit3 = rr = 0.0
        latencies, misses = [], []
        for case in cases:
            t0 = time.perf_counter()
            ranked = [h["page_id"] for h in store.search(case["question"], k=3)]
            latencies.append(time.perf_counter() - t0)
            rank = next((i for i, p in enumerate(ranked, 1) if p in case["expected_pages"]), None)
            hit1 += rank == 1
            hit3 += rank is not None
            rr += 1 / rank if rank else 0
            if rank != 1:
                misses.append(f"{case['id']}->{ranked[0]}")
        n = len(cases)
        row = {"config": f"{label} batch={args.batch}" if "col" in label.lower() else label, "hit@1": round(hit1 / n, 3), "hit@3": round(hit3 / n, 3), "MRR": round(rr / n, 3),
               "index_s_per_page": round(per_page, 2), "query_ms": round(1000 * sum(latencies) / n),
               "misses": misses}
        report.append(row)
        print(json.dumps(row), flush=True)
        del embedder, store
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(f"\n{len(cases)} questions over {n_pages} pages")
    print(f"{'configuration':<36} {'hit@1':>6} {'hit@3':>6} {'MRR':>6} {'index s/page':>13} {'query ms':>9}")
    for r in report:
        print(f"{r['config']:<36} {r['hit@1']:>6} {r['hit@3']:>6} {r['MRR']:>6} {r['index_s_per_page']:>13} {r['query_ms']:>9}")
    out = Path(__file__).with_name("visual_results.json")
    previous = json.loads(out.read_text()) if out.exists() else []
    keep = [r for r in previous if r["config"] not in {x["config"] for x in report}]
    out.write_text(json.dumps(keep + report, indent=2))  # accumulate runs made one configuration at a time
    return 0


if __name__ == "__main__":
    sys.exit(main())
