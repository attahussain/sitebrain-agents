"""Retrieval-only comparison of vector, keyword (BM25) and hybrid (RRF) search.

No LLM calls, so it is free to run. For every dataset case with expected documents it reports:
  hit@1  - the top-ranked chunk comes from an expected document
  hit@k  - any of the top-k chunks comes from an expected document
  MRR    - mean reciprocal rank of the first chunk from an expected document

Usage:  python evals/compare_retrieval.py                 # demo corpus (data/docs)
        python evals/make_scale_corpus.py && python evals/compare_retrieval.py --scale
                                                         # 60 near-identical synthetic daily reports
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitebrain.config import get_settings  # noqa: E402
from sitebrain.store import DocStore  # noqa: E402

DATASET = Path(__file__).with_name("dataset.jsonl")
MODES = ["vector", "keyword", "hybrid"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", action="store_true", help="use the synthetic scale corpus")
    args = parser.parse_args()

    dataset = DATASET
    if args.scale:
        root = Path(__file__).resolve().parent.parent
        os.environ["SITEBRAIN_DOCS"] = str(root / "data" / "scale_docs")
        os.environ["SITEBRAIN_DB"] = str(Path(tempfile.gettempdir()) / "sitebrain_scale_db")
        dataset = Path(__file__).with_name("scale_queries.jsonl")
        if not dataset.exists():
            sys.exit("Run evals/make_scale_corpus.py first.")

    settings = get_settings()
    store = DocStore(settings)
    store.ingest()
    k = settings.top_k
    cases = [json.loads(ln) for ln in dataset.read_text(encoding="utf-8").splitlines() if ln.strip()]
    cases = [c for c in cases if c["expected_docs"]]

    results: dict[str, dict] = {}
    misses: dict[str, list[str]] = {m: [] for m in MODES}
    for mode in MODES:
        hit1 = hitk = rr = 0.0
        for case in cases:
            ranked = [h["doc_id"] for h in store.search(case["question"], k=k, mode=mode)]
            rank = next((i for i, d in enumerate(ranked, 1) if d in case["expected_docs"]), None)
            hit1 += rank == 1
            hitk += rank is not None
            rr += 1.0 / rank if rank else 0.0
            if rank != 1:
                misses[mode].append(f"{case['id']} (rank {rank})")
        n = len(cases)
        results[mode] = {"hit@1": round(hit1 / n, 3), f"hit@{k}": round(hitk / n, 3), "MRR": round(rr / n, 3)}

    print(f"{len(cases)} cases, embedder={settings.embedder}, k={k}\n")
    print(f"{'mode':<8} " + "  ".join(f"{m:>7}" for m in results["vector"]))
    for mode, r in results.items():
        print(f"{mode:<8} " + "  ".join(f"{v:>7}" for v in r.values()))
    print("\nnot ranked first:")
    for mode, ids in misses.items():
        print(f"  {mode:<8} {', '.join(ids) or '-'}")
    out = "retrieval_results_scale.json" if args.scale else "retrieval_results.json"
    Path(__file__).with_name(out).write_text(json.dumps({"k": k, "results": results, "misses": misses}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
