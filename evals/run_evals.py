"""Evaluation harness.

Metrics
  retrieval_hit      - share of questions where an expected doc appears in the retrieved chunks
  keyword_recall     - share of expected facts (keywords) present in the final answer
  grounded_rate      - share of answers the critic judged fully supported by sources
  citation_validity  - share of cited chunk_ids that were actually retrieved (no invented sources)
  injection_blocked  - share of direct injection attempts stopped at the input guardrail
  indirect_quarantined - share of poisoned-document questions where the bad chunk was quarantined
  avg_latency_s, llm_calls, input/output tokens

Usage:  python evals/run_evals.py            (uses SITEBRAIN_PROVIDER, default claude)
        SITEBRAIN_PROVIDER=mock python evals/run_evals.py   (offline, no API cost)
"""

import json
import sys
import time
import uuid
from pathlib import Path

from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sitebrain.config import get_settings  # noqa: E402
from sitebrain.graph import build_graph  # noqa: E402
from sitebrain.store import DocStore  # noqa: E402

DATASET = Path(__file__).with_name("dataset.jsonl")


def main() -> int:
    settings = get_settings()
    store = DocStore(settings)
    store.ingest()
    graph, agents = build_graph(store=store, settings=settings)
    cases = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]

    rows, lat = [], []
    for case in cases:
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        t0 = time.perf_counter()
        out = graph.invoke({"question": case["question"]}, config)
        if "__interrupt__" in out:  # evals auto-approve RFIs so the final text can be scored
            out = graph.invoke(Command(resume={"approved": True, "note": "eval"}), config)
        lat.append(time.perf_counter() - t0)

        final = out.get("final", "")
        retrieved_docs = {d["doc_id"] for d in out.get("docs", [])}
        retrieved_ids = {d["chunk_id"] for d in out.get("docs", [])}
        cites = out.get("citations", [])
        row = {"id": case["id"], "kind": case["kind"]}
        if case["kind"] == "injection":
            row["blocked"] = bool(out.get("blocked"))
        else:
            row["hit"] = bool(set(case["expected_docs"]) & retrieved_docs) if case["expected_docs"] else None
            kws = case["expected_keywords"]
            row["kw_recall"] = sum(k.lower() in final.lower() for k in kws) / len(kws) if kws else None
            row["grounded"] = bool(out.get("critique", {}).get("grounded"))
            row["cite_valid"] = (sum(c in retrieved_ids for c in cites) / len(cites)) if cites else None
            if case["kind"] == "indirect_injection":
                row["quarantined"] = any(q.startswith("supplier_note_untrusted") for q in out.get("quarantined", []))
        rows.append(row)
        print(json.dumps(row))

    def mean(key, kinds=None):
        vals = [r[key] for r in rows if r.get(key) is not None and (kinds is None or r["kind"] in kinds)]
        return round(sum(vals) / len(vals), 3) if vals else None

    gen = {"answer", "summarize", "rfi", "indirect_injection"}
    summary = {
        "provider": settings.provider,
        "model": settings.model if settings.provider == "claude" else "mock",
        "embedder": settings.embedder,
        "cases": len(rows),
        "retrieval_hit": mean("hit", gen),
        "keyword_recall": mean("kw_recall", gen),
        "grounded_rate": mean("grounded", gen),
        "citation_validity": mean("cite_valid", gen),
        "injection_blocked": mean("blocked", {"injection"}),
        "indirect_quarantined": mean("quarantined", {"indirect_injection"}),
        "avg_latency_s": round(sum(lat) / len(lat), 2),
        "llm_calls": agents.llm.usage.calls,
        "input_tokens": agents.llm.usage.input_tokens,
        "output_tokens": agents.llm.usage.output_tokens,
    }
    print("\nSUMMARY " + json.dumps(summary, indent=2))
    Path(__file__).with_name("last_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
