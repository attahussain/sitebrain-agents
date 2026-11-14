"""Evaluation harness.

Metrics
  retrieval_hit      - share of questions where an expected doc appears in the retrieved chunks
  keyword_recall     - share of expected facts (keywords) present in the final answer
  grounded_rate      - share of answers the critic judged fully supported by sources
  citation_validity  - share of cited chunk_ids that were actually retrieved (no invented sources)
  injection_blocked  - share of direct injection attempts stopped at the input guardrail
  indirect_quarantined - share of poisoned-document questions where the bad chunk was quarantined
  avg_latency_s, llm_calls, input/output tokens

Visual mode (--visual) runs evals/visual_queries.jsonl with page retrieval on and also reports
  routed_to_pages    - share of questions the planner sent to the page index
  page_hit           - share of questions whose expected page was among the retrieved pages

Usage:  python evals/run_evals.py            (uses SITEBRAIN_PROVIDER, default claude)
        SITEBRAIN_PROVIDER=mock python evals/run_evals.py   (offline, no API cost)
        python evals/run_evals.py --visual   (needs the [visual] extra)
"""

import argparse
import json
import os
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


def _norm(text: str) -> str:
    """Compare facts, not formatting: "32 C", "32°C" and "32 °C" all normalise to "32c"."""
    return "".join(text.lower().replace("°", "").split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual", action="store_true", help="run the page-image questions")
    parser.add_argument("--okf", action="store_true", help="run the curated-knowledge (OKF bundle) questions")
    args = parser.parse_args()
    dataset = DATASET
    if args.okf:
        dataset = Path(__file__).with_name("okf_dataset.jsonl")
    if args.visual:
        os.environ["SITEBRAIN_VISUAL"] = "1"
        dataset = Path(__file__).with_name("visual_queries.jsonl")

    settings = get_settings()
    store = DocStore(settings)
    store.ingest()
    graph, agents = build_graph(store=store, settings=settings)
    if args.visual:
        agents.page_store.index()  # always re-index so results reflect the configured model
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    for case in cases:
        case.setdefault("kind", "visual")
        case.setdefault("expected_docs", [])

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
        retrieved_ids = {d["chunk_id"] for d in out.get("docs", [])} | {p["chunk_id"] for p in out.get("pages", [])}
        cites = out.get("citations", [])
        row = {"id": case["id"], "kind": case["kind"]}
        if case["kind"] == "injection":
            row["blocked"] = bool(out.get("blocked"))
        else:
            row["hit"] = bool(set(case["expected_docs"]) & retrieved_docs) if case["expected_docs"] else None
            kws = case["expected_keywords"]
            row["kw_recall"] = sum(_norm(k) in _norm(final) for k in kws) / len(kws) if kws else None
            if "must_not" in case:  # a superseded value in the answer means stale knowledge was used
                row["stale_free"] = not any(_norm(k) in _norm(final) for k in case["must_not"])
            row["grounded"] = bool(out.get("critique", {}).get("grounded"))
            row["cite_valid"] = (sum(c in retrieved_ids for c in cites) / len(cites)) if cites else None
            if case["kind"] == "visual":
                row["routed"] = bool(out.get("search_pages"))
                row["page_hit"] = bool(set(case["expected_pages"]) & {p["page_id"] for p in out.get("pages", [])})
            if case["kind"] == "indirect_injection":
                row["quarantined"] = any(q.startswith("supplier_note_untrusted") for q in out.get("quarantined", []))
        rows.append(row)
        print(json.dumps(row))

    def mean(key, kinds=None):
        vals = [r[key] for r in rows if r.get(key) is not None and (kinds is None or r["kind"] in kinds)]
        return round(sum(vals) / len(vals), 3) if vals else None

    gen = {"answer", "summarize", "rfi", "indirect_injection", "visual"}
    summary = {
        "provider": settings.provider,
        "model": settings.model if settings.provider == "claude" else "mock",
        "embedder": settings.embedder,
        "cases": len(rows),
        "retrieval_hit": mean("hit", gen),
        "keyword_recall": mean("kw_recall", gen),
        "grounded_rate": mean("grounded", gen),
        "citation_validity": mean("cite_valid", gen),
        "stale_free": mean("stale_free", gen),
        "sources": settings.sources,
        "injection_blocked": mean("blocked", {"injection"}),
        "indirect_quarantined": mean("quarantined", {"indirect_injection"}),
        "routed_to_pages": mean("routed", {"visual"}),
        "page_hit": mean("page_hit", {"visual"}),
        "visual_model": agents.page_store.embedder.name if agents.page_store else None,
        "avg_latency_s": round(sum(lat) / len(lat), 2),
        "llm_calls": agents.llm.usage.calls,
        "input_tokens_total": agents.llm.usage.total_input_tokens,
        "input_tokens_uncached": agents.llm.usage.input_tokens,
        "cache_write_tokens": agents.llm.usage.cache_write_tokens,
        "cache_read_tokens": agents.llm.usage.cache_read_tokens,
        "output_tokens": agents.llm.usage.output_tokens,
        # Claude Opus 5.5 list prices per 1M tokens: input $4, cache write 1.25x, cache read 0.1x, output $20.
        "est_cost_usd": round((agents.llm.usage.input_tokens * 4 + agents.llm.usage.cache_write_tokens * 5
                               + agents.llm.usage.cache_read_tokens * 0.4 + agents.llm.usage.output_tokens * 20) / 1e6, 3),
    }
    print("\nSUMMARY " + json.dumps(summary, indent=2))
    Path(__file__).with_name("last_results_visual.json" if args.visual else
                            f"last_results_okf_{settings.sources}.json" if args.okf else "last_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
