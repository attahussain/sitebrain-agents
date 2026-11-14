"""Generate a synthetic corpus of near-identical daily site reports for the retrieval benchmark.

Real projects accumulate hundreds of daily reports that share the same template and vocabulary and
differ mainly in dates, identifiers and counts. That is where pure vector search tends to blur
documents together. This script reproduces that shape: N reports from one template, each holding one
unique fact (an inspection reference, a crane fault, a rework batch), plus one question per report
whose answer exists only in that report.

The corpus is synthetic and deterministic (fixed seed). It is generated, not committed.

Usage:  python evals/make_scale_corpus.py [--n 60]
"""

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DOCS = ROOT / "data" / "scale_docs"
OUT_QUERIES = Path(__file__).with_name("scale_queries.jsonl")

CRANES = ["TC-1", "TC-2", "TC-3"]
FAULTS = ["slewing motor fault", "hoist brake wear", "anemometer failure", "hydraulic hose leak",
          "limit switch fault", "trolley cable damage", "load moment indicator error"]
ACTIVITIES = ["column rebar fixing", "slab formwork", "core wall pour", "blockwork", "MEP first fix"]


def build(n: int, seed: int = 7) -> None:
    rng = random.Random(seed)
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    for old in OUT_DOCS.glob("*.md"):
        old.unlink()

    combos = [(c, f) for c in CRANES for f in FAULTS]
    rng.shuffle(combos)
    queries = []
    start = date(2026, 8, 1)
    for i in range(n):
        day = start + timedelta(days=i)
        doc_id = f"daily_report_{day.isoformat()}"
        level = 3 + i % 30
        manpower = rng.randint(150, 230)
        # Each crane+fault pair occurs exactly once, so "on which day..." has a single correct answer.
        if i < len(combos):
            crane, fault = combos[i]
            plant = f"Tower crane {crane} was stopped for 3 hours due to a {fault}. Other cranes operated normally."
        else:
            plant = "All tower cranes operated normally."
        ref = f"INS-{1000 + i * 7}"
        first_col, last_col = rng.randint(1, 9), rng.randint(10, 24)
        text = (
            f"# Daily Site Report - {day:%d %b %Y} - Al Noor Tower (synthetic benchmark)\n\n"
            f"## Progress\nWork continued on Level {level}: {rng.choice(ACTIVITIES)} and {rng.choice(ACTIVITIES)}. "
            f"Consultant inspection reference {ref} recorded rework on columns C{level}-{first_col:02d} "
            f"to C{level}-{last_col:02d}.\n\n"
            f"## Plant\n{plant}\n\n"
            f"## Manpower\nTotal manpower on site: {manpower}.\n\n"
            f"## Safety\nNo incidents. Daily toolbox talk held on working at height.\n"
        )
        (OUT_DOCS / f"{doc_id}.md").write_text(text, encoding="utf-8")
        queries.append({"id": f"s{i:02d}a", "question": f"What did consultant inspection {ref} find?",
                        "expected_docs": [doc_id]})
        if i < len(combos):
            queries.append({"id": f"s{i:02d}b", "question": f"On which day was {crane} stopped because of a {fault}?",
                            "expected_docs": [doc_id]})
    OUT_QUERIES.write_text("\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8")
    print(f"wrote {n} reports to {OUT_DOCS} and {len(queries)} queries to {OUT_QUERIES}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    build(parser.parse_args().n)
