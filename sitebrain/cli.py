"""Command line: `python -m sitebrain.cli ingest` and `python -m sitebrain.cli ask "question"`."""

import argparse
import sys
import uuid

from langgraph.types import Command

from sitebrain.config import get_settings
from sitebrain.graph import build_graph
from sitebrain.store import DocStore


def run_question(question: str, approve: str | None = None, show_trace: bool = True) -> dict:
    graph, agents = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke({"question": question}, config)

    while "__interrupt__" in result:  # human-in-the-loop: RFI waiting for approval
        payload = result["__interrupt__"][0].value
        print("\n=== RFI DRAFT AWAITING APPROVAL ===\n" + payload["draft"] + "\n")
        if approve is None:
            answer = input("Approve and release this RFI? [y/N] ").strip().lower()
            note = input("Optional note: ").strip()
        else:
            answer, note = approve, "auto-decision from --approve flag"
        result = graph.invoke(Command(resume={"approved": answer in ("y", "yes"), "note": note}), config)

    if show_trace:
        print("\n--- trace ---")
        for step in result.get("trace", []):
            print(" ", step)
        u = agents.llm.usage
        print(f"--- llm calls={u.calls} input_tokens={u.total_input_tokens} (cached reads {u.cache_read_tokens}) output_tokens={u.output_tokens}")
    print("\n=== ANSWER ===\n" + result.get("final", ""))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sitebrain")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="Chunk, embed and index data/docs into LanceDB")
    ask = sub.add_parser("ask", help="Ask the multi-agent assistant a question")
    ask.add_argument("question")
    ask.add_argument("--approve", choices=["y", "n"], help="Non-interactive decision for RFI approval")
    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        settings = get_settings()
        n = DocStore(settings).ingest()
        print(f"Indexed {n} chunks.")
        if settings.visual:
            from sitebrain.visual import PageStore

            pages = PageStore(settings)
            print(f"Indexed {pages.index()} page images with {pages.embedder.name} "
                  f"in {pages.last_index_seconds:.1f}s.")
        return 0
    run_question(args.question, approve=args.approve)
    return 0


if __name__ == "__main__":
    sys.exit(main())
