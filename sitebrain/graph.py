"""LangGraph wiring.

guard_input -> planner -> researcher -> writer -> critic --(not grounded)--> researcher (bounded retries)
                                                     |--(answer/summary)--> finalize -> guard_output
                                                     |--(RFI)--> human_review (interrupt) -> finalize -> guard_output
"""

import time
from collections.abc import Callable

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from sitebrain.agents import Agents, State
from sitebrain.config import Settings, get_settings
from sitebrain.llm import LLM, make_llm
from sitebrain.store import DocStore


def open_page_store(settings: Settings):
    """Load the visual page index, building it on first use. Imported lazily: needs the [visual] extra."""
    from sitebrain.visual import PAGE_TABLE, PageStore

    page_store = PageStore(settings)
    if PAGE_TABLE not in page_store.db.list_tables().tables:
        page_store.index()
    return page_store


def _timed(name: str, node: Callable[[State], State]) -> Callable[[State], State]:
    """Record each node's wall-clock time in state["timings"] for per-request tracing."""

    def run(state: State) -> State:
        start = time.perf_counter()
        update = node(state)
        update["timings"] = [{"node": name, "ms": round((time.perf_counter() - start) * 1000, 1)}]
        return update

    return run


def build_graph(llm: LLM | None = None, store: DocStore | None = None, settings: Settings | None = None,
                checkpointer: BaseCheckpointSaver | None = None, page_store=None):
    settings = settings or get_settings()
    if page_store is None and settings.visual:
        page_store = open_page_store(settings)
    agents = Agents(llm or make_llm(settings), store or DocStore(settings), settings.max_research_attempts,
                    page_store=page_store)

    g = StateGraph(State)
    for name, node in [
        ("guard_input", agents.guard_input),
        ("planner", agents.planner),
        ("researcher", agents.researcher),
        ("writer", agents.writer),
        ("critic", agents.critic),
        ("human_review", agents.human_review),
        ("finalize", agents.finalize),
        ("guard_output", agents.guard_output),
    ]:
        # human_review is not timed: its duration is the human's response time, not system latency.
        g.add_node(name, node if name == "human_review" else _timed(name, node))

    g.add_edge(START, "guard_input")
    g.add_conditional_edges("guard_input", agents.after_guard, {"blocked": END, "ok": "planner"})
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", agents.after_critic,
                            {"retry": "researcher", "rfi": "human_review", "done": "finalize"})
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", "guard_output")
    g.add_edge("guard_output", END)

    # The checkpointer persists state between steps, which is what lets the RFI path pause for a human.
    # The CLI uses memory; the API passes a SqliteSaver so a paused RFI survives a restart.
    return g.compile(checkpointer=checkpointer or InMemorySaver()), agents
