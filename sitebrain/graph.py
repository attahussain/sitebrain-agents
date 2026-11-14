"""LangGraph wiring.

guard_input -> planner -> researcher -> writer -> critic --(not grounded)--> researcher (bounded retries)
                                                     |--(answer/summary)--> finalize -> guard_output
                                                     |--(RFI)--> human_review (interrupt) -> finalize -> guard_output
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from sitebrain.agents import Agents, State
from sitebrain.config import Settings, get_settings
from sitebrain.llm import LLM, make_llm
from sitebrain.store import DocStore


def build_graph(llm: LLM | None = None, store: DocStore | None = None, settings: Settings | None = None):
    settings = settings or get_settings()
    agents = Agents(llm or make_llm(settings), store or DocStore(settings), settings.max_research_attempts)

    g = StateGraph(State)
    g.add_node("guard_input", agents.guard_input)
    g.add_node("planner", agents.planner)
    g.add_node("researcher", agents.researcher)
    g.add_node("writer", agents.writer)
    g.add_node("critic", agents.critic)
    g.add_node("human_review", agents.human_review)
    g.add_node("finalize", agents.finalize)
    g.add_node("guard_output", agents.guard_output)

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
    return g.compile(checkpointer=InMemorySaver()), agents
