"""HTTP API.

POST /ask                       run a question through the agent graph
POST /rfi/{thread_id}/decision  approve or reject a paused RFI draft
GET  /health                    liveness and configuration
GET  /metrics                   cumulative LLM usage and cache hit counts

Run:  uvicorn sitebrain.api:app --reload
"""

import sqlite3
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from sitebrain.cache import cache_key, make_cache
from sitebrain.config import get_settings
from sitebrain.graph import build_graph
from sitebrain.llm import RefusalError
from sitebrain.store import TABLE, DocStore


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Decision(BaseModel):
    approved: bool
    note: str = ""


class AskResponse(BaseModel):
    status: str  # "complete" | "awaiting_approval" | "blocked"
    thread_id: str
    answer: str | None = None
    draft: str | None = None
    citations: list[str] = []
    trace: list[str] = []
    timings: list[dict] = []
    cached: bool = False


def _ensure_index(store: DocStore) -> None:
    if TABLE not in store.db.list_tables().tables:
        store.ingest()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = DocStore(settings)
    _ensure_index(store)
    settings.checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.checkpoint_db), check_same_thread=False)
    graph, agents = build_graph(store=store, settings=settings, checkpointer=SqliteSaver(conn))
    app.state.settings = settings
    app.state.graph = graph
    app.state.agents = agents
    app.state.cache = make_cache(settings)
    app.state.cache_hits = 0
    yield
    conn.close()


app = FastAPI(title="SiteBrain", version="0.3.0", lifespan=lifespan)


def _response(result: dict, thread_id: str) -> AskResponse:
    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return AskResponse(status="awaiting_approval", thread_id=thread_id, draft=payload["draft"],
                           citations=payload.get("citations", []), trace=result.get("trace", []),
                           timings=result.get("timings", []))
    status = "blocked" if result.get("blocked") else "complete"
    return AskResponse(status=status, thread_id=thread_id, answer=result.get("final", ""),
                       citations=result.get("citations", []), trace=result.get("trace", []),
                       timings=result.get("timings", []))


@app.get("/health")
def health():
    s = app.state.settings
    return {"status": "ok", "provider": s.provider, "model": s.model, "search_mode": s.search_mode,
            "cache": type(app.state.cache).__name__ if app.state.cache else "disabled"}


@app.get("/metrics")
def metrics():
    u = app.state.agents.llm.usage
    return {"llm_calls": u.calls, "input_tokens": u.total_input_tokens, "cache_write_tokens": u.cache_write_tokens,
            "cache_read_tokens": u.cache_read_tokens, "output_tokens": u.output_tokens,
            "cache_hits": app.state.cache_hits}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    settings, cache = app.state.settings, app.state.cache
    key = cache_key(req.question, settings)
    if cache and (hit := cache.get(key)):
        app.state.cache_hits += 1
        return AskResponse(**{**hit, "cached": True})

    thread_id = str(uuid.uuid4())
    try:
        result = app.state.graph.invoke({"question": req.question}, {"configurable": {"thread_id": thread_id}})
    except RefusalError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    response = _response(result, thread_id)
    # Only finished answers are cached; RFIs need a human decision and blocked requests are cheap.
    if cache and response.status == "complete":
        cache.set(key, response.model_dump(exclude={"cached"}), settings.cache_ttl)
    return response


@app.post("/rfi/{thread_id}/decision", response_model=AskResponse)
def decide(thread_id: str, decision: Decision):
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = app.state.graph.get_state(config)
    if not snapshot.next or "human_review" not in snapshot.next:
        raise HTTPException(status_code=404, detail="No RFI awaiting approval for this thread")
    result = app.state.graph.invoke(Command(resume=decision.model_dump()), config)
    return _response(result, thread_id)
