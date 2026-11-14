"""LLM providers.

ClaudeLLM  - real calls via the official Anthropic SDK, with schema-constrained structured outputs.
MockLLM    - deterministic, offline stand-in so the graph, tests and evals run with no API key.
             It is extractive (copies sentences from retrieved chunks), so retrieval and guardrail
             metrics stay meaningful; generation quality is only measured with ClaudeLLM.
"""

import re
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel

from sitebrain.config import Settings
from sitebrain.schemas import Answer, Critique, Plan, RFIDraft

T = TypeVar("T", bound=BaseModel)


class RefusalError(RuntimeError):
    """Raised when the model (and its server-side fallback) declines the request."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


class LLM(Protocol):
    usage: Usage

    def structured(self, system: str, user: str, schema: type[T], context: dict | None = None) -> T: ...


class ClaudeLLM:
    def __init__(self, settings: Settings):
        import anthropic

        self._client = anthropic.Anthropic()  # credentials resolved from the environment
        self.model = settings.model
        self.effort = settings.effort
        self.usage = Usage()

    def structured(self, system: str, user: str, schema: type[T], context: dict | None = None) -> T:
        response = self._client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
            output_config={"effort": self.effort},
            # On a safety decline, re-run on Anthropic's recommended fallback model server-side.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        self.usage.calls += 1
        self.usage.input_tokens += response.usage.input_tokens
        self.usage.output_tokens += response.usage.output_tokens
        if response.stop_reason == "refusal":
            category = response.stop_details.category if response.stop_details else None
            raise RefusalError(f"Model declined the request (category={category})")
        if response.parsed_output is None:
            raise RuntimeError(f"No structured output (stop_reason={response.stop_reason})")
        return response.parsed_output


_STOP = set(
    "a an the of to in on for at by with and or is are was were be been what which who whom when "
    "where why how does do did our we you your i me my please tell about from as that this these "
    "those it its there their any all can could should would will shall per vs".split()
)


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


def _best_sentences(question: str, docs: list[dict], n: int = 3) -> list[tuple[str, str]]:
    kw = _keywords(question)
    scored = []
    for d in docs:
        body = "\n".join(ln for ln in d["text"].splitlines() if not ln.lstrip().startswith("#"))
        for sent in re.split(r"(?<=[.!?])\s+|\n+", body):
            sent = sent.strip(" -*")
            overlap = len(kw & _keywords(sent))
            if overlap and len(sent) > 15:
                scored.append((overlap, sent, d["chunk_id"]))
    scored.sort(key=lambda x: -x[0])
    return [(s, cid) for _, s, cid in scored[:n]]


class MockLLM:
    def __init__(self, settings: Settings | None = None):
        self.usage = Usage()

    def structured(self, system: str, user: str, schema: type[T], context: dict | None = None) -> T:
        ctx = context or {}
        self.usage.calls += 1
        question = ctx.get("question", "")
        docs = ctx.get("docs", [])
        if schema is Plan:
            q = question.lower()
            if re.search(r"\brfi\b|request for information|draft (an? )?(rfi|query|letter|email)", q):
                route = "draft_rfi"
            elif re.search(r"summar|overview|digest", q):
                route = "summarize"
            else:
                route = "answer"
            return Plan(route=route, search_query=question, reasoning="keyword routing (mock)")
        if schema is Answer:
            picked = _best_sentences(question, docs)
            if not picked:
                return Answer(answer="I could not find this in the project documents.", citations=[])
            return Answer(
                answer=" ".join(f"{s} [{cid}]" for s, cid in picked),
                citations=sorted({cid for _, cid in picked}),
            )
        if schema is Critique:
            draft_cites = set(ctx.get("citations", []))
            retrieved = {d["chunk_id"] for d in docs}
            if not draft_cites:
                return Critique(grounded=False, issues=["No sources cited"], retry_query=f"{question} specification")
            bad = sorted(draft_cites - retrieved)
            if bad:
                return Critique(grounded=False, issues=[f"Cites unknown sources: {bad}"], retry_query=question)
            return Critique(grounded=True, issues=[], retry_query="")
        if schema is RFIDraft:
            picked = _best_sentences(question, docs)
            facts = "\n".join(f"- {s} [{cid}]" for s, cid in picked) or "- (no supporting records found)"
            return RFIDraft(
                subject=f"RFI: {question[:80]}",
                to="Project Consultant",
                body=f"Dear Consultant,\n\nPlease clarify the following.\n\nRelevant records:\n{facts}\n\n"
                "Kindly advise at your earliest convenience.\n\nRegards,\nSite Automation Team",
                citations=sorted({cid for _, cid in picked}),
            )
        raise ValueError(f"MockLLM has no rule for {schema.__name__}")


def make_llm(settings: Settings) -> LLM:
    return MockLLM(settings) if settings.provider == "mock" else ClaudeLLM(settings)
