"""Structured outputs each agent must return. Claude is constrained to these schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """Planner (supervisor) decision."""

    route: Literal["answer", "summarize", "draft_rfi"] = Field(
        description="answer = factual question; summarize = digest of reports/records; "
        "draft_rfi = the user wants a Request For Information drafted for the consultant"
    )
    search_query: str = Field(description="A focused search query for the document store")
    search_pages: bool = Field(
        description="True if the answer is likely shown visually: a drawing, drawing number or revision, "
        "bar bending schedule, checklist or form, site plan, or progress chart"
    )
    reasoning: str = Field(description="One sentence explaining the route")


class Answer(BaseModel):
    """Writer output. Every factual sentence must be supported by a cited chunk."""

    answer: str
    citations: list[str] = Field(description="chunk_ids of the sources actually used")


class Critique(BaseModel):
    """Critic verdict on whether the draft is fully supported by the retrieved sources."""

    grounded: bool = Field(description="Every factual claim is supported by a cited source")
    answered: bool = Field(description="The draft actually answers the question, rather than saying the sources lack it")
    issues: list[str] = Field(description="Unsupported claims or missing information; empty if none")
    retry_query: str = Field(description="Better search query if more evidence is needed, else empty")


class RFIDraft(BaseModel):
    """Request For Information to the project consultant; requires human approval before release."""

    subject: str
    to: str
    body: str
    citations: list[str]
