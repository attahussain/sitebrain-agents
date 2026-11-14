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
    reasoning: str = Field(description="One sentence explaining the route")


class Answer(BaseModel):
    """Writer output. Every factual sentence must be supported by a cited chunk."""

    answer: str
    citations: list[str] = Field(description="chunk_ids of the sources actually used")


class Critique(BaseModel):
    """Critic verdict on whether the draft is fully supported by the retrieved sources."""

    grounded: bool
    issues: list[str] = Field(description="Unsupported or wrong claims; empty if grounded")
    retry_query: str = Field(description="Better search query if more evidence is needed, else empty")


class RFIDraft(BaseModel):
    """Request For Information to the project consultant; requires human approval before release."""

    subject: str
    to: str
    body: str
    citations: list[str]
