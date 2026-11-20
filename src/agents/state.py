import operator
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


class CriticVerdict(BaseModel):
    is_faithful: bool = Field(
        description="True only if all numbers and facts in the draft match the verified sources."
    )
    feedback: str = Field(
        default="",
        description="Specific discrepancy notes if is_faithful is False; empty otherwise.",
    )


class AgentState(TypedDict, total=False):
    ticker: str
    fiscal_year: int | str | None
    query: str
    sub_queries: list[str]

    # Optional pre-loaded data for deterministic testing or custom uploads
    financial_input: dict[str, Any] | None
    news_headlines: list[str] | None

    # Computed agent outputs
    financial_metrics: dict[str, Any]
    extracted_contexts: list[dict[str, Any]]
    sentiment_summary: dict[str, Any]

    # Synthesis & Guardrail loop
    draft_report: str
    review_feedback: str
    is_faithful: bool
    retry_count: int
    max_retries: int

    # Execution trace log
    agent_trace: Annotated[list[str], operator.add]
