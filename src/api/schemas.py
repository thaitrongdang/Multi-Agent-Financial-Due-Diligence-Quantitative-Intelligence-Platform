from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, examples=["AAPL", "FPT.VN"])
    fiscal_year: int | str | None = Field(default=None, examples=[2025])
    query: str = Field(
        default="Comprehensive financial due diligence and risk assessment",
        min_length=3,
    )
    financial_input: dict[str, Any] | None = None
    news_headlines: list[str] | None = None
    max_retries: int = Field(default=2, ge=0, le=5)


class AnalyzeResponse(BaseModel):
    ticker: str
    fiscal_year: int | str | None = None
    is_faithful: bool
    retry_count: int
    financial_metrics: dict[str, Any]
    sentiment_summary: dict[str, Any]
    extracted_contexts: list[dict[str, Any]]
    draft_report: str
    agent_trace: list[str]


class IngestMarkdownRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    fiscal_year: int | str
    page_number: int = Field(default=1, ge=1)
    markdown_content: str = Field(..., min_length=1)


class IngestResponse(BaseModel):
    ticker: str
    chunks_indexed: int
    table_chunks: int
    text_chunks: int
