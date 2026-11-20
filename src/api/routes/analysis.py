import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.graph import build_due_diligence_graph
from src.agents.state import AgentState
from src.api.dependencies import (
    get_data_fetcher,
    get_hybrid_retriever,
    get_llm_client,
    get_reranker,
)
from src.api.schemas import AnalyzeRequest, AnalyzeResponse
from src.core.tracing import get_langfuse_callback
from src.quant.data_fetcher import FinancialDataNotFoundError, YFinanceDataFetcher
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import FinancialReranker

router = APIRouter(tags=["Due Diligence Analysis"])


def _build_initial_state(payload: AnalyzeRequest) -> AgentState:
    return {
        "ticker": payload.ticker.strip().upper(),
        "fiscal_year": payload.fiscal_year,
        "query": payload.query,
        "financial_input": payload.financial_input,
        "news_headlines": payload.news_headlines,
        "max_retries": payload.max_retries,
    }


def _build_run_config(ticker: str) -> dict[str, Any]:
    callback = get_langfuse_callback(
        trace_name=f"due-diligence-{ticker}",
        metadata={"ticker": ticker},
    )
    if callback is None:
        return {}
    return {"callbacks": [callback]}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_company(
    payload: AnalyzeRequest,
    llm: BaseChatModel = Depends(get_llm_client),
    fetcher: YFinanceDataFetcher = Depends(get_data_fetcher),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: FinancialReranker = Depends(get_reranker),
) -> AnalyzeResponse:
    graph = build_due_diligence_graph(
        llm=llm,
        fetcher=fetcher,
        retriever=retriever,
        reranker=reranker,
    )
    initial_state = _build_initial_state(payload)
    run_config = _build_run_config(initial_state["ticker"])

    try:
        result = graph.invoke(initial_state, config=run_config)
    except FinancialDataNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return AnalyzeResponse(
        ticker=result["ticker"],
        fiscal_year=result.get("fiscal_year"),
        is_faithful=result.get("is_faithful", False),
        retry_count=result.get("retry_count", 0),
        financial_metrics=result.get("financial_metrics", {}),
        sentiment_summary=result.get("sentiment_summary", {}),
        extracted_contexts=result.get("extracted_contexts", []),
        draft_report=result.get("draft_report", ""),
        agent_trace=result.get("agent_trace", []),
    )


@router.post("/analyze/stream")
async def stream_analyze_company(
    payload: AnalyzeRequest,
    llm: BaseChatModel = Depends(get_llm_client),
    fetcher: YFinanceDataFetcher = Depends(get_data_fetcher),
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
    reranker: FinancialReranker = Depends(get_reranker),
) -> StreamingResponse:
    graph = build_due_diligence_graph(
        llm=llm,
        fetcher=fetcher,
        retriever=retriever,
        reranker=reranker,
    )
    initial_state = _build_initial_state(payload)
    run_config = _build_run_config(initial_state["ticker"])

    async def event_generator() -> AsyncGenerator[str, None]:
        final_state: dict[str, Any] = dict(initial_state)
        try:
            for update in graph.stream(initial_state, config=run_config, stream_mode="updates"):
                for node_name, node_output in update.items():
                    if isinstance(node_output, dict):
                        final_state.update(node_output)
                    event_payload = {
                        "event": "node_update",
                        "node": node_name,
                        "trace": node_output.get("agent_trace", [])
                        if isinstance(node_output, dict)
                        else [],
                    }
                    yield f"data: {json.dumps(event_payload)}\n\n"

            done_payload = {
                "event": "completed",
                "ticker": final_state.get("ticker"),
                "is_faithful": final_state.get("is_faithful", False),
                "retry_count": final_state.get("retry_count", 0),
                "financial_metrics": final_state.get("financial_metrics", {}),
                "sentiment_summary": final_state.get("sentiment_summary", {}),
                "extracted_contexts": final_state.get("extracted_contexts", []),
                "draft_report": final_state.get("draft_report", ""),
            }
            yield f"data: {json.dumps(done_payload)}\n\n"
        except FinancialDataNotFoundError as exc:
            err_payload = {"event": "error", "detail": str(exc)}
            yield f"data: {json.dumps(err_payload)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
