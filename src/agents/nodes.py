import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.prompts import CRITIC_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT
from src.agents.state import AgentState, CriticVerdict
from src.quant.data_fetcher import YFinanceDataFetcher
from src.quant.ratio_calculator import calculate_financial_metrics
from src.quant.schemas import FinancialStatementInput
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import FinancialReranker


def planner_node(state: AgentState) -> dict[str, Any]:
    ticker = state["ticker"].strip().upper()
    query = state.get("query") or f"Comprehensive financial due diligence for {ticker}"
    sub_queries = state.get("sub_queries") or [
        query,
        "debt maturity schedule bonds liquidity covenants",
        "contingent liabilities off-balance sheet risks",
    ]
    return {
        "ticker": ticker,
        "query": query,
        "sub_queries": sub_queries,
        "retry_count": state.get("retry_count", 0),
        "max_retries": state.get("max_retries", 2),
        "agent_trace": [f"Planner: decomposed analysis plan for {ticker}"],
    }


def quant_node(
    state: AgentState,
    fetcher: YFinanceDataFetcher | None = None,
) -> dict[str, Any]:
    ticker = state["ticker"]
    raw_input = state.get("financial_input")

    if raw_input is not None:
        stmt = FinancialStatementInput.model_validate(raw_input)
    else:
        active_fetcher = fetcher or YFinanceDataFetcher()
        stmt = active_fetcher.fetch_financial_input(ticker)

    metrics = calculate_financial_metrics(stmt)
    return {
        "financial_metrics": metrics.model_dump(),
        "agent_trace": [
            f"QuantEngine: computed ratios for {ticker} (red_flags={len(metrics.red_flags)})"
        ],
    }


def rag_node(
    state: AgentState,
    retriever: HybridRetriever | None = None,
    reranker: FinancialReranker | None = None,
) -> dict[str, Any]:
    if retriever is None:
        return {
            "extracted_contexts": state.get("extracted_contexts", []),
            "agent_trace": ["HybridRAG: skipped (no retriever configured)"],
        }

    ticker = state["ticker"]
    fiscal_year = state.get("fiscal_year")
    sub_queries = state.get("sub_queries") or [state.get("query", "financial risks")]
    active_reranker = reranker or FinancialReranker()

    seen_ids: set[str] = set()
    collected = []
    for sq in sub_queries:
        hits = retriever.retrieve(
            query=sq,
            ticker=ticker,
            fiscal_year=fiscal_year,
            top_k=6,
        )
        reranked = active_reranker.rerank(query=sq, candidates=hits, top_n=3)
        for item in reranked:
            if item.chunk.chunk_id not in seen_ids:
                seen_ids.add(item.chunk.chunk_id)
                collected.append(
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "page_number": item.chunk.page_number,
                        "chunk_type": item.chunk.chunk_type,
                        "rerank_score": item.rerank_score,
                        "content": item.chunk.content,
                    }
                )

    return {
        "extracted_contexts": collected[:6],
        "agent_trace": [f"HybridRAG: retrieved {len(collected[:6])} reranked filing chunks"],
    }


def sentiment_node(state: AgentState) -> dict[str, Any]:
    headlines = state.get("news_headlines") or []
    positive_words = {"growth", "beat", "record", "surge", "upgrade", "profit", "expansion"}
    negative_words = {"lawsuit", "default", "downgrade", "loss", "decline", "debt", "investigation"}

    pos_hits = 0
    neg_hits = 0
    for title in headlines:
        tokens = set(re.findall(r"[a-z]+", title.lower()))
        pos_hits += len(tokens & positive_words)
        neg_hits += len(tokens & negative_words)

    total = pos_hits + neg_hits
    score = round((pos_hits - neg_hits) / total, 3) if total > 0 else 0.0
    label = "positive" if score > 0.15 else ("negative" if score < -0.15 else "neutral")

    return {
        "sentiment_summary": {
            "headline_count": len(headlines),
            "sentiment_score": score,
            "sentiment_label": label,
            "headlines": headlines[:5],
        },
        "agent_trace": [f"NewsSentiment: scored {len(headlines)} headlines ({label}, {score})"],
    }


def synthesizer_node(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    retry_count = state.get("retry_count", 0)
    feedback = state.get("review_feedback", "")

    user_payload = (
        f"TICKER: {state['ticker']}\n"
        f"USER QUERY: {state.get('query', '')}\n\n"
        f"VERIFIED QUANTITATIVE METRICS:\n"
        f"{json.dumps(state.get('financial_metrics', {}), indent=2)}\n\n"
        f"RETRIEVED FILING CONTEXTS:\n"
        f"{json.dumps(state.get('extracted_contexts', []), indent=2)}\n\n"
        f"MARKET SENTIMENT SUMMARY:\n"
        f"{json.dumps(state.get('sentiment_summary', {}), indent=2)}\n"
    )

    if retry_count > 0 and feedback:
        user_payload += (
            f"\nPREVIOUS DRAFT WAS REJECTED BY COMPLIANCE AUDITOR (Attempt {retry_count}).\n"
            f"FIX THESE EXACT ISSUES:\n{feedback}\n"
        )

    response = llm.invoke(
        [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=user_payload),
        ]
    )
    draft = str(response.content).strip()
    return {
        "draft_report": draft,
        "agent_trace": [f"Synthesizer: generated report draft (attempt {retry_count + 1})"],
    }


def _parse_critic_response(raw_text: str) -> CriticVerdict:
    cleaned = raw_text.strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        data = json.loads(cleaned)
        return CriticVerdict.model_validate(data)
    except Exception:
        is_ok = '"is_faithful": true' in raw_text.lower()
        return CriticVerdict(is_faithful=is_ok, feedback="" if is_ok else raw_text)


def critic_node(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    audit_payload = (
        f"VERIFIED QUANTITATIVE METRICS:\n"
        f"{json.dumps(state.get('financial_metrics', {}), indent=2)}\n\n"
        f"RETRIEVED FILING CONTEXTS:\n"
        f"{json.dumps(state.get('extracted_contexts', []), indent=2)}\n\n"
        f"DRAFT REPORT TO AUDIT:\n"
        f"{state.get('draft_report', '')}\n"
    )

    response = llm.invoke(
        [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=audit_payload),
        ]
    )
    verdict = _parse_critic_response(str(response.content))
    current_retries = state.get("retry_count", 0)
    next_retries = current_retries if verdict.is_faithful else current_retries + 1

    status_tag = "PASSED" if verdict.is_faithful else f"FAILED (retry={next_retries})"
    return {
        "is_faithful": verdict.is_faithful,
        "review_feedback": verdict.feedback,
        "retry_count": next_retries,
        "agent_trace": [f"CriticGuardrail: {status_tag}"],
    }
