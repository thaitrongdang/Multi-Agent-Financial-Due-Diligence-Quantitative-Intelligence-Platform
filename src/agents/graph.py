from functools import partial

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from src.agents.llm_factory import get_chat_model
from src.agents.nodes import (
    critic_node,
    planner_node,
    quant_node,
    rag_node,
    sentiment_node,
    synthesizer_node,
)
from src.agents.state import AgentState
from src.quant.data_fetcher import YFinanceDataFetcher
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.reranker import FinancialReranker


def route_after_critic(state: AgentState) -> str:
    if state.get("is_faithful", False):
        return "end"
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    if retry_count >= max_retries:
        return "end"
    return "synthesize"


def build_due_diligence_graph(
    llm: BaseChatModel | None = None,
    fetcher: YFinanceDataFetcher | None = None,
    retriever: HybridRetriever | None = None,
    reranker: FinancialReranker | None = None,
):
    active_llm = llm or get_chat_model()

    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("quant", partial(quant_node, fetcher=fetcher))
    workflow.add_node("rag", partial(rag_node, retriever=retriever, reranker=reranker))
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("synthesizer", partial(synthesizer_node, llm=active_llm))
    workflow.add_node("critic", partial(critic_node, llm=active_llm))

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "quant")
    workflow.add_edge("quant", "rag")
    workflow.add_edge("rag", "sentiment")
    workflow.add_edge("sentiment", "synthesizer")
    workflow.add_edge("synthesizer", "critic")

    workflow.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "synthesize": "synthesizer",
            "end": END,
        },
    )

    return workflow.compile()
