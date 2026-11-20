import json
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from src.agents.graph import build_due_diligence_graph
from src.agents.state import AgentState
from src.quant.schemas import FinancialStatementInput
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.schemas import DocumentChunk
from src.rag.vector_store import QdrantVectorStore
from tests.unit.test_hybrid_rag import DeterministicTestEmbedder


class ScriptedChatModel(BaseChatModel):
    """Deterministic scripted LLM to test multi-agent synthesis and critic loops."""

    responses: list[str]
    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-test-llm"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        idx = min(self.call_index, len(self.responses) - 1)
        text = self.responses[idx]
        self.call_index += 1
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def _build_test_retriever() -> HybridRetriever:
    embedder = DeterministicTestEmbedder()
    store = QdrantVectorStore(embedder=embedder, collection_name="agent_test", in_memory=True)
    retriever = HybridRetriever(vector_store=store)
    retriever.index_chunks(
        [
            DocumentChunk(
                chunk_id="FPT_2025_p12_t0",
                ticker="FPT.VN",
                fiscal_year=2025,
                page_number=12,
                chunk_type="table",
                content="| Facility | Amount |\n| --- | --- |\n| Short-term Debt | 500 VND Bn |",
            )
        ]
    )
    return retriever


def _sample_financial_input() -> dict[str, Any]:
    return FinancialStatementInput(
        ticker="FPT.VN",
        fiscal_year=2025,
        currency="VND",
        revenue=60_000.0,
        net_income=9_000.0,
        current_assets=30_000.0,
        current_liabilities=20_000.0,
        total_debt=10_000.0,
        total_equity=30_000.0,
    ).model_dump()


def test_graph_completes_happy_path_with_faithful_draft() -> None:
    scripted_llm = ScriptedChatModel(
        responses=[
            "## Due Diligence Memo: FPT.VN\n"
            "Current ratio is 1.5 [Chunk: FPT_2025_p12_t0, Page 12].",
            json.dumps({"is_faithful": True, "feedback": ""}),
        ]
    )
    graph = build_due_diligence_graph(llm=scripted_llm, retriever=_build_test_retriever())

    initial_state: AgentState = {
        "ticker": "fpt.vn",
        "fiscal_year": 2025,
        "query": "Assess liquidity and short-term debt",
        "financial_input": _sample_financial_input(),
        "news_headlines": ["FPT reports record software export growth in Japan"],
    }

    result = graph.invoke(initial_state)

    assert result["is_faithful"] is True
    assert result["retry_count"] == 0
    assert result["financial_metrics"]["current_ratio"] == 1.5
    assert result["sentiment_summary"]["sentiment_label"] == "positive"
    assert len(result["extracted_contexts"]) == 1
    assert any("CriticGuardrail: PASSED" in step for step in result["agent_trace"])


def test_graph_self_corrects_when_critic_detects_hallucination() -> None:
    scripted_llm = ScriptedChatModel(
        responses=[
            "Draft 1: Current ratio is 9.9 (hallucinated).",
            json.dumps(
                {
                    "is_faithful": False,
                    "feedback": "Current ratio is 1.5, not 9.9.",
                }
            ),
            "Draft 2: Corrected current ratio is 1.5 [Chunk: FPT_2025_p12_t0, Page 12].",
            json.dumps({"is_faithful": True, "feedback": ""}),
        ]
    )
    graph = build_due_diligence_graph(llm=scripted_llm, retriever=_build_test_retriever())

    result = graph.invoke(
        {
            "ticker": "FPT.VN",
            "financial_input": _sample_financial_input(),
            "max_retries": 2,
        }
    )

    assert result["is_faithful"] is True
    assert result["retry_count"] == 1
    assert "Corrected current ratio is 1.5" in result["draft_report"]
    assert scripted_llm.call_index == 4


def test_graph_stops_at_max_retries_circuit_breaker() -> None:
    scripted_llm = ScriptedChatModel(
        responses=[
            "Bad draft 1",
            json.dumps({"is_faithful": False, "feedback": "Unverified numbers"}),
            "Bad draft 2",
            json.dumps({"is_faithful": False, "feedback": "Still unverified"}),
        ]
    )
    graph = build_due_diligence_graph(llm=scripted_llm, retriever=_build_test_retriever())

    result = graph.invoke(
        {
            "ticker": "FPT.VN",
            "financial_input": _sample_financial_input(),
            "max_retries": 2,
        }
    )

    assert result["is_faithful"] is False
    assert result["retry_count"] == 2
    assert scripted_llm.call_index == 4
