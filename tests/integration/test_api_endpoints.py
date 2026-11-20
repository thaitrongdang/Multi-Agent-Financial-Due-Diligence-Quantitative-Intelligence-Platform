import json
from typing import Any
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.api.dependencies import get_data_fetcher, get_hybrid_retriever, get_llm_client
from src.api.main import create_app
from src.quant.data_fetcher import FinancialDataNotFoundError
from src.quant.schemas import FinancialStatementInput
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.vector_store import QdrantVectorStore
from tests.unit.test_hybrid_rag import DeterministicTestEmbedder
from tests.unit.test_multi_agent_graph import ScriptedChatModel


def _create_test_client(llm: ScriptedChatModel) -> TestClient:
    app = create_app()
    embedder = DeterministicTestEmbedder()
    store = QdrantVectorStore(embedder=embedder, collection_name="api_int_test", in_memory=True)
    retriever = HybridRetriever(vector_store=store)

    app.dependency_overrides[get_hybrid_retriever] = lambda: retriever
    app.dependency_overrides[get_llm_client] = lambda: llm
    return TestClient(app)


def _sample_statement() -> dict[str, Any]:
    return FinancialStatementInput(
        ticker="FPT.VN",
        fiscal_year=2025,
        currency="VND",
        revenue=65_000.0,
        net_income=10_000.0,
        current_assets=32_000.0,
        current_liabilities=16_000.0,
        total_debt=12_000.0,
        total_equity=35_000.0,
    ).model_dump()


def test_ingest_and_analyze_end_to_end() -> None:
    llm = ScriptedChatModel(
        responses=[
            "## Due Diligence Memo: FPT.VN\n"
            "Current ratio is 2.0 [Chunk: FPT.VN_2025_p5_t0, Page 5].",
            json.dumps({"is_faithful": True, "feedback": ""}),
        ]
    )
    client = _create_test_client(llm)

    ingest_res = client.post(
        "/api/v1/documents/ingest-text",
        json={
            "ticker": "FPT.VN",
            "fiscal_year": 2025,
            "page_number": 5,
            "markdown_content": (
                "| Debt Facility | Balance (VND Bn) |\n"
                "| --- | --- |\n"
                "| Working Capital Loan | 12,000 |"
            ),
        },
    )
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["chunks_indexed"] == 1
    assert ingest_data["table_chunks"] == 1

    analyze_res = client.post(
        "/api/v1/analyze",
        json={
            "ticker": "FPT.VN",
            "fiscal_year": 2025,
            "query": "Evaluate working capital loan and liquidity",
            "financial_input": _sample_statement(),
            "news_headlines": ["FPT signs major AI expansion contract in Tokyo"],
        },
    )
    assert analyze_res.status_code == 200
    body = analyze_res.json()
    assert body["ticker"] == "FPT.VN"
    assert body["is_faithful"] is True
    assert body["financial_metrics"]["current_ratio"] == 2.0
    assert len(body["extracted_contexts"]) == 1
    assert "Current ratio is 2.0" in body["draft_report"]


def test_analyze_sse_stream_emits_node_updates_and_completion() -> None:
    llm = ScriptedChatModel(
        responses=[
            "Streamed Due Diligence Report for FPT.VN.",
            json.dumps({"is_faithful": True, "feedback": ""}),
        ]
    )
    client = _create_test_client(llm)

    with client.stream(
        "POST",
        "/api/v1/analyze/stream",
        json={
            "ticker": "FPT.VN",
            "financial_input": _sample_statement(),
        },
    ) as response:
        assert response.status_code == 200
        events = [
            json.loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    node_events = [e["node"] for e in events if e["event"] == "node_update"]
    assert node_events == ["planner", "quant", "rag", "sentiment", "synthesizer", "critic"]
    assert events[-1]["event"] == "completed"
    assert events[-1]["is_faithful"] is True


def test_analyze_returns_404_when_ticker_not_found() -> None:
    llm = ScriptedChatModel(responses=[])
    app = create_app()
    failing_fetcher = MagicMock()
    failing_fetcher.fetch_financial_input.side_effect = FinancialDataNotFoundError("Unknown ticker")

    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_data_fetcher] = lambda: failing_fetcher
    app.dependency_overrides[get_hybrid_retriever] = lambda: MagicMock()
    client = TestClient(app)

    res = client.post("/api/v1/analyze", json={"ticker": "UNKNOWN_XYZ"})
    assert res.status_code == 404
    assert "Unknown ticker" in res.json()["detail"]
