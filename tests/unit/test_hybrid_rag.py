import hashlib

from src.rag.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion
from src.rag.parser import FinancialPDFParser, table_rows_to_markdown
from src.rag.reranker import FinancialReranker
from src.rag.schemas import DocumentChunk
from src.rag.vector_store import QdrantVectorStore


class DeterministicTestEmbedder:
    """Fast deterministic bag-of-words hash embedder for isolated CI testing."""

    @property
    def dimension(self) -> int:
        return 32

    def _encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for word in text.lower().split():
            digest = hashlib.sha256(word.encode("utf-8")).digest()
            idx = digest[0] % self.dimension
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


def test_table_rows_to_markdown_preserves_structure() -> None:
    rows = [
        ["Metric", "FY2024", "FY2025"],
        ["Total Revenue", "120,000", "145,500"],
        ["Net Debt", "30,000", "25,000"],
    ]
    md = table_rows_to_markdown(rows)
    lines = md.splitlines()
    assert len(lines) == 4
    assert lines[0] == "| Metric | FY2024 | FY2025 |"
    assert lines[1] == "| --- | --- | --- |"
    assert "| Total Revenue | 120,000 | 145,500 |" in lines[2]


def test_parser_keeps_markdown_tables_intact() -> None:
    raw_doc = """
Management discussion regarding liquidity and bond maturity schedule.

| Liability Item | Amount (VND Bn) | Maturity |
| --- | --- | --- |
| Short-term Bond A | 1,250 | Q3 2026 |
| Syndicated Loan B | 3,400 | Q1 2027 |

Additional footnote explanation on collateral assets.
"""
    parser = FinancialPDFParser(chunk_size=60, chunk_overlap=10)
    chunks = parser.parse_markdown_sections(raw_doc, ticker="fpt.vn", fiscal_year=2025)

    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    assert len(table_chunks) == 1
    assert "Short-term Bond A" in table_chunks[0].content
    assert "Syndicated Loan B" in table_chunks[0].content
    assert table_chunks[0].ticker == "FPT.VN"


def test_reciprocal_rank_fusion_prioritizes_dual_hits() -> None:
    c1 = DocumentChunk(chunk_id="c1", ticker="AAPL", fiscal_year=2025, content="Bond maturity 1250")
    c2 = DocumentChunk(chunk_id="c2", ticker="AAPL", fiscal_year=2025, content="General overview")
    c3 = DocumentChunk(
        chunk_id="c3", ticker="AAPL", fiscal_year=2025, content="Exact 1250 covenant"
    )

    dense = [(c2, 0.91), (c1, 0.85)]
    sparse = [(c3, 12.4), (c1, 9.8)]

    fused = reciprocal_rank_fusion(dense, sparse, rrf_k=60)
    assert fused[0].chunk.chunk_id == "c1"
    assert fused[0].dense_score == 0.85
    assert fused[0].sparse_score == 9.8


def test_hybrid_retriever_and_reranker_end_to_end() -> None:
    embedder = DeterministicTestEmbedder()
    store = QdrantVectorStore(embedder=embedder, collection_name="test_filings", in_memory=True)
    retriever = HybridRetriever(vector_store=store)
    reranker = FinancialReranker()

    chunks = [
        DocumentChunk(
            chunk_id="FPT_2025_1",
            ticker="FPT.VN",
            fiscal_year=2025,
            chunk_type="text",
            content="FPT software export revenue grew strongly across Japan and US markets.",
        ),
        DocumentChunk(
            chunk_id="FPT_2025_2",
            ticker="FPT.VN",
            fiscal_year=2025,
            chunk_type="table",
            content="| Item | Value |\n| --- | --- |\n| Corporate Bond Due | 1250 VND Bn |",
        ),
        DocumentChunk(
            chunk_id="VCB_2025_1",
            ticker="VCB.VN",
            fiscal_year=2025,
            chunk_type="text",
            content="VCB capital adequacy ratio remained above Basel II requirements.",
        ),
    ]
    inserted = retriever.index_chunks(chunks)
    assert inserted == 3

    hits = retriever.retrieve(query="corporate bond 1250 maturity", ticker="FPT.VN", top_k=5)
    assert all(h.chunk.ticker == "FPT.VN" for h in hits)

    top_reranked = reranker.rerank(query="corporate bond 1250", candidates=hits, top_n=1)
    assert len(top_reranked) == 1
    assert top_reranked[0].chunk.chunk_id == "FPT_2025_2"
