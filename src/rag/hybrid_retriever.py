import re

from rank_bm25 import BM25Okapi

from src.rag.schemas import DocumentChunk, RetrievedChunk
from src.rag.vector_store import QdrantVectorStore


def tokenize_financial_text(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_.-]+", text.lower())


def reciprocal_rank_fusion(
    dense_results: list[tuple[DocumentChunk, float]],
    sparse_results: list[tuple[DocumentChunk, float]],
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    merged: dict[str, RetrievedChunk] = {}

    for rank, (chunk, score) in enumerate(dense_results, start=1):
        entry = merged.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk))
        entry.dense_score = score
        entry.rrf_score += 1.0 / (rrf_k + rank)

    for rank, (chunk, score) in enumerate(sparse_results, start=1):
        entry = merged.setdefault(chunk.chunk_id, RetrievedChunk(chunk=chunk))
        entry.sparse_score = score
        entry.rrf_score += 1.0 / (rrf_k + rank)

    return sorted(merged.values(), key=lambda x: x.rrf_score, reverse=True)


class HybridRetriever:
    def __init__(self, vector_store: QdrantVectorStore, rrf_k: int = 60) -> None:
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self._corpus_chunks: dict[str, DocumentChunk] = {}

    def index_chunks(self, chunks: list[DocumentChunk]) -> int:
        for chunk in chunks:
            self._corpus_chunks[chunk.chunk_id] = chunk
        return self.vector_store.upsert_chunks(chunks)

    def _sparse_search(
        self,
        query: str,
        ticker: str,
        fiscal_year: int | str | None = None,
        limit: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        clean_ticker = ticker.strip().upper()
        candidates = [
            c
            for c in self._corpus_chunks.values()
            if c.ticker == clean_ticker and (fiscal_year is None or c.fiscal_year == fiscal_year)
        ]
        if not candidates:
            return []

        tokenized_corpus = [tokenize_financial_text(c.content) for c in candidates]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = tokenize_financial_text(query)
        scores = bm25.get_scores(query_tokens)

        scored = [
            (chunk, float(score))
            for chunk, score in zip(candidates, scores, strict=True)
            if score > 0.0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def retrieve(
        self,
        query: str,
        ticker: str,
        fiscal_year: int | str | None = None,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        dense_hits = self.vector_store.dense_search(
            query=query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            limit=top_k * 2,
        )
        sparse_hits = self._sparse_search(
            query=query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            limit=top_k * 2,
        )
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, rrf_k=self.rrf_k)
        return fused[:top_k]
