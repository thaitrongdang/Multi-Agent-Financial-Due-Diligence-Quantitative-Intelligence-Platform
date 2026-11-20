import re

import httpx

from src.rag.hybrid_retriever import tokenize_financial_text
from src.rag.schemas import RetrievedChunk


class FinancialReranker:
    """Reranks retrieved chunks via neural rerank API or financial lexical-numeric scoring."""

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str = "jina-reranker-v2-base-multilingual",
    ) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.model = model

    def _neural_api_rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        response = httpx.post(
            str(self.api_url),
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "documents": [c.chunk.content for c in candidates],
                "top_n": top_n,
            },
            timeout=20.0,
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        reranked: list[RetrievedChunk] = []
        for item in results:
            idx = int(item["index"])
            candidate = candidates[idx].model_copy()
            candidate.rerank_score = float(item["relevance_score"])
            reranked.append(candidate)
        return reranked

    def _numeric_lexical_rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        query_tokens = set(tokenize_financial_text(query))
        query_numbers = set(re.findall(r"\d+(?:\.\d+)?", query))

        scored: list[RetrievedChunk] = []
        for item in candidates:
            doc_tokens = set(tokenize_financial_text(item.chunk.content))
            overlap = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
            number_bonus = 0.25 if (query_numbers and query_numbers & doc_tokens) else 0.0
            table_bonus = 0.1 if (item.chunk.chunk_type == "table" and query_numbers) else 0.0

            updated = item.model_copy()
            updated.rerank_score = round(
                (0.5 * overlap)
                + (0.5 * min(item.rrf_score * 30.0, 1.0))
                + number_bonus
                + table_bonus,
                4,
            )
            scored.append(updated)

        scored.sort(key=lambda x: x.rerank_score, reverse=True)
        return scored[:top_n]

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_n: int = 5,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        if self.api_url and self.api_key:
            return self._neural_api_rerank(query, candidates, top_n)
        return self._numeric_lexical_rerank(query, candidates, top_n)
