import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.rag.embeddings import EmbeddingProvider
from src.rag.schemas import DocumentChunk


def _deterministic_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


class QdrantVectorStore:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        collection_name: str = "financial_filings",
        url: str | None = "http://localhost:6333",
        in_memory: bool = False,
    ) -> None:
        self.embedder = embedder
        self.collection_name = collection_name
        self.client = QdrantClient(":memory:") if in_memory else QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedder.dimension,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed_documents([c.content for c in chunks])
        points = [
            qmodels.PointStruct(
                id=_deterministic_uuid(chunk.chunk_id),
                vector=vec,
                payload=chunk.model_dump(),
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def dense_search(
        self,
        query: str,
        ticker: str,
        fiscal_year: int | str | None = None,
        limit: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        query_vec = self.embedder.embed_query(query)
        must_conditions: list[qmodels.Condition] = [
            qmodels.FieldCondition(
                key="ticker",
                match=qmodels.MatchValue(value=ticker.strip().upper()),
            )
        ]
        if fiscal_year is not None:
            must_conditions.append(
                qmodels.FieldCondition(
                    key="fiscal_year",
                    match=qmodels.MatchValue(value=fiscal_year),
                )
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            query_filter=qmodels.Filter(must=must_conditions),
            limit=limit,
            with_payload=True,
        )

        results: list[tuple[DocumentChunk, float]] = []
        for hit in response.points:
            if hit.payload:
                chunk = DocumentChunk.model_validate(hit.payload)
                results.append((chunk, float(hit.score)))
        return results
