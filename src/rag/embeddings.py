from typing import Protocol

import httpx

from src.core.config import Settings, get_settings


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
            timeout=30.0,
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OllamaEmbeddingProvider:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimension: int = 768,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    cfg = settings or get_settings()
    if cfg.llm_provider.lower() == "ollama":
        return OllamaEmbeddingProvider(base_url=cfg.ollama_base_url)
    return OpenAIEmbeddingProvider(api_key=cfg.openai_api_key)
