from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from src.agents.llm_factory import get_chat_model
from src.core.config import get_settings
from src.quant.data_fetcher import YFinanceDataFetcher
from src.rag.embeddings import get_embedding_provider
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.parser import FinancialPDFParser
from src.rag.reranker import FinancialReranker
from src.rag.vector_store import QdrantVectorStore


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()
    embedder = get_embedding_provider(settings)
    store = QdrantVectorStore(embedder=embedder, url=settings.qdrant_url, in_memory=False)
    return HybridRetriever(vector_store=store)


def get_llm_client() -> BaseChatModel:
    return get_chat_model(get_settings())


def get_data_fetcher() -> YFinanceDataFetcher:
    return YFinanceDataFetcher()


def get_reranker() -> FinancialReranker:
    return FinancialReranker()


def get_pdf_parser() -> FinancialPDFParser:
    return FinancialPDFParser()
