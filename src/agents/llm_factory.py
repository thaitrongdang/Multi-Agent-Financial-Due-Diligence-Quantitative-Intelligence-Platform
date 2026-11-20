from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from src.core.config import Settings, get_settings


def get_chat_model(
    settings: Settings | None = None,
    model_name: str | None = None,
    temperature: float = 0.0,
) -> BaseChatModel:
    cfg = settings or get_settings()
    provider = cfg.llm_provider.strip().lower()

    if provider == "groq":
        return ChatGroq(
            api_key=cfg.groq_api_key,
            model=model_name or "llama-3.3-70b-versatile",
            temperature=temperature,
        )

    if provider == "ollama":
        return ChatOllama(
            base_url=cfg.ollama_base_url,
            model=model_name or "qwen2.5:7b",
            temperature=temperature,
        )

    return ChatOpenAI(
        api_key=cfg.openai_api_key,
        model=model_name or "gpt-4o-mini",
        temperature=temperature,
    )
