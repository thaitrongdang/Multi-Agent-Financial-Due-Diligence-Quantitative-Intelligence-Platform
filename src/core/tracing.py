from typing import Any

from src.core.config import Settings, get_settings


def get_langfuse_callback(
    settings: Settings | None = None,
    trace_name: str = "financial-due-diligence",
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    """Return a Langfuse callback handler when keys are configured, else None."""
    cfg = settings or get_settings()
    if not cfg.langfuse_public_key or not cfg.langfuse_secret_key:
        return None

    handler_cls = None
    try:
        from langfuse.langchain import CallbackHandler

        handler_cls = CallbackHandler
    except ImportError:
        try:
            from langfuse.callback import CallbackHandler

            handler_cls = CallbackHandler
        except ImportError:
            return None

    try:
        return handler_cls(
            public_key=cfg.langfuse_public_key,
            secret_key=cfg.langfuse_secret_key,
            host=cfg.langfuse_host,
            trace_name=trace_name,
            metadata=metadata or {},
        )
    except TypeError:
        return handler_cls()
