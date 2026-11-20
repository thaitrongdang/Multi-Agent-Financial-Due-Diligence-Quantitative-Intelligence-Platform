from fastapi import FastAPI

from src.api.routes.health import router as health_router
from src.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
    )
    app.include_router(health_router, prefix="/api/v1")
    return app


app = create_app()
