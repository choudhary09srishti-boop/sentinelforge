"""Application factory. Run with: uvicorn app.main:create_app --factory"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import health_router, v1_router
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.request_id import add_request_id_middleware


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="AI coding platform API: retrieval, agent tasks, sandboxed checks.",
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-API-Key", "Content-Type"],
    )
    add_request_id_middleware(app)
    register_error_handlers(app)
    app.include_router(health_router)
    app.include_router(v1_router)
    return app
