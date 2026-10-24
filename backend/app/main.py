"""Application factory. Run with: uvicorn app.main:create_app --factory"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.rag_routes import router as rag_router
from app.api.repo_routes import router as repo_router
from app.api.routes import health_router, v1_router
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.request_id import add_request_id_middleware
from app.mcp_client import McpGateway
from app.mcp_server import create_mcp_server
from app.rag.service import RagService


def create_app(settings: Settings | None = None, rag: RagService | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="AI coding platform API: retrieval, agent tasks, sandboxed checks.",
    )
    app.state.settings = settings
    server = create_mcp_server(settings, rag)
    app.state.gateway = McpGateway(server, timeout_seconds=settings.rag_tool_timeout_seconds + 10)
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
    app.include_router(repo_router)
    app.include_router(rag_router)
    return app
