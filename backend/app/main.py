"""Application factory. Run with: uvicorn app.main:create_app --factory"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.agent.service import AgentService, GraphRunner
from app.api.agent_routes import router as agent_router
from app.api.rag_routes import router as rag_router
from app.api.repo_routes import router as repo_router
from app.api.routes import health_router, v1_router
from app.core.config import Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.rate_limit import RateLimitMiddleware, RateLimitSettings
from app.core.request_id import add_request_id_middleware
from app.mcp_client import McpGateway
from app.mcp_server import create_mcp_server
from app.mcp_server.server import SandboxLike
from app.providers.base import LLMProvider
from app.rag.service import RagService
from app.storage.runs import RunStore


def create_app(
    settings: Settings | None = None,
    rag: RagService | None = None,
    *,
    graph_runner: GraphRunner | None = None,
    provider: LLMProvider | None = None,
    sandbox: SandboxLike | None = None,
    rate_limits: RateLimitSettings | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "AI coding platform API: document retrieval, agent runs with sandboxed checks, "
            "and human approval before any patch is written to the workspace."
        ),
    )
    app.state.settings = settings
    # The sandbox lives behind the MCP server: the agent reaches it only through the gateway.
    server = create_mcp_server(settings, rag, sandbox=sandbox)
    app.state.gateway = McpGateway(server, timeout_seconds=settings.rag_tool_timeout_seconds + 10)
    # Added first so it sits inside CORS and request-id: 429 responses carry both.
    app.add_middleware(RateLimitMiddleware, config=rate_limits or RateLimitSettings())
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
    run_store = RunStore(Path(settings.data_dir) / "runs.sqlite3")
    # No lifespan: a run left queued or running by a dead process is failed here instead.
    run_store.recover_interrupted()
    app.state.run_store = run_store
    app.state.agent_service = AgentService(
        settings,
        run_store,
        app.state.gateway,
        provider=provider,
        graph_runner=graph_runner,
    )
    app.include_router(agent_router)
    return app
