"""HTTP routes. Business logic lives elsewhere; routes stay thin."""

from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import SettingsDep, require_api_key
from app.api.schemas import ConfigResponse, ErrorResponse, HealthResponse
from app.providers.status import provider_status

health_router = APIRouter(tags=["health"])

v1_router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(require_api_key)],
    responses={401: {"model": ErrorResponse}},
)


@health_router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
        environment=settings.app_env,
        provider=provider_status(settings),
    )


@v1_router.get("/config", response_model=ConfigResponse, tags=["config"])
def get_config(settings: SettingsDep) -> ConfigResponse:
    return ConfigResponse(
        app=settings.app_name,
        environment=settings.app_env,
        provider=provider_status(settings),
        max_upload_mb=settings.max_upload_mb,
    )
