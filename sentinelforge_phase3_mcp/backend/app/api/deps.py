"""Shared FastAPI dependencies: settings, MCP gateway and API-key auth."""

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from app.core.config import Settings
from app.mcp_client import McpGateway


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_gateway(request: Request) -> McpGateway:
    gateway: McpGateway = request.app.state.gateway
    return gateway


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
GatewayDep = Annotated[McpGateway, Depends(get_gateway)]


def require_api_key(
    settings: SettingsDep, x_api_key: Annotated[str | None, Header()] = None
) -> None:
    expected = settings.api_key.get_secret_value().encode()
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
