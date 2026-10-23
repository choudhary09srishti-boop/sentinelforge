"""Shared FastAPI dependencies: settings access and API-key auth."""

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from app.core.config import Settings


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def require_api_key(
    settings: SettingsDep, x_api_key: Annotated[str | None, Header()] = None
) -> None:
    expected = settings.api_key.get_secret_value().encode()
    if x_api_key is None or not secrets.compare_digest(x_api_key.encode(), expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")
