"""Typed request and response models for the HTTP API."""

from pydantic import BaseModel, Field

from app.providers.status import ProviderStatus


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    app: str
    version: str
    environment: str
    provider: ProviderStatus


class ConfigResponse(BaseModel):
    app: str
    environment: str
    provider: ProviderStatus
    max_upload_mb: int


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody
