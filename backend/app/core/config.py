"""Typed application settings, loaded from environment variables and .env."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

INSECURE_DEFAULT_KEY = "change-me"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "SentinelForge"
    app_env: Literal["development", "production", "test"] = "development"
    api_key: SecretStr = SecretStr(INSECURE_DEFAULT_KEY)
    cors_origins: str = "http://localhost:5173,http://localhost:8501"
    data_dir: str = "./data"
    max_upload_mb: int = Field(default=5, ge=1, le=50)
    workspace_root: str = "./samples/workspace"
    max_read_bytes: int = Field(default=200_000, ge=1_000, le=5_000_000)
    tool_timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    rag_tool_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    model_cache_dir: str = "./models"
    chunk_max_chars: int = Field(default=1000, ge=200, le=4000)
    llm_provider: Literal["demo", "groq"] = "demo"
    groq_api_key: SecretStr = SecretStr("")
    groq_model: str = ""

    @model_validator(mode="after")
    def _reject_insecure_production(self) -> Self:
        key = self.api_key.get_secret_value()
        if self.app_env == "production" and key == INSECURE_DEFAULT_KEY:
            raise ValueError("API_KEY must be changed in production")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
