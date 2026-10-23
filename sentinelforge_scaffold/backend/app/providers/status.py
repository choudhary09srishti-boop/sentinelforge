"""Describes the active LLM provider mode and where user data is processed."""

from pydantic import BaseModel

from app.core.config import Settings


class ProviderStatus(BaseModel):
    mode: str
    external: bool
    ready: bool
    data_notice: str


def provider_status(settings: Settings) -> ProviderStatus:
    if settings.llm_provider == "groq":
        return ProviderStatus(
            mode="groq",
            external=True,
            ready=settings.groq_configured and bool(settings.groq_model),
            data_notice=(
                "External provider: prompts, retrieved snippets and code excerpts "
                "are sent to Groq's servers."
            ),
        )
    return ProviderStatus(
        mode="demo",
        external=False,
        ready=True,
        data_notice=(
            "Demo mode: no external LLM is used. Output is rule-based, labelled "
            "as demo output, and all data stays on this server."
        ),
    )
