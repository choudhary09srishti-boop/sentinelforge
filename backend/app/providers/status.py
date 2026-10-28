"""Describes the active LLM provider mode and where user data is processed."""

from pydantic import BaseModel

from app.core.config import Settings
from app.providers.local import is_remote_url


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
    if settings.llm_provider == "local":
        remote = is_remote_url(settings.local_llm_url)
        notice = (
            "Local model server on a non-local address: prompts leave this machine."
            if remote
            else "Local model: prompts stay on this machine."
        )
        return ProviderStatus(
            mode="local",
            external=remote,
            ready=bool(settings.local_llm_model),
            data_notice=notice,
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
