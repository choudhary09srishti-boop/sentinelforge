"""Pick the configured provider. Missing configuration fails loudly instead of silently demoing."""

from app.core.config import Settings
from app.providers.base import LLMProvider, ProviderError
from app.providers.demo import DemoProvider
from app.providers.groq import GroqProvider
from app.providers.local import LocalProvider


def get_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "groq":
        if not settings.groq_configured or not settings.groq_model:
            raise ProviderError("LLM_PROVIDER=groq requires GROQ_API_KEY and GROQ_MODEL.")
        return GroqProvider(settings.groq_api_key.get_secret_value(), settings.groq_model)
    if settings.llm_provider == "local":
        if not settings.local_llm_model:
            raise ProviderError("LLM_PROVIDER=local requires LOCAL_LLM_MODEL.")
        return LocalProvider(settings.local_llm_url, settings.local_llm_model)
    return DemoProvider()
