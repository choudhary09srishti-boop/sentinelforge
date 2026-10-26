"""Pick the configured provider. Missing configuration fails loudly instead of silently demoing."""

from app.core.config import Settings
from app.providers.base import LLMProvider, ProviderError
from app.providers.demo import DemoProvider
from app.providers.groq import GroqProvider


def get_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "groq":
        if not settings.groq_configured or not settings.groq_model:
            raise ProviderError("LLM_PROVIDER=groq requires GROQ_API_KEY and GROQ_MODEL.")
        return GroqProvider(settings.groq_api_key.get_secret_value(), settings.groq_model)
    return DemoProvider()
