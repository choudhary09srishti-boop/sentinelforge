import asyncio
import json

import httpx
import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.providers.base import ChatMessage, ProviderError
from app.providers.demo import DemoProvider
from app.providers.factory import get_provider
from app.providers.groq import GroqProvider

MESSAGES = [ChatMessage("system", "be brief"), ChatMessage("user", "hello")]


def _groq(handler: httpx.MockTransport) -> GroqProvider:
    return GroqProvider("secret-key-123", "test-model", transport=handler)


def test_demo_provider_is_offline_and_not_external() -> None:
    provider = DemoProvider()
    text = asyncio.run(provider.complete(MESSAGES))
    assert "[demo provider]" in text and "hello" in text
    assert provider.external is False


def test_groq_sends_expected_request_and_parses_reply() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi there"}}]})

    provider = _groq(httpx.MockTransport(handler))
    assert asyncio.run(provider.complete(MESSAGES)) == "hi there"
    assert seen["auth"] == "Bearer secret-key-123"
    body = seen["body"]
    assert isinstance(body, dict) and body["model"] == "test-model"
    assert provider.external is True


def test_groq_error_does_not_leak_api_key() -> None:
    provider = _groq(httpx.MockTransport(lambda r: httpx.Response(401, text="bad secret-key-123")))
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.complete(MESSAGES))
    assert "secret-key-123" not in str(info.value)
    assert "401" in str(info.value)


def test_groq_malformed_response_raises_provider_error() -> None:
    provider = _groq(httpx.MockTransport(lambda r: httpx.Response(200, json={"unexpected": 1})))
    with pytest.raises(ProviderError):
        asyncio.run(provider.complete(MESSAGES))


def test_groq_network_failure_raises_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(ProviderError):
        asyncio.run(_groq(httpx.MockTransport(handler)).complete(MESSAGES))


def test_factory_demo_by_default_and_groq_requires_config() -> None:
    assert isinstance(get_provider(Settings(llm_provider="demo")), DemoProvider)
    with pytest.raises(ProviderError):
        get_provider(Settings(llm_provider="groq", groq_api_key=SecretStr(""), groq_model=""))
    configured = Settings(llm_provider="groq", groq_api_key=SecretStr("k"), groq_model="m")
    assert isinstance(get_provider(configured), GroqProvider)
