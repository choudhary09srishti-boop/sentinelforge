import asyncio
import json

import httpx
import pytest

from app.core.config import Settings
from app.providers.base import ChatMessage, ProviderError
from app.providers.factory import get_provider
from app.providers.local import LocalProvider, is_remote_url
from app.providers.status import provider_status

MESSAGES = [ChatMessage("system", "be brief"), ChatMessage("user", "hello")]
LOCAL_URL = "http://127.0.0.1:11434/v1"


def test_local_provider_calls_the_openai_compatible_endpoint() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        seen["auth"] = "authorization" in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "local hi"}}]})

    provider = LocalProvider(LOCAL_URL, "qwen", transport=httpx.MockTransport(handler))
    assert asyncio.run(provider.complete(MESSAGES)) == "local hi"
    assert seen["url"] == f"{LOCAL_URL}/chat/completions"
    assert isinstance(seen["body"], dict) and seen["body"]["model"] == "qwen"
    assert seen["auth"] is False
    assert provider.external is False


def test_remote_server_is_reported_as_external() -> None:
    assert LocalProvider("https://llm.example.com/v1", "m").external is True
    assert is_remote_url("http://localhost:8080/v1") is False
    assert is_remote_url("http://[::1]:8080/v1") is False
    assert is_remote_url("http://192.168.1.50:11434/v1") is True


def test_non_http_url_is_rejected() -> None:
    with pytest.raises(ProviderError):
        LocalProvider("file:///etc/passwd", "m")


def test_unreachable_server_gives_an_actionable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider = LocalProvider(LOCAL_URL, "m", transport=httpx.MockTransport(handler))
    with pytest.raises(ProviderError, match="local model server"):
        asyncio.run(provider.complete(MESSAGES))


def test_malformed_reply_raises_provider_error() -> None:
    provider = LocalProvider(
        LOCAL_URL, "m", transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    )
    with pytest.raises(ProviderError):
        asyncio.run(provider.complete(MESSAGES))


def test_factory_requires_a_model_and_builds_the_provider() -> None:
    with pytest.raises(ProviderError):
        get_provider(Settings(_env_file=None, llm_provider="local"))
    provider = get_provider(Settings(_env_file=None, llm_provider="local", local_llm_model="m"))
    assert isinstance(provider, LocalProvider)


def test_status_reports_local_processing() -> None:
    local = provider_status(Settings(_env_file=None, llm_provider="local", local_llm_model="m"))
    assert (local.mode, local.external, local.ready) == ("local", False, True)
    remote = provider_status(
        Settings(
            _env_file=None,
            llm_provider="local",
            local_llm_model="m",
            local_llm_url="http://10.0.0.5:11434/v1",
        )
    )
    assert remote.external is True
    unready = provider_status(Settings(_env_file=None, llm_provider="local"))
    assert unready.ready is False
