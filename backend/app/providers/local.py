"""Local chat provider for any OpenAI-compatible server (Ollama, llama.cpp server).

Prompts stay on this machine when the server is on loopback. A non-loopback URL is reported
as external so the UI never claims local processing it cannot guarantee.
"""

from collections.abc import Sequence
from urllib.parse import urlparse

import httpx

from app.providers.base import ChatMessage, ProviderError

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def is_remote_url(url: str) -> bool:
    """True unless the URL points at this machine."""
    return (urlparse(url).hostname or "") not in LOOPBACK_HOSTS


class LocalProvider:
    name = "local"
    external: bool

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if urlparse(base_url).scheme not in ("http", "https"):
            raise ProviderError("LOCAL_LLM_URL must start with http:// or https://.")
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._timeout = timeout
        self._transport = transport
        self.external = is_remote_url(base_url)

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(self._url, json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(
                "Could not reach the local model server. Is Ollama or llama.cpp running?"
            ) from exc
        if response.status_code != 200:
            raise ProviderError(f"The local model server returned HTTP {response.status_code}.")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("The local model server returned an unexpected response.") from exc
        if not isinstance(content, str):
            raise ProviderError("The local model server returned an unexpected response.")
        return content
