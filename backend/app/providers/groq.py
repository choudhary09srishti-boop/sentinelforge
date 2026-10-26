"""Groq chat provider over its OpenAI-compatible HTTP API. The key never appears in errors."""

from collections.abc import Sequence

import httpx

from app.providers.base import ChatMessage, ProviderError

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider:
    name = "groq"
    external = True

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout
        self._transport = transport

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
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, transport=self._transport
            ) as client:
                response = await client.post(GROQ_URL, json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise ProviderError("Could not reach Groq.") from exc
        if response.status_code != 200:
            raise ProviderError(f"Groq returned HTTP {response.status_code}.")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Groq returned an unexpected response.") from exc
        if not isinstance(content, str):
            raise ProviderError("Groq returned an unexpected response.")
        return content
