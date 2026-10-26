"""Keyless offline provider: lets the whole app run and be tested with no external calls."""

from collections.abc import Sequence

from app.providers.base import ChatMessage


class DemoProvider:
    name = "demo"
    external = False

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return f"[demo provider] No LLM is configured. Last request: {last_user[:200]}"
