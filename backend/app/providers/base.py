"""LLM provider interface. The agent depends only on this, never on a vendor SDK."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


class ProviderError(Exception):
    """The provider failed (config, network, auth, bad response). Message is safe to show."""


class LLMProvider(Protocol):
    name: str
    external: bool  # True when prompts leave this machine

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> str: ...
