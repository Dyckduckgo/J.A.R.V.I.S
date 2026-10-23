"""Provider-neutral chat types and the :class:`ChatClient` protocol."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from jarvis.errors import ToolArgumentError


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class ToolCall:
    """A function call requested by the model. ``arguments`` is the raw JSON text."""

    id: str
    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> dict[str, Any]:
        text = (self.arguments or "").strip() or "{}"
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ToolArgumentError(f"Arguments are not valid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ToolArgumentError("Arguments must be a JSON object.")
        return value

    def to_api(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments or "{}"},
        }

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        return cls(id=str(data["id"]), name=str(data["name"]), arguments=str(data.get("arguments") or "{}"))


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    created_at: float = field(default_factory=time.time)

    @classmethod
    def system(cls, text: str) -> ChatMessage:
        return cls(Role.SYSTEM, text)

    @classmethod
    def user(cls, text: str) -> ChatMessage:
        return cls(Role.USER, text)

    @classmethod
    def assistant(cls, text: str | None, tool_calls: Sequence[ToolCall] = ()) -> ChatMessage:
        return cls(Role.ASSISTANT, text, tuple(tool_calls))

    @classmethod
    def tool(cls, call: ToolCall, content: str) -> ChatMessage:
        return cls(Role.TOOL, content, tool_call_id=call.id, name=call.name)

    def to_api(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role.value}
        if self.role is Role.ASSISTANT and self.tool_calls:
            data["content"] = self.content or None
            data["tool_calls"] = [call.to_api() for call in self.tool_calls]
        else:
            data["content"] = self.content or ""
        if self.role is Role.TOOL:
            data["tool_call_id"] = self.tool_call_id
            if self.name:
                data["name"] = self.name
        return data


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


@dataclass(frozen=True)
class LLMResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    model: str | None = None


TextCallback = Callable[[str], None]


class ChatClient(Protocol):
    """A chat-completions backend that supports function calling."""

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str,
        tools: Sequence[dict[str, Any]] | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        stream: bool = True,
        on_text: TextCallback | None = None,
        cancel: threading.Event | None = None,
    ) -> LLMResponse:
        """Run one completion. Streams text through ``on_text`` when ``stream``.

        Raises :class:`jarvis.errors.LLMError` subclasses on failure and
        :class:`jarvis.errors.OperationCancelled` if ``cancel`` is set.
        """
        ...

    def list_models(self) -> list[str]:
        """Return the model IDs available to this account."""
        ...
