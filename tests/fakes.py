"""Test doubles shared across the suite.

These replace only *external* boundaries (the OS credential store and the
remote LLM). Everything else in the tests runs the real implementation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from jarvis.errors import OperationCancelled
from jarvis.llm.base import ChatMessage, LLMResponse, TextCallback


class MemoryKeyring:
    """In-memory stand-in for the :mod:`keyring` module."""

    def __init__(self, fail: bool = False) -> None:
        self.store: dict[tuple[str, str], str] = {}
        self.fail = fail

    def get_password(self, service: str, username: str) -> str | None:
        if self.fail:
            raise RuntimeError("keyring backend unavailable")
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        if self.fail:
            raise RuntimeError("keyring backend unavailable")
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if self.fail:
            raise RuntimeError("keyring backend unavailable")
        self.store.pop((service, username), None)


@dataclass
class RecordedCall:
    messages: list[ChatMessage]
    model: str
    tools: list[dict[str, Any]] | None
    temperature: float
    max_tokens: int
    stream: bool


Step = LLMResponse | Exception | Callable[[RecordedCall], LLMResponse]


@dataclass
class ScriptedChatClient:
    """A :class:`jarvis.llm.base.ChatClient` that replays scripted responses.

    Each entry in ``script`` is an :class:`LLMResponse` to return, an
    exception to raise, or a callable producing a response from the call.
    """

    script: list[Step] = field(default_factory=list)
    models: list[str] = field(default_factory=lambda: ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"])
    chunk_size: int = 4
    calls: list[RecordedCall] = field(default_factory=list)
    list_models_error: Exception | None = None

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
        call = RecordedCall(list(messages), model, list(tools) if tools else None, temperature, max_tokens, stream)
        self.calls.append(call)
        if cancel is not None and cancel.is_set():
            raise OperationCancelled()
        if not self.script:
            raise AssertionError("ScriptedChatClient ran out of scripted responses")
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        response = step(call) if callable(step) else step
        if stream and on_text and response.content:
            for i in range(0, len(response.content), self.chunk_size):
                if cancel is not None and cancel.is_set():
                    raise OperationCancelled()
                on_text(response.content[i : i + self.chunk_size])
        return response

    def list_models(self) -> list[str]:
        if self.list_models_error:
            raise self.list_models_error
        return list(self.models)
