"""Groq chat-completions client built on the official ``groq`` SDK.

Responsibilities:

* translate :class:`ChatMessage` objects to the OpenAI-compatible wire format;
* stream text deltas and reassemble streamed tool calls;
* honour cancellation between stream chunks;
* map SDK exceptions to the :mod:`jarvis.errors` hierarchy with messages that
  are safe to show to the user (no keys, no raw response bodies).

Transient failures (429, 5xx, connection errors) are retried by the SDK with
exponential back-off (``max_retries``).
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import groq
import httpx

from jarvis.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServiceError,
    LLMToolCallError,
    OperationCancelled,
)
from jarvis.llm.base import ChatMessage, LLMResponse, TextCallback, TokenUsage, ToolCall

log = logging.getLogger(__name__)

# Model families that are not chat models (speech, TTS, safety classifiers).
_NON_CHAT_MARKERS = ("whisper", "tts", "orpheus", "playai", "prompt-guard", "distil-whisper")


class GroqChatClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise LLMAuthenticationError("empty API key", user_message="No Groq API key is configured.")
        self._client = groq.Groq(
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            base_url=base_url,
            http_client=http_client,
        )

    def close(self) -> None:
        self._client.close()

    # -- public API ----------------------------------------------------------

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
        if cancel is not None and cancel.is_set():
            raise OperationCancelled()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_api() for m in messages],
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        log.info("Groq request: model=%s messages=%d tools=%d stream=%s", model, len(messages), len(tools or ()), stream)
        try:
            if stream:
                return self._complete_streaming(payload, on_text, cancel)
            return self._complete_blocking(payload, on_text)
        except groq.GroqError as exc:
            raise _map_error(exc, model) from exc

    def list_models(self) -> list[str]:
        try:
            page = self._client.models.list()
        except groq.GroqError as exc:
            raise _map_error(exc, None) from exc
        ids = []
        for item in getattr(page, "data", []) or []:
            model_id = getattr(item, "id", None)
            if not model_id or any(marker in model_id.lower() for marker in _NON_CHAT_MARKERS):
                continue
            if getattr(item, "active", True) is False:
                continue
            ids.append(model_id)
        return sorted(set(ids))

    # -- internals -----------------------------------------------------------

    def _complete_blocking(self, payload: dict[str, Any], on_text: TextCallback | None) -> LLMResponse:
        completion = self._client.chat.completions.create(**payload)
        if not completion.choices:
            raise LLMServiceError("empty choices", user_message="The model returned an empty response.")
        choice = completion.choices[0]
        message = choice.message
        calls = tuple(
            ToolCall(
                id=call.id or _new_call_id(),
                name=call.function.name,
                arguments=call.function.arguments or "{}",
            )
            for call in (message.tool_calls or [])
        )
        content = message.content or ""
        if content and on_text:
            on_text(content)
        return LLMResponse(
            content=content,
            tool_calls=calls,
            finish_reason=choice.finish_reason,
            usage=_usage(completion.usage),
            model=completion.model,
        )

    def _complete_streaming(
        self, payload: dict[str, Any], on_text: TextCallback | None, cancel: threading.Event | None
    ) -> LLMResponse:
        stream = self._client.chat.completions.create(**payload, stream=True)
        text_parts: list[str] = []
        pending: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        usage: TokenUsage | None = None
        model: str | None = None
        try:
            for chunk in stream:
                if cancel is not None and cancel.is_set():
                    raise OperationCancelled()
                model = chunk.model or model
                x_groq = getattr(chunk, "x_groq", None)
                if x_groq is not None and getattr(x_groq, "usage", None) is not None:
                    usage = _usage(x_groq.usage)
                if getattr(chunk, "usage", None) is not None:
                    usage = _usage(chunk.usage)
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta.content:
                        text_parts.append(delta.content)
                        if on_text:
                            on_text(delta.content)
                    for call in delta.tool_calls or ():
                        slot = pending.setdefault(call.index, {"id": None, "name": "", "args": []})
                        if call.id:
                            slot["id"] = call.id
                        if call.function is not None:
                            if call.function.name and not slot["name"]:
                                slot["name"] = call.function.name
                            if call.function.arguments:
                                slot["args"].append(call.function.arguments)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
        finally:
            stream.close()

        calls = tuple(
            ToolCall(id=slot["id"] or _new_call_id(), name=slot["name"], arguments="".join(slot["args"]) or "{}")
            for _, slot in sorted(pending.items())
            if slot["name"]
        )
        return LLMResponse("".join(text_parts), calls, finish_reason, usage, model)


def _new_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _usage(raw: Any) -> TokenUsage | None:
    if raw is None:
        return None
    return TokenUsage(
        prompt_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
        total_tokens=int(getattr(raw, "total_tokens", 0) or 0),
    )


def _error_details(exc: groq.APIError) -> tuple[str | None, str]:
    """Return (error code, provider message) from an SDK error body."""
    body = exc.body
    if isinstance(body, Mapping):
        inner = body.get("error", body)
        if isinstance(inner, Mapping):
            return (inner.get("code") or inner.get("type")), str(inner.get("message") or exc.message)
    return None, exc.message


def _map_error(exc: groq.GroqError, model: str | None) -> LLMError:
    if isinstance(exc, groq.APITimeoutError):
        return LLMConnectionError(str(exc), user_message="The request to Groq timed out. Check your connection and try again.")
    if isinstance(exc, groq.APIConnectionError):
        return LLMConnectionError(str(exc), user_message="Could not reach the Groq API. Check your internet connection.")
    if not isinstance(exc, groq.APIError):
        return LLMError(str(exc), user_message="Unexpected error from the Groq client.")

    code, provider_message = _error_details(exc)
    status = getattr(exc, "status_code", None)
    detail = f"HTTP {status} code={code}: {provider_message}"

    if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
        return LLMAuthenticationError(
            detail, user_message="Groq rejected the API key. Check the key in Settings (it may be invalid or revoked)."
        )
    if isinstance(exc, groq.RateLimitError):
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after")
        hint = f" Try again in about {retry_after} seconds." if retry_after else " Wait a moment and try again."
        return LLMRateLimitError(detail, user_message="Groq rate limit reached." + hint)
    if code == "tool_use_failed":
        return LLMToolCallError(detail, user_message="The model produced an invalid tool call. Try rephrasing the request.")
    if isinstance(exc, groq.NotFoundError) or code == "model_not_found":
        return LLMRequestError(
            detail,
            user_message=f"The model '{model}' is not available. Choose another model in Settings.",
        )
    if status == 413 or code in ("context_length_exceeded", "request_too_large"):
        return LLMRequestError(
            detail,
            user_message=(
                "The conversation is too large for this model or your Groq plan's limits. "
                "Start a new conversation or lower 'History messages' in Settings."
            ),
        )
    if isinstance(exc, groq.InternalServerError) or (status is not None and status >= 500):
        return LLMServiceError(detail, user_message="Groq is having problems right now (server error). Try again shortly.")
    if isinstance(exc, groq.APIStatusError):
        return LLMRequestError(detail, user_message=f"Groq rejected the request: {provider_message}")
    # APIError raised from inside an SSE stream (no HTTP status available).
    return LLMServiceError(detail, user_message=f"The Groq stream reported an error: {provider_message}")
