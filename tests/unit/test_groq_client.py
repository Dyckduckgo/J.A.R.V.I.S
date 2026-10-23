"""GroqChatClient tests.

The real ``groq`` SDK and :class:`GroqChatClient` run unmodified; only the
HTTP transport is replaced (``httpx.MockTransport``) so we can assert on the
exact wire format and replay realistic Groq responses, including SSE streams.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable

import httpx
import pytest

from jarvis.errors import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMRequestError,
    LLMServiceError,
    LLMToolCallError,
    OperationCancelled,
)
from jarvis.llm.base import ChatMessage, ToolCall
from jarvis.llm.groq_client import GroqChatClient

API_KEY = "gsk_" + "testkey0" * 6
MODEL = "llama-3.3-70b-versatile"
TOOLS = [{"type": "function", "function": {"name": "calculate", "description": "math", "parameters": {"type": "object", "properties": {}}}}]


def make_client(handler: Callable[[httpx.Request], httpx.Response], max_retries: int = 0) -> GroqChatClient:
    return GroqChatClient(
        API_KEY, max_retries=max_retries, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def chunk(delta: dict, finish: str | None = None, **extra) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **extra,
    }


def sse(*events: dict, done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    if done:
        body += "data: [DONE]\n\n"
    return body.encode()


def sse_response(payload: bytes) -> httpx.Response:
    return httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})


def error_response(status: int, code: str, message: str, headers: dict | None = None) -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"message": message, "type": "invalid_request_error", "code": code}}, headers=headers
    )


def test_streaming_text_and_request_format() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return sse_response(
            sse(
                chunk({"role": "assistant", "content": "Hel"}),
                chunk({"content": "lo!"}),
                chunk({}, "stop", x_groq={"id": "req_1", "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}),
            )
        )

    call = ToolCall("call_9", "calculate", '{"expression": "1+1"}')
    messages = [
        ChatMessage.system("sys"),
        ChatMessage.user("hi"),
        ChatMessage.assistant(None, [call]),
        ChatMessage.tool(call, "2"),
    ]
    deltas: list[str] = []
    result = make_client(handler).complete(messages, model=MODEL, tools=TOOLS, temperature=0.2, max_tokens=256, on_text=deltas.append)

    assert deltas == ["Hel", "lo!"]
    assert result.content == "Hello!" and result.tool_calls == () and result.finish_reason == "stop"
    assert result.usage is not None and result.usage.total_tokens == 15
    request = seen[0]
    assert request.url.path.endswith("/openai/v1/chat/completions")
    assert request.headers["authorization"] == f"Bearer {API_KEY}"
    body = json.loads(request.content)
    assert body["model"] == MODEL and body["stream"] is True
    assert body["temperature"] == 0.2 and body["max_completion_tokens"] == 256
    assert body["tools"] == TOOLS and body["tool_choice"] == "auto"
    assert body["messages"][2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call_9", "type": "function", "function": {"name": "calculate", "arguments": '{"expression": "1+1"}'}}],
    }
    assert body["messages"][3] == {"role": "tool", "content": "2", "tool_call_id": "call_9", "name": "calculate"}


def test_streaming_tool_calls_are_reassembled() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(
            sse(
                chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "call_a", "type": "function", "function": {"name": "calculate", "arguments": '{"expr'}}]}),
                chunk({"tool_calls": [{"index": 0, "function": {"arguments": 'ession": "6*7"}'}}]}),
                chunk({"tool_calls": [{"index": 1, "id": "call_b", "type": "function", "function": {"name": "get_datetime", "arguments": "{}"}}]}),
                chunk({}, "tool_calls"),
            )
        )

    result = make_client(handler).complete([ChatMessage.user("x")], model=MODEL, tools=TOOLS)
    assert result.finish_reason == "tool_calls"
    assert result.tool_calls == (
        ToolCall("call_a", "calculate", '{"expression": "6*7"}'),
        ToolCall("call_b", "get_datetime", "{}"),
    )
    assert result.tool_calls[0].parsed_arguments() == {"expression": "6*7"}


def test_non_streaming_completion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content).get("stream") in (None, False)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-2",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL,
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"id": "call_z", "type": "function", "function": {"name": "calculate", "arguments": '{"expression":"2"}'}}],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
            },
        )

    result = make_client(handler).complete([ChatMessage.user("x")], model=MODEL, stream=False)
    assert result.tool_calls == (ToolCall("call_z", "calculate", '{"expression":"2"}'),)
    assert result.usage.completion_tokens == 7


def test_cancel_during_stream() -> None:
    cancel = threading.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(sse(chunk({"content": "one "}), chunk({"content": "two "}), chunk({"content": "three"}), chunk({}, "stop")))

    received: list[str] = []

    def on_text(text: str) -> None:
        received.append(text)
        cancel.set()

    with pytest.raises(OperationCancelled):
        make_client(handler).complete([ChatMessage.user("x")], model=MODEL, on_text=on_text, cancel=cancel)
    assert received == ["one "]


def test_pre_cancelled_request_is_not_sent() -> None:
    calls = []
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(OperationCancelled):
        make_client(lambda r: calls.append(r) or sse_response(sse())).complete([ChatMessage.user("x")], model=MODEL, cancel=cancel)
    assert calls == []


@pytest.mark.parametrize(
    ("response", "error_type", "fragment"),
    [
        (error_response(401, "invalid_api_key", "Invalid API Key"), LLMAuthenticationError, "rejected the API key"),
        (error_response(429, "rate_limit_exceeded", "slow down", {"retry-after": "7"}), LLMRateLimitError, "about 7 seconds"),
        (error_response(400, "tool_use_failed", "Failed to call a function"), LLMToolCallError, "invalid tool call"),
        (error_response(404, "model_not_found", "The model does not exist"), LLMRequestError, "is not available"),
        (error_response(413, "request_too_large", "Request too large"), LLMRequestError, "too large"),
        (error_response(503, "service_unavailable", "over capacity"), LLMServiceError, "server error"),
        (error_response(400, "invalid_request_error", "messages: bad"), LLMRequestError, "messages: bad"),
    ],
)
def test_http_errors_are_mapped(response: httpx.Response, error_type: type, fragment: str) -> None:
    client = make_client(lambda request: response)
    with pytest.raises(error_type) as info:
        client.complete([ChatMessage.user("x")], model=MODEL)
    assert fragment in info.value.user_message
    assert API_KEY not in info.value.user_message and API_KEY not in str(info.value)


def test_error_event_inside_stream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(sse(chunk({"content": "partial"}), {"error": {"message": "stream broke", "type": "internal_server_error"}}, done=False))

    with pytest.raises(LLMServiceError) as info:
        make_client(handler).complete([ChatMessage.user("x")], model=MODEL)
    assert "stream broke" in info.value.user_message


def test_tool_use_failed_inside_stream_maps_to_tool_call_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return sse_response(sse({"error": {"message": "bad call", "type": "invalid_request_error", "code": "tool_use_failed"}}, done=False))

    with pytest.raises(LLMToolCallError):
        make_client(handler).complete([ChatMessage.user("x")], model=MODEL)


def test_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with pytest.raises(LLMConnectionError) as info:
        make_client(handler).complete([ChatMessage.user("x")], model=MODEL)
    assert "internet connection" in info.value.user_message


def test_transient_errors_are_retried_by_sdk() -> None:
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) == 1:
            return error_response(503, "service_unavailable", "busy", {"retry-after-ms": "10"})
        return sse_response(sse(chunk({"content": "ok"}), chunk({}, "stop")))

    result = make_client(handler, max_retries=1).complete([ChatMessage.user("x")], model=MODEL)
    assert result.content == "ok" and len(attempts) == 2


def test_list_models_filters_non_chat_and_inactive() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/openai/v1/models")
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "openai/gpt-oss-120b", "object": "model", "created": 1, "owned_by": "OpenAI", "active": True},
                    {"id": "llama-3.3-70b-versatile", "object": "model", "created": 1, "owned_by": "Meta", "active": True},
                    {"id": "whisper-large-v3", "object": "model", "created": 1, "owned_by": "OpenAI", "active": True},
                    {"id": "meta-llama/llama-prompt-guard-2-22m", "object": "model", "created": 1, "owned_by": "Meta"},
                    {"id": "retired-model", "object": "model", "created": 1, "owned_by": "x", "active": False},
                ],
            },
        )

    assert make_client(handler).list_models() == ["llama-3.3-70b-versatile", "openai/gpt-oss-120b"]


def test_list_models_auth_error() -> None:
    with pytest.raises(LLMAuthenticationError):
        make_client(lambda r: error_response(401, "invalid_api_key", "Invalid API Key")).list_models()


def test_empty_key_rejected() -> None:
    with pytest.raises(LLMAuthenticationError):
        GroqChatClient("")
