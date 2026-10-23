from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fakes import MemoryKeyring, ScriptedChatClient

from jarvis.agent.events import AgentCallbacks
from jarvis.agent.service import MAX_INPUT_CHARS, title_from_text
from jarvis.app_context import AppContext, GroqClientFactory
from jarvis.config.secrets import SecretStore
from jarvis.config.settings import Settings
from jarvis.errors import LLMRateLimitError, MissingAPIKeyError, StorageError
from jarvis.llm.base import LLMResponse, Role, ToolCall
from jarvis.llm.groq_client import GroqChatClient
from jarvis.paths import AppPaths
from jarvis.storage.database import DEFAULT_TITLE
from jarvis.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from jarvis.tools.registry import ToolRegistry

KEY = "gsk_" + "Z9y8X7w6" * 6


class NoteTool(Tool):
    name = "remember"
    description = "Store a note."
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    risk = RiskLevel.SAFE

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        assert ctx.notes is not None
        note = ctx.notes.add_note(args["text"])
        return ToolResult.success(f"saved note {note.id}")


@pytest.fixture
def ctx(tmp_path: Path) -> AppContext:
    client = ScriptedChatClient()
    context = AppContext.create(
        AppPaths.under(tmp_path / "home"),
        secrets=SecretStore(backend=MemoryKeyring(), env={}),
        client_factory=lambda: client,
        registry=ToolRegistry([NoteTool()]),
    )
    context.fake_client = client  # type: ignore[attr-defined]
    yield context
    context.close()


def test_send_message_persists_turn_and_titles_conversation(ctx: AppContext) -> None:
    client: ScriptedChatClient = ctx.fake_client  # type: ignore[attr-defined]
    client.script += [
        LLMResponse("", (ToolCall("c1", "remember", '{"text": "dentist on Friday"}'),), "tool_calls"),
        LLMResponse("Noted.", (), "stop"),
    ]
    conv = ctx.store.create_conversation()
    result = ctx.chat.send_message(conv.id, "  Remember: dentist on Friday  ", AgentCallbacks(), model="m-x")

    assert result.final_text == "Noted."
    stored = ctx.store.get_messages(conv.id)
    assert [m.role for m in stored] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    assert stored[0].content == "Remember: dentist on Friday"
    refreshed = ctx.store.get_conversation(conv.id)
    assert refreshed.title == "Remember: dentist on Friday" and refreshed.model == "m-x"
    assert [n.content for n in ctx.store.search_notes("dentist")] == ["dentist on Friday"]

    # Second turn: history is replayed, title is not overwritten.
    client.script.append(LLMResponse("Friday.", (), "stop"))
    ctx.chat.send_message(conv.id, "When is it?", AgentCallbacks())
    assert ctx.store.get_conversation(conv.id).title == "Remember: dentist on Friday"
    replayed = [m.content for m in client.calls[-1].messages if m.role is Role.USER]
    assert replayed == ["Remember: dentist on Friday", "When is it?"]


def test_invalid_input_is_rejected_without_side_effects(ctx: AppContext) -> None:
    conv = ctx.store.create_conversation()
    with pytest.raises(ValueError):
        ctx.chat.send_message(conv.id, "   ", AgentCallbacks())
    with pytest.raises(ValueError):
        ctx.chat.send_message(conv.id, "x" * (MAX_INPUT_CHARS + 1), AgentCallbacks())
    with pytest.raises(StorageError):
        ctx.chat.send_message(conv.id + 100, "hello", AgentCallbacks())
    assert ctx.store.get_messages(conv.id) == []
    assert ctx.store.get_conversation(conv.id).title == DEFAULT_TITLE


def test_failed_turn_without_reply_is_rolled_back(ctx: AppContext) -> None:
    client: ScriptedChatClient = ctx.fake_client  # type: ignore[attr-defined]
    client.script.append(LLMRateLimitError("429", user_message="slow down"))
    conv = ctx.store.create_conversation()
    with pytest.raises(LLMRateLimitError):
        ctx.chat.send_message(conv.id, "first try", AgentCallbacks())
    assert ctx.store.get_messages(conv.id) == []
    assert ctx.store.get_conversation(conv.id).title == DEFAULT_TITLE

    # Retrying the same text now yields exactly one copy of it.
    client.script.append(LLMResponse("ok", (), "stop"))
    ctx.chat.send_message(conv.id, "first try", AgentCallbacks())
    assert [m.content for m in ctx.store.get_messages(conv.id)] == ["first try", "ok"]


def test_failed_turn_after_tool_output_is_kept(ctx: AppContext) -> None:
    client: ScriptedChatClient = ctx.fake_client  # type: ignore[attr-defined]
    client.script += [
        LLMResponse("", (ToolCall("c1", "remember", '{"text": "x"}'),), "tool_calls"),
        LLMRateLimitError("429", user_message="slow down"),
    ]
    conv = ctx.store.create_conversation()
    with pytest.raises(LLMRateLimitError):
        ctx.chat.send_message(conv.id, "save x", AgentCallbacks())
    assert [m.role for m in ctx.store.get_messages(conv.id)] == [Role.USER, Role.ASSISTANT, Role.TOOL]


def test_title_from_text() -> None:
    assert title_from_text("short\nmessage") == "short message"
    long = title_from_text("word " * 50)
    assert len(long) == 60 and long.endswith("…")


def test_context_reports_recovery_notices(tmp_path: Path) -> None:
    paths = AppPaths.under(tmp_path / "home").ensure()
    paths.settings_file.write_text("{broken", encoding="utf-8")
    paths.database_file.write_bytes(b"garbage" * 1000)
    context = AppContext.create(
        paths, secrets=SecretStore(backend=MemoryKeyring(), env={}), client_factory=ScriptedChatClient, registry=ToolRegistry()
    )
    try:
        assert len(context.notices) == 2
        assert any("settings" in n for n in context.notices) and any("database" in n for n in context.notices)
    finally:
        context.close()


def test_default_registry_and_factory_are_wired(tmp_path: Path) -> None:
    keyring = MemoryKeyring()
    context = AppContext.create(AppPaths.under(tmp_path / "home"), secrets=SecretStore(backend=keyring, env={}))
    try:
        assert len(context.registry) > 0
        assert isinstance(context.client_factory, GroqClientFactory)
        with pytest.raises(MissingAPIKeyError):
            context.client_factory()
    finally:
        context.close()


def test_client_factory_caches_and_invalidates() -> None:
    secrets = SecretStore(backend=MemoryKeyring(), env={})
    settings = {"value": Settings()}
    factory = GroqClientFactory(secrets, lambda: settings["value"])
    with pytest.raises(MissingAPIKeyError):
        factory()
    secrets.set_api_key(KEY)
    first = factory()
    assert isinstance(first, GroqChatClient)
    assert factory() is first  # cached
    settings["value"] = Settings(request_timeout_seconds=99)
    second = factory()
    assert second is not first  # timeout change -> new client
    factory.invalidate()
    assert factory() is not second
    secrets.set_api_key(KEY.replace("Z", "Q"))
    assert factory() is not second
