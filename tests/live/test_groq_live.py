"""Opt-in tests against the real Groq API.

Run with:  GROQ_API_KEY=gsk_... python -m pytest -m live
They are skipped automatically when no key is available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.agent.agent import Agent
from jarvis.agent.events import AgentCallbacks
from jarvis.config.settings import DEFAULT_MODEL, Settings
from jarvis.llm.base import ChatMessage, Role
from jarvis.llm.groq_client import GroqChatClient
from jarvis.tools.builtin import create_default_registry

pytestmark = pytest.mark.live


def test_list_models(live_api_key: str) -> None:
    models = GroqChatClient(live_api_key).list_models()
    assert models, "account should have at least one chat model"
    assert not any("whisper" in m for m in models)


def test_streaming_completion(live_api_key: str) -> None:
    chunks: list[str] = []
    result = GroqChatClient(live_api_key).complete(
        [ChatMessage.user("Reply with exactly the single word: pong")],
        model=DEFAULT_MODEL,
        temperature=0,
        max_tokens=16,
        on_text=chunks.append,
    )
    assert "pong" in result.content.lower()
    assert "".join(chunks) == result.content


def test_agent_uses_safe_tool_end_to_end(live_api_key: str, tmp_path: Path) -> None:
    client = GroqChatClient(live_api_key)
    settings = Settings(workspace_dir=str(tmp_path), temperature=0)
    agent = Agent(lambda: client, create_default_registry(), lambda: settings)
    result = agent.run_turn([], "Use the calculate tool to compute 1234 * 5678, then tell me the result.", AgentCallbacks())
    tool_messages = [m for m in result.new_messages if m.role is Role.TOOL]
    assert tool_messages, "the model should have called a tool"
    assert "7006652" in result.final_text.replace(",", "").replace(" ", "")
