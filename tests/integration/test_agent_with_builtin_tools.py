"""Integration: real ChatService + Agent + policy + built-in tools + SQLite.

Only the remote model is scripted; every tool call below really executes (or
is really refused) against the file system, the shell and the notes table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fakes import MemoryKeyring, ScriptedChatClient

from jarvis.agent.events import AgentCallbacks, ApprovalRequest
from jarvis.app_context import AppContext
from jarvis.config.secrets import SecretStore
from jarvis.llm.base import LLMResponse, Role, ToolCall
from jarvis.paths import AppPaths
from jarvis.tools.base import RiskLevel
from jarvis.tools.policy import Decision


class Host(AgentCallbacks):
    def __init__(self, approve: bool) -> None:
        self.approve = approve
        self.approvals: list[ApprovalRequest] = []
        self.decisions: list[tuple[str, Decision]] = []

    def on_tool_started(self, call: ToolCall, summary: str, decision: Decision) -> None:
        self.decisions.append((call.name, decision))

    def request_approval(self, request: ApprovalRequest) -> bool:
        self.approvals.append(request)
        return self.approve


@pytest.fixture
def app(tmp_path: Path) -> AppContext:
    client = ScriptedChatClient()
    context = AppContext.create(
        AppPaths.under(tmp_path / "home"),
        secrets=SecretStore(backend=MemoryKeyring(), env={}),
        client_factory=lambda: client,
    )
    workspace = tmp_path / "ws"
    context.settings.update(workspace_dir=str(workspace), shell_timeout_seconds=30)
    context.client = client  # type: ignore[attr-defined]
    context.workspace = workspace.resolve()  # type: ignore[attr-defined]
    yield context
    context.close()


def run(app: AppContext, approve: bool, *calls: tuple[str, dict[str, Any]]) -> tuple[Host, list]:
    tool_calls = tuple(ToolCall(f"c{i}", name, json.dumps(args)) for i, (name, args) in enumerate(calls))
    app.client.script += [LLMResponse("", tool_calls, "tool_calls"), LLMResponse("done", (), "stop")]  # type: ignore[attr-defined]
    conv = app.store.create_conversation()
    host = Host(approve)
    app.chat.send_message(conv.id, "do it", host)
    tool_results = [m.content for m in app.store.get_messages(conv.id) if m.role is Role.TOOL]
    return host, tool_results


def test_safe_reads_inside_workspace_run_without_asking(app: AppContext) -> None:
    ws: Path = app.workspace  # type: ignore[attr-defined]
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "hello.txt").write_text("hello from disk", encoding="utf-8")
    host, results = run(app, False, ("read_file", {"path": "hello.txt"}), ("calculate", {"expression": "6*7"}))
    assert host.approvals == []
    assert host.decisions == [("read_file", Decision.AUTO), ("calculate", Decision.AUTO)]
    assert "hello from disk" in results[0] and "42" in results[1]


def test_read_outside_workspace_requires_approval(app: AppContext, tmp_path: Path) -> None:
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("TOP SECRET", encoding="utf-8")
    host, results = run(app, False, ("read_file", {"path": str(secret)}))
    assert len(host.approvals) == 1 and host.approvals[0].risk is RiskLevel.CONSEQUENTIAL
    assert "TOP SECRET" not in results[0] and "declined" in results[0]


def test_write_file_approved_and_denied(app: AppContext) -> None:
    ws: Path = app.workspace  # type: ignore[attr-defined]
    host, results = run(app, False, ("write_file", {"path": "notes/plan.txt", "content": "v1"}))
    assert not (ws / "notes" / "plan.txt").exists()
    summary = host.approvals[0].summary
    assert str(ws / "notes" / "plan.txt") in summary, "approval dialog must show the absolute path"

    host, results = run(app, True, ("write_file", {"path": "notes/plan.txt", "content": "v1"}))
    assert (ws / "notes" / "plan.txt").read_text(encoding="utf-8") == "v1"
    assert not results[0].startswith("Error")


def test_write_outside_workspace_refused_even_if_approved(app: AppContext, tmp_path: Path) -> None:
    target = tmp_path / "escape.txt"
    host, results = run(app, True, ("write_file", {"path": "../escape.txt", "content": "x"}))
    assert not target.exists()
    assert results[0].startswith("Error")


def test_shell_command_requires_approval_and_runs_when_allowed(app: AppContext) -> None:
    command = "Write-Output jarvis-shell-ok" if sys.platform == "win32" else "echo jarvis-shell-ok"
    host, results = run(app, False, ("run_shell_command", {"command": command}))
    assert host.approvals[0].tool_name == "run_shell_command" and command in host.approvals[0].summary
    assert "jarvis-shell-ok" not in results[0]

    host, results = run(app, True, ("run_shell_command", {"command": command}))
    assert "jarvis-shell-ok" in results[0]


def test_notes_are_saved_automatically(app: AppContext) -> None:
    host, results = run(app, False, ("save_note", {"content": "Passport renewal in March", "tags": "admin"}))
    assert host.approvals == []
    assert [n.content for n in app.store.search_notes("passport")] == ["Passport renewal in March"]


def test_disabled_tool_is_refused(app: AppContext) -> None:
    app.settings.update(disabled_tools=["calculate"])
    host, results = run(app, True, ("calculate", {"expression": "1+1"}))
    assert host.decisions == [("calculate", Decision.DENY)]
    assert "disabled" in results[0]
    assert all(s["function"]["name"] != "calculate" for s in app.client.calls[-2].tools)  # type: ignore[attr-defined]
