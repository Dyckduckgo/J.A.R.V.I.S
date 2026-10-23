"""Agent loop tests: real Agent, real policy, real tools defined here; only the
remote model is replaced by ScriptedChatClient."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedChatClient

from jarvis.agent.agent import (
    CANCELLED_MESSAGE,
    DECLINED_MESSAGE,
    ERROR_PREFIX,
    MAX_CONTEXT_CHARS,
    OLD_TOOL_RESULT_CHARS,
    Agent,
    prepare_history,
)
from jarvis.agent.events import AgentCallbacks, ApprovalRequest
from jarvis.config.settings import Settings
from jarvis.errors import LLMRateLimitError, LLMToolCallError, MissingAPIKeyError, ToolError
from jarvis.llm.base import ChatMessage, LLMResponse, Role, ToolCall
from jarvis.tools.base import RiskLevel, Tool, ToolContext, ToolResult
from jarvis.tools.policy import Decision
from jarvis.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Echo text back."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.runs.append(args)
        return ToolResult.success(f"echo: {args['text']}")


class WriteMarkerTool(Tool):
    """Consequential tool with a real side effect (creates a file)."""

    name = "write_marker"
    description = "Create marker file."
    parameters = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False}
    risk = RiskLevel.CONSEQUENTIAL

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        (ctx.workspace / args["name"]).write_text("written")
        return ToolResult.success("created")


class EscalatingTool(EchoTool):
    name = "peek"

    def assess_risk(self, args: dict[str, Any], ctx: ToolContext) -> RiskLevel:
        return RiskLevel.CONSEQUENTIAL if args["text"].startswith("/") else RiskLevel.SAFE


class FailingTool(Tool):
    name = "fails"
    description = "Always fails."
    parameters = {"type": "object", "properties": {"kind": {"type": "string"}}, "required": ["kind"]}

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if args["kind"] == "expected":
            raise ToolError("disk is full")
        raise RuntimeError("kaboom")


class Recorder(AgentCallbacks):
    def __init__(self, approve: bool | None = None, on_approval: Any = None) -> None:
        self.events: list[tuple] = []
        self.approve = approve
        self.on_approval = on_approval
        self.approvals: list[ApprovalRequest] = []

    def on_message(self, message: ChatMessage) -> None:
        self.events.append(("message", message.role.value, message.content))

    def on_llm_started(self, iteration: int) -> None:
        self.events.append(("llm", iteration))

    def on_text_delta(self, text: str) -> None:
        self.events.append(("delta", text))

    def on_tool_started(self, call: ToolCall, summary: str, decision: Decision) -> None:
        self.events.append(("tool_started", call.name, decision))

    def on_tool_finished(self, call: ToolCall, result: ToolResult) -> None:
        self.events.append(("tool_finished", call.name, result.ok))

    def request_approval(self, request: ApprovalRequest) -> bool:
        self.approvals.append(request)
        if self.on_approval:
            return self.on_approval(request)
        if self.approve is None:
            return super().request_approval(request)
        return self.approve

    @property
    def messages(self) -> list[tuple]:
        return [e for e in self.events if e[0] == "message"]


def call(name: str, args: str, cid: str = "call_1") -> ToolCall:
    return ToolCall(cid, name, args)


def tool_reply(*calls: ToolCall, content: str = "") -> LLMResponse:
    return LLMResponse(content, tuple(calls), "tool_calls")


def text_reply(text: str) -> LLMResponse:
    return LLMResponse(text, (), "stop")


@pytest.fixture
def echo() -> EchoTool:
    return EchoTool()


def build(workspace: Path, script: list, *tools: Tool, **settings: Any) -> tuple[Agent, ScriptedChatClient]:
    client = ScriptedChatClient(script=list(script))
    config = Settings(workspace_dir=str(workspace), **settings)
    agent = Agent(lambda: client, ToolRegistry(tools), lambda: config)
    return agent, client


def test_plain_answer_streams_and_emits_messages(workspace: Path, echo: EchoTool) -> None:
    agent, client = build(workspace, [text_reply("Hello there")], echo, custom_instructions="Speak like a butler.")
    rec = Recorder()
    result = agent.run_turn([], "hi", rec)

    assert result.final_text == "Hello there" and result.iterations == 1 and not result.cancelled
    assert "".join(e[1] for e in rec.events if e[0] == "delta") == "Hello there"
    assert rec.messages == [("message", "user", "hi"), ("message", "assistant", "Hello there")]
    sent = client.calls[0]
    assert sent.messages[0].role is Role.SYSTEM
    assert "Speak like a butler." in sent.messages[0].content and str(workspace) in sent.messages[0].content
    assert sent.tools and sent.tools[0]["function"]["name"] == "echo"


def test_safe_tool_runs_automatically(workspace: Path, echo: EchoTool) -> None:
    agent, client = build(workspace, [tool_reply(call("echo", '{"text": "ping"}')), text_reply("Done")], echo)
    rec = Recorder()
    result = agent.run_turn([], "echo ping", rec)

    assert echo.runs == [{"text": "ping"}]
    assert rec.approvals == []
    assert ("tool_started", "echo", Decision.AUTO) in rec.events
    assert ("tool_finished", "echo", True) in rec.events
    second_request = client.calls[1].messages
    assert second_request[-1].role is Role.TOOL and second_request[-1].content == "echo: ping"
    assert second_request[-1].tool_call_id == "call_1"
    assert [m.role for m in result.new_messages] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]


def test_consequential_tool_runs_only_after_approval(workspace: Path) -> None:
    agent, _ = build(workspace, [tool_reply(call("write_marker", '{"name": "ok.txt"}')), text_reply("Created")], WriteMarkerTool())
    rec = Recorder(approve=True)
    agent.run_turn([], "make it", rec)
    assert len(rec.approvals) == 1
    request = rec.approvals[0]
    assert request.tool_name == "write_marker" and request.risk is RiskLevel.CONSEQUENTIAL
    assert request.arguments == {"name": "ok.txt"} and "ok.txt" in request.summary
    assert (workspace / "ok.txt").read_text() == "written"


def test_denied_consequential_tool_never_runs(workspace: Path) -> None:
    agent, client = build(workspace, [tool_reply(call("write_marker", '{"name": "no.txt"}')), text_reply("Okay, I won't.")], WriteMarkerTool())
    rec = Recorder(approve=False)
    result = agent.run_turn([], "make it", rec)
    assert not (workspace / "no.txt").exists()
    tool_msg = result.new_messages[2]
    assert tool_msg.role is Role.TOOL and tool_msg.content == ERROR_PREFIX + DECLINED_MESSAGE
    assert ("tool_finished", "write_marker", False) in rec.events
    assert client.calls[1].messages[-1].content == ERROR_PREFIX + DECLINED_MESSAGE


def test_host_without_approval_handler_denies_by_default(workspace: Path) -> None:
    agent, _ = build(workspace, [tool_reply(call("write_marker", '{"name": "x.txt"}')), text_reply("ok")], WriteMarkerTool())
    agent.run_turn([], "go", AgentCallbacks())
    assert not (workspace / "x.txt").exists()


def test_approval_callback_exception_is_treated_as_denial(workspace: Path) -> None:
    def explode(request: ApprovalRequest) -> bool:
        raise RuntimeError("dialog crashed")

    agent, _ = build(workspace, [tool_reply(call("write_marker", '{"name": "x.txt"}')), text_reply("ok")], WriteMarkerTool())
    agent.run_turn([], "go", Recorder(on_approval=explode))
    assert not (workspace / "x.txt").exists()


def test_truthy_non_bool_approval_is_not_accepted(workspace: Path) -> None:
    agent, _ = build(workspace, [tool_reply(call("write_marker", '{"name": "x.txt"}')), text_reply("ok")], WriteMarkerTool())
    agent.run_turn([], "go", Recorder(on_approval=lambda r: "yes"))
    assert not (workspace / "x.txt").exists()


def test_per_call_risk_escalation_requires_confirmation(workspace: Path) -> None:
    tool = EscalatingTool()
    agent, _ = build(
        workspace,
        [tool_reply(call("peek", '{"text": "inside"}', "c1"), call("peek", '{"text": "/etc/passwd"}', "c2")), text_reply("ok")],
        tool,
    )
    rec = Recorder(approve=False)
    agent.run_turn([], "peek", rec)
    assert tool.runs == [{"text": "inside"}]
    assert [r.arguments["text"] for r in rec.approvals] == ["/etc/passwd"]
    assert "outside" in rec.approvals[0].reason


def test_disabled_tool_is_hidden_and_refused(workspace: Path, echo: EchoTool) -> None:
    agent, client = build(workspace, [tool_reply(call("echo", '{"text": "x"}')), text_reply("ok")], echo, disabled_tools=["echo"])
    rec = Recorder()
    result = agent.run_turn([], "go", rec)
    assert client.calls[0].tools is None  # no enabled tools -> no schemas sent
    assert echo.runs == []
    assert ("tool_started", "echo", Decision.DENY) in rec.events
    assert "disabled" in result.new_messages[2].content


@pytest.mark.parametrize(
    ("bad_call", "fragment"),
    [
        (call("nope", "{}"), "Unknown tool 'nope'"),
        (call("echo", "{not json"), "not valid JSON"),
        (call("echo", "[1, 2]"), "must be a JSON object"),
        (call("echo", '{"text": 5}'), "must be of type string"),
        (call("echo", '{"text": "a", "extra": 1}'), "Unknown argument"),
        (call("echo", "{}"), "Missing required argument 'text'"),
    ],
)
def test_bad_tool_calls_become_error_results(workspace: Path, echo: EchoTool, bad_call: ToolCall, fragment: str) -> None:
    agent, _ = build(workspace, [tool_reply(bad_call), text_reply("sorry")], echo)
    result = agent.run_turn([], "go", Recorder())
    tool_msg = result.new_messages[2]
    assert tool_msg.role is Role.TOOL and tool_msg.content.startswith(ERROR_PREFIX)
    assert fragment in tool_msg.content
    assert echo.runs == []
    assert result.final_text == "sorry"


def test_tool_exceptions_are_contained(workspace: Path) -> None:
    agent, _ = build(
        workspace,
        [tool_reply(call("fails", '{"kind": "expected"}', "a"), call("fails", '{"kind": "bug"}', "b")), text_reply("ok")],
        FailingTool(),
    )
    result = agent.run_turn([], "go", Recorder())
    contents = [m.content for m in result.new_messages if m.role is Role.TOOL]
    assert contents[0] == ERROR_PREFIX + "disk is full"
    assert "RuntimeError" in contents[1] and "kaboom" in contents[1]


def test_iteration_limit_stops_loop_with_valid_history(workspace: Path, echo: EchoTool) -> None:
    script = [tool_reply(call("echo", '{"text": "again"}', f"c{i}")) for i in range(3)]
    agent, client = build(workspace, script, echo, max_tool_iterations=3)
    result = agent.run_turn([], "loop forever", Recorder())
    assert result.stopped_at_iteration_limit and len(client.calls) == 3
    assert "stopped after 3 tool steps" in result.final_text
    assert_history_valid(list(result.new_messages))


def test_cancel_while_waiting_for_approval(workspace: Path) -> None:
    cancel = threading.Event()

    def approve_then_cancel(request: ApprovalRequest) -> bool:
        cancel.set()  # user pressed Stop while the dialog was open
        return True

    agent, client = build(
        workspace,
        [tool_reply(call("write_marker", '{"name": "a.txt"}', "c1"), call("write_marker", '{"name": "b.txt"}', "c2"))],
        WriteMarkerTool(),
    )
    result = agent.run_turn([], "go", Recorder(on_approval=approve_then_cancel), cancel)
    assert result.cancelled
    assert not (workspace / "a.txt").exists() and not (workspace / "b.txt").exists()
    tool_contents = [m.content for m in result.new_messages if m.role is Role.TOOL]
    assert tool_contents == [ERROR_PREFIX + CANCELLED_MESSAGE] * 2
    assert len(client.calls) == 1
    assert_history_valid(list(result.new_messages))


def test_cancel_during_streaming_keeps_partial_text(workspace: Path, echo: EchoTool) -> None:
    cancel = threading.Event()

    class CancellingRecorder(Recorder):
        def on_text_delta(self, text: str) -> None:
            super().on_text_delta(text)
            cancel.set()

    agent, _ = build(workspace, [text_reply("A long answer that gets interrupted")], echo)
    result = agent.run_turn([], "talk", CancellingRecorder(), cancel)
    assert result.cancelled
    last = result.new_messages[-1]
    assert last.role is Role.ASSISTANT and last.content.startswith("A lo") and "[stopped]" in last.content


def test_malformed_tool_call_is_retried_once(workspace: Path, echo: EchoTool) -> None:
    agent, client = build(workspace, [LLMToolCallError("bad"), text_reply("recovered")], echo)
    assert agent.run_turn([], "go", Recorder()).final_text == "recovered"
    assert len(client.calls) == 2

    agent, _ = build(workspace, [LLMToolCallError("bad"), LLMToolCallError("bad again")], echo)
    with pytest.raises(LLMToolCallError):
        agent.run_turn([], "go", Recorder())


def test_llm_error_propagates_after_user_message_recorded(workspace: Path, echo: EchoTool) -> None:
    agent, _ = build(workspace, [LLMRateLimitError("429", user_message="Rate limited")], echo)
    rec = Recorder()
    with pytest.raises(LLMRateLimitError):
        agent.run_turn([], "hello", rec)
    assert rec.messages == [("message", "user", "hello")]


def test_missing_key_fails_before_anything_is_recorded(workspace: Path, echo: EchoTool) -> None:
    def no_client():
        raise MissingAPIKeyError()

    agent = Agent(no_client, ToolRegistry([echo]), lambda: Settings(workspace_dir=str(workspace)))
    rec = Recorder()
    with pytest.raises(MissingAPIKeyError):
        agent.run_turn([], "hello", rec)
    assert rec.events == []


def test_workspace_is_created(tmp_path: Path, echo: EchoTool) -> None:
    target = tmp_path / "new" / "workspace"
    agent, _ = build(target, [text_reply("hi")], echo)
    agent.run_turn([], "hi", Recorder())
    assert target.is_dir()


def test_history_is_sent_to_model(workspace: Path, echo: EchoTool) -> None:
    history = [ChatMessage.user("my name is Ada"), ChatMessage.assistant("Nice to meet you, Ada.")]
    agent, client = build(workspace, [text_reply("Ada")], echo)
    agent.run_turn(history, "what is my name?", Recorder())
    sent = [(m.role, m.content) for m in client.calls[0].messages[1:]]
    assert sent == [
        (Role.USER, "my name is Ada"),
        (Role.ASSISTANT, "Nice to meet you, Ada."),
        (Role.USER, "what is my name?"),
    ]


# --- prepare_history ---------------------------------------------------------


def assert_history_valid(messages: list[ChatMessage]) -> None:
    """Every assistant tool call is answered by exactly one following tool message."""
    pending: list[str] = []
    for m in messages:
        if m.role is Role.TOOL:
            assert m.tool_call_id in pending, f"orphan tool result {m.tool_call_id}"
            pending.remove(m.tool_call_id)
        else:
            assert not pending, f"unanswered tool calls {pending}"
            if m.role is Role.ASSISTANT:
                pending = [c.id for c in m.tool_calls]
    assert not pending


def turn(i: int, tool_output: str = "result") -> list[ChatMessage]:
    c = ToolCall(f"c{i}", "echo", "{}")
    return [ChatMessage.user(f"q{i}"), ChatMessage.assistant(None, [c]), ChatMessage.tool(c, tool_output), ChatMessage.assistant(f"a{i}")]


def test_prepare_history_starts_at_user_message() -> None:
    history = [ChatMessage.system("old system")] + turn(1) + turn(2)
    trimmed = prepare_history(history, max_messages=6)  # would start mid-turn 1
    assert trimmed[0].role is Role.USER and trimmed[0].content == "q2"
    assert all(m.role is not Role.SYSTEM for m in trimmed)
    assert_history_valid(trimmed)


def test_prepare_history_compacts_old_tool_output() -> None:
    trimmed = prepare_history(turn(1, "x" * (OLD_TOOL_RESULT_CHARS * 3)), max_messages=50)
    assert len(trimmed[2].content) < OLD_TOOL_RESULT_CHARS + 100
    assert "truncated" in trimmed[2].content


def test_prepare_history_respects_char_budget() -> None:
    big = "y" * (MAX_CONTEXT_CHARS // 3)
    history = []
    for i in range(6):
        history += [ChatMessage.user(f"q{i} {big}"), ChatMessage.assistant(f"a{i}")]
    trimmed = prepare_history(history, max_messages=500)
    assert sum(len(m.content or "") for m in trimmed) <= MAX_CONTEXT_CHARS
    assert trimmed[0].role is Role.USER and trimmed[-1].content == "a5"


def test_prepare_history_without_user_message_is_empty() -> None:
    c = ToolCall("c", "echo", "{}")
    assert prepare_history([ChatMessage.tool(c, "orphan"), ChatMessage.assistant("x")], 10) == []
