from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from jarvis.errors import ToolArgumentError
from jarvis.llm.base import ChatMessage, Role, TokenUsage, ToolCall
from jarvis.tools.base import RiskLevel, Tool, ToolContext, ToolResult, truncate, validate_arguments
from jarvis.tools.policy import ApprovalPolicy, Decision
from jarvis.tools.registry import ToolRegistry

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "minLength": 1, "maxLength": 10},
        "count": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
        "ratio": {"type": "number"},
        "flag": {"type": "boolean"},
        "mode": {"type": "string", "enum": ["a", "b"]},
        "items": {"type": "array"},
    },
    "required": ["path"],
    "additionalProperties": False,
}


def test_validate_applies_defaults_and_accepts_valid() -> None:
    out = validate_arguments(SCHEMA, {"path": "x", "ratio": 1, "flag": True, "mode": "a", "items": []})
    assert out["count"] == 2 and out["ratio"] == 1


def test_validate_coerces_integral_float() -> None:
    assert validate_arguments(SCHEMA, {"path": "x", "count": 3.0})["count"] == 3


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"path": None},
        {"path": 1},
        {"path": ""},
        {"path": "x" * 11},
        {"path": "x", "count": 0},
        {"path": "x", "count": 6},
        {"path": "x", "count": 2.5},
        {"path": "x", "count": True},
        {"path": "x", "ratio": "1"},
        {"path": "x", "ratio": float("nan")},
        {"path": "x", "ratio": float("inf")},
        {"path": "x", "flag": "yes"},
        {"path": "x", "mode": "c"},
        {"path": "x", "items": "nope"},
        {"path": "x", "surprise": 1},
        [],
    ],
)
def test_validate_rejects_bad_arguments(args: Any) -> None:
    with pytest.raises(ToolArgumentError):
        validate_arguments(SCHEMA, args)


def test_truncate() -> None:
    assert truncate("abc", 5) == "abc"
    out = truncate("x" * 30, 10)
    assert out.startswith("x" * 10) and "truncated 20 characters" in out


class Safe(Tool):
    name = "safe_tool"
    description = "d"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return ToolResult.success("ok")


class Risky(Safe):
    name = "risky_tool"
    risk = RiskLevel.CONSEQUENTIAL


class TriesToDowngrade(Risky):
    name = "sneaky"

    def assess_risk(self, args: dict[str, Any], ctx: ToolContext) -> RiskLevel:
        return RiskLevel.SAFE


class BrokenAssessment(Safe):
    name = "broken"

    def assess_risk(self, args: dict[str, Any], ctx: ToolContext) -> RiskLevel:
        raise RuntimeError("bug")


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


def test_policy_decisions(ctx: ToolContext) -> None:
    policy = ApprovalPolicy(disabled_tools=["risky_tool"])
    assert policy.evaluate(Safe(), {}, ctx).decision is Decision.AUTO
    assert policy.evaluate(None, {}, ctx).decision is Decision.DENY
    assert policy.evaluate(Risky(), {}, ctx).decision is Decision.DENY  # disabled
    assert ApprovalPolicy().evaluate(Risky(), {}, ctx).decision is Decision.CONFIRM


def test_policy_never_downgrades_and_fails_closed(ctx: ToolContext) -> None:
    policy = ApprovalPolicy()
    assert policy.evaluate(TriesToDowngrade(), {}, ctx).decision is Decision.CONFIRM
    result = policy.evaluate(BrokenAssessment(), {}, ctx)
    assert result.decision is Decision.CONFIRM and result.risk is RiskLevel.CONSEQUENTIAL


def test_registry() -> None:
    registry = ToolRegistry([Safe(), Risky()])
    assert registry.names() == ["risky_tool", "safe_tool"] and "safe_tool" in registry and len(registry) == 2
    assert [s["function"]["name"] for s in registry.schemas(disabled=["risky_tool"])] == ["safe_tool"]
    with pytest.raises(ValueError):
        registry.register(Safe())

    class BadName(Safe):
        name = "has space"

    with pytest.raises(ValueError):
        registry.register(BadName())


def test_default_describe_call() -> None:
    assert Safe().describe_call({}) == "safe_tool()"
    assert Safe().describe_call({"a": "b" * 100}).endswith("...)")


def test_message_wire_format() -> None:
    c = ToolCall("id1", "safe_tool", "")
    assert c.parsed_arguments() == {}
    assert ChatMessage.user("hi").to_api() == {"role": "user", "content": "hi"}
    assert ChatMessage.assistant("text").to_api() == {"role": "assistant", "content": "text"}
    wire = ChatMessage.assistant(None, [c]).to_api()
    assert wire["content"] is None and wire["tool_calls"][0]["function"]["arguments"] == "{}"
    assert ChatMessage.tool(c, "out").to_api() == {"role": "tool", "content": "out", "tool_call_id": "id1", "name": "safe_tool"}
    assert ToolCall.from_dict(c.to_dict()).arguments == "{}"
    assert ChatMessage.system("s").role is Role.SYSTEM
    assert TokenUsage(1, 2, 3) + TokenUsage(1, 1, 2) == TokenUsage(2, 3, 5)
