"""Registry, schemas and approval-policy integration for all built-in tools."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jarvis.errors import ToolArgumentError
from jarvis.tools.base import RiskLevel, ToolContext
from jarvis.tools.builtin import builtin_tools, create_default_registry
from jarvis.tools.policy import ApprovalPolicy, Decision
from jarvis.tools.registry import ToolRegistry

SAFE = RiskLevel.SAFE
CONSEQUENTIAL = RiskLevel.CONSEQUENTIAL

EXPECTED_RISK = {
    "get_datetime": SAFE,
    "calculate": SAFE,
    "system_info": SAFE,
    "list_directory": SAFE,
    "read_file": SAFE,
    "search_files": SAFE,
    "write_file": CONSEQUENTIAL,
    "run_shell_command": CONSEQUENTIAL,
    "fetch_url": CONSEQUENTIAL,
    "open_target": CONSEQUENTIAL,
    "save_note": SAFE,
    "search_notes": SAFE,
}

JSON_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def test_default_registry_has_every_tool_once() -> None:
    registry = create_default_registry()
    assert isinstance(registry, ToolRegistry)
    assert set(registry.names()) == set(EXPECTED_RISK)
    assert len(registry) == len(EXPECTED_RISK)
    names = [tool.name for tool in builtin_tools()]
    assert len(names) == len(set(names))


def test_builtin_tools_returns_fresh_instances() -> None:
    first, second = builtin_tools(), builtin_tools()
    assert [t.name for t in first] == [t.name for t in second]
    assert all(a is not b for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("tool", builtin_tools(), ids=lambda t: t.name)
def test_schema_is_a_valid_function_schema(tool) -> None:
    schema = tool.schema()
    json.dumps(schema)  # serialisable
    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == tool.name
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", function["name"])
    assert isinstance(function["description"], str) and 20 <= len(function["description"]) <= 1024

    params = function["parameters"]
    assert params["type"] == "object"
    assert params["additionalProperties"] is False
    properties = params["properties"]
    assert isinstance(properties, dict)
    required = params.get("required", [])
    assert isinstance(required, list)
    assert set(required) <= set(properties)
    for name, spec in properties.items():
        assert spec.get("type") in JSON_TYPES, name
        assert spec.get("description"), f"{tool.name}.{name} has no description"
        if "enum" in spec:
            assert all(isinstance(v, JSON_TYPES[spec["type"]]) for v in spec["enum"])
        if "default" in spec:
            assert name not in required, f"{tool.name}.{name} is required but has a default"
            assert isinstance(spec["default"], JSON_TYPES[spec["type"]])
            if "enum" in spec:
                assert spec["default"] in spec["enum"]
            if "minimum" in spec:
                assert spec["default"] >= spec["minimum"]
            if "maximum" in spec:
                assert spec["default"] <= spec["maximum"]


@pytest.mark.parametrize("tool", builtin_tools(), ids=lambda t: t.name)
def test_intrinsic_risk_levels(tool) -> None:
    assert tool.risk is EXPECTED_RISK[tool.name]


@pytest.mark.parametrize("tool", builtin_tools(), ids=lambda t: t.name)
def test_unknown_arguments_are_rejected(tool) -> None:
    with pytest.raises(ToolArgumentError, match="Unknown argument"):
        tool.validate({"definitely_not_an_argument": 1})


@pytest.mark.parametrize("tool", builtin_tools(), ids=lambda t: t.name)
def test_non_object_arguments_are_rejected(tool) -> None:
    with pytest.raises(ToolArgumentError):
        tool.validate(["not", "an", "object"])  # type: ignore[arg-type]


SAFE_CALLS = {
    "get_datetime": {},
    "calculate": {"expression": "1 + 1"},
    "system_info": {},
    "list_directory": {},
    "read_file": {"path": "notes.txt"},
    "search_files": {"pattern": "*.txt"},
    "save_note": {"content": "remember the milk"},
    "search_notes": {"query": "milk"},
}

CONSEQUENTIAL_CALLS = {
    "write_file": {"path": "new.txt", "content": "hello"},
    "run_shell_command": {"command": "echo hello"},
    "fetch_url": {"url": "https://example.com/"},
    "open_target": {"target": "https://example.com/"},
}


def test_policy_runs_safe_in_workspace_calls_automatically(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "notes.txt").write_text("hi", encoding="utf-8")
    registry = create_default_registry()
    policy = ApprovalPolicy()
    assert set(SAFE_CALLS) | set(CONSEQUENTIAL_CALLS) == set(EXPECTED_RISK)
    for name, raw in SAFE_CALLS.items():
        tool = registry.get(name)
        result = policy.evaluate(tool, tool.validate(raw), ctx)
        assert result.decision is Decision.AUTO, name
        assert result.risk is SAFE


@pytest.mark.parametrize("name", sorted(CONSEQUENTIAL_CALLS))
def test_policy_requires_confirmation_for_every_consequential_tool(name: str, ctx: ToolContext) -> None:
    tool = create_default_registry().get(name)
    args = tool.validate(CONSEQUENTIAL_CALLS[name])
    assert tool.assess_risk(args, ctx) is CONSEQUENTIAL
    result = ApprovalPolicy().evaluate(tool, args, ctx)
    assert result.decision is Decision.CONFIRM
    assert result.risk is CONSEQUENTIAL


def test_policy_denies_disabled_builtin_tool(ctx: ToolContext) -> None:
    tool = create_default_registry().get("run_shell_command")
    result = ApprovalPolicy(disabled_tools={"run_shell_command"}).evaluate(tool, {"command": "echo hi"}, ctx)
    assert result.decision is Decision.DENY


@pytest.mark.parametrize("tool", builtin_tools(), ids=lambda t: t.name)
def test_describe_call_never_raises_on_odd_arguments(tool) -> None:
    # The GUI must always be able to render something, even for garbage.
    for args in ({}, {"path": 5, "content": None, "url": [], "target": None, "command": None}):
        assert isinstance(tool.describe_call(args), str)
