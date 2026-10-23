"""Approval policy: decides whether a tool call runs automatically, needs the
user's confirmation, or is refused.

The rules are intentionally not configurable beyond enabling/disabling tools:

* unknown or disabled tool                -> DENY
* call assessed as ``CONSEQUENTIAL``      -> CONFIRM (always)
* call assessed as ``SAFE``               -> AUTO

A tool's per-call assessment can only escalate its intrinsic risk; a
consequential tool can never be downgraded to automatic execution.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from jarvis.tools.base import RiskLevel, Tool, ToolContext


class Decision(str, Enum):
    AUTO = "auto"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    risk: RiskLevel
    reason: str


class ApprovalPolicy:
    def __init__(self, disabled_tools: Iterable[str] = ()) -> None:
        self._disabled = frozenset(disabled_tools)

    def evaluate(self, tool: Tool | None, args: dict[str, Any], ctx: ToolContext) -> PolicyResult:
        if tool is None:
            return PolicyResult(Decision.DENY, RiskLevel.CONSEQUENTIAL, "Unknown tool.")
        if tool.name in self._disabled:
            return PolicyResult(Decision.DENY, tool.risk, f"The tool '{tool.name}' is disabled in Settings.")
        try:
            assessed = tool.assess_risk(args, ctx)
        except Exception:  # a broken assessment must fail closed
            assessed = RiskLevel.CONSEQUENTIAL
        risk = RiskLevel.CONSEQUENTIAL if RiskLevel.CONSEQUENTIAL in (assessed, tool.risk) else RiskLevel.SAFE
        if risk is RiskLevel.CONSEQUENTIAL:
            reason = "This action can change your system, run programs, or send data over the network."
            if tool.risk is RiskLevel.SAFE:
                reason = "This call reaches outside the assistant's sandbox (for example, a path outside the workspace)."
            return PolicyResult(Decision.CONFIRM, risk, reason)
        return PolicyResult(Decision.AUTO, risk, "Read-only action inside the workspace.")
