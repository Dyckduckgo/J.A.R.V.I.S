"""Callback interface between the agent loop and its host (GUI or tests).

All callbacks are invoked on the thread running the agent. GUI hosts must
marshal them onto the UI thread (see :mod:`jarvis.ui.worker`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jarvis.llm.base import ChatMessage, TokenUsage, ToolCall
from jarvis.tools.base import RiskLevel, ToolResult
from jarvis.tools.policy import Decision


@dataclass(frozen=True)
class ApprovalRequest:
    """Shown to the user before a consequential tool call executes."""

    call: ToolCall
    tool_name: str
    summary: str
    risk: RiskLevel
    reason: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnResult:
    final_text: str
    new_messages: tuple[ChatMessage, ...]
    iterations: int
    cancelled: bool = False
    stopped_at_iteration_limit: bool = False
    usage: TokenUsage = TokenUsage()


class AgentCallbacks:
    """Base class with no-op defaults; override what you need.

    ``request_approval`` defaults to *deny* so that a host which forgets to
    implement it can never execute a consequential action by accident.
    """

    def on_message(self, message: ChatMessage) -> None:
        """A new message (user, assistant or tool) became part of the conversation."""

    def on_llm_started(self, iteration: int) -> None:
        """A model request is about to be sent (1-based iteration)."""

    def on_text_delta(self, text: str) -> None:
        """A chunk of streamed assistant text."""

    def on_tool_started(self, call: ToolCall, summary: str, decision: Decision) -> None:
        """A tool call was evaluated by the policy and is about to run (or be refused)."""

    def on_tool_finished(self, call: ToolCall, result: ToolResult) -> None:
        """A tool call completed, failed, was denied or was cancelled."""

    def request_approval(self, request: ApprovalRequest) -> bool:
        """Block until the user approves (True) or denies (False) the call."""
        return False
