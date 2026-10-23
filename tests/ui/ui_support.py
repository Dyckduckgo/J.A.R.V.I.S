"""Test tools, harness and helpers shared by the GUI tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fakes import MemoryKeyring, ScriptedChatClient
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QAbstractButton, QLabel, QLineEdit, QPlainTextEdit, QTextEdit, QWidget

from jarvis.app_context import AppContext
from jarvis.tools.base import RiskLevel, Tool, ToolContext, ToolResult

TEST_API_KEY = "gsk_" + "UiTestKey0123456789" * 2


class EchoTool(Tool):
    name = "echo_text"
    description = "Echo the given text back (read-only)."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls.append(args)
        return ToolResult.success(f"echo: {args['text']}")


class WriteMarkerTool(Tool):
    name = "write_marker"
    description = "Create a marker file in the workspace."
    parameters = {
        "type": "object",
        "properties": {"filename": {"type": "string"}, "content": {"type": "string", "default": "marker"}},
        "required": ["filename"],
        "additionalProperties": False,
    }
    risk = RiskLevel.CONSEQUENTIAL

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        where = ctx.workspace / args["filename"] if ctx is not None else args["filename"]
        return f"Create the file {where} with content {args['content']!r}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls.append(args)
        target = ctx.workspace / args["filename"]
        target.write_text(args["content"], encoding="utf-8")
        return ToolResult.success(f"Wrote {target.name}")


class CountingFactory:
    """client_factory with an ``invalidate`` hook, like the real Groq factory."""

    def __init__(self, client: ScriptedChatClient) -> None:
        self.client = client
        self.invalidations = 0

    def __call__(self) -> ScriptedChatClient:
        return self.client

    def invalidate(self) -> None:
        self.invalidations += 1


@dataclass
class Harness:
    ctx: AppContext
    client: ScriptedChatClient
    keyring: MemoryKeyring
    echo: EchoTool
    writer: WriteMarkerTool
    factory: Any

    @property
    def workspace(self) -> Path:
        return self.ctx.settings.settings.workspace_path

    def messages(self, conversation_id: int):
        return self.ctx.store.get_messages(conversation_id)


def send(qtbot, window, text: str) -> None:
    """Type ``text`` into the composer and press Enter."""
    window.input.setPlainText(text)
    window.input.setFocus()
    qtbot.keyClick(window.input, Qt.Key.Key_Return)


def wait_idle(qtbot, window, timeout: int = 8000) -> None:
    qtbot.waitUntil(lambda: not window.is_busy(), timeout=timeout)


def wait_for_approval(qtbot, window, timeout: int = 8000):
    qtbot.waitUntil(
        lambda: window.approval_dialog is not None and window.approval_dialog.isVisible(), timeout=timeout
    )
    return window.approval_dialog


def all_widget_texts(root: QWidget) -> list[str]:
    """Every user-visible string under ``root`` (labels, fields, buttons, tooltips, titles)."""
    texts: list[str] = [root.windowTitle(), root.toolTip()]
    for widget in [root, *root.findChildren(QWidget)]:
        texts += [widget.toolTip(), widget.statusTip(), widget.whatsThis(), widget.accessibleName()]
        if isinstance(widget, QLabel):
            texts.append(widget.text())
        elif isinstance(widget, QLineEdit):
            texts += [widget.text(), widget.displayText(), widget.placeholderText()]
        elif isinstance(widget, QAbstractButton):
            texts.append(widget.text())
        elif isinstance(widget, (QPlainTextEdit, QTextEdit)):
            texts.append(widget.toPlainText())
    return [t for t in texts if t]
