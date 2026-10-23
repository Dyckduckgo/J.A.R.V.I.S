"""Confirmation-dialog text must not be able to hide or reorder what will happen."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.tools.base import ToolContext
from jarvis.tools.builtin._display import reveal_hidden
from jarvis.tools.builtin.files import WriteFileTool
from jarvis.tools.builtin.launcher import OpenTargetTool
from jarvis.tools.builtin.shell import RunShellCommandTool
from jarvis.tools.builtin.web import FetchUrlTool

RLO = "\u202e"  # right-to-left override
ZWSP = "\u200b"  # zero-width space


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain ascii", "plain ascii"),
        ("caf\u00e9 \u65e5\u672c", "caf\u00e9 \u65e5\u672c"),
        ("line 1\n\tline 2", "line 1\n\tline 2"),
        (f"evil{RLO}txt.exe", "evil<U+202E>txt.exe"),
        (f"rm{ZWSP} -rf", "rm<U+200B> -rf"),
        ("a\rb\x1b[2Kc", "a<U+000D>b<U+001B>[2Kc"),
        ("x\u2028y", "x<U+2028>y"),
    ],
)
def test_reveal_hidden(text: str, expected: str) -> None:
    assert reveal_hidden(text) == expected


def test_shell_command_with_bidi_override_is_revealed(ctx: ToolContext) -> None:
    tool = RunShellCommandTool()
    command = f"echo safe {RLO}# del /s C:\\"
    text = tool.describe_call(tool.validate({"command": command}), ctx)
    assert RLO not in text and "<U+202E>" in text


def test_write_file_path_with_bidi_override_is_revealed(workspace: Path, ctx: ToolContext) -> None:
    tool = WriteFileTool()
    text = tool.describe_call(tool.validate({"path": f"report{RLO}fdp.exe", "content": "x"}), ctx)
    assert RLO not in text and "report<U+202E>fdp.exe" in text


def test_url_and_open_target_with_hidden_characters_are_revealed(ctx: ToolContext) -> None:
    url = f"https://example.com/{ZWSP}login"
    fetch = FetchUrlTool()
    assert "<U+200B>" in fetch.describe_call(fetch.validate({"url": url}))
    opener = OpenTargetTool()
    assert "<U+200B>" in opener.describe_call(opener.validate({"target": url}), ctx)
