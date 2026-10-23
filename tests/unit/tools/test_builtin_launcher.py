"""open_target with a recording launcher: nothing is ever really opened."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from builtin_helpers import call

import jarvis.tools.builtin.launcher as launcher_module
from jarvis.errors import ToolArgumentError, ToolError
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.launcher import BLOCKED_SUFFIXES, OpenTargetTool


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace the OS launcher for every OpenTargetTool in the test."""
    calls: list[str] = []
    monkeypatch.setattr(launcher_module, "launch_target", calls.append)
    return calls


@pytest.mark.parametrize("url", ["https://example.com/docs?q=1#top", "http://127.0.0.1:8000/"])
def test_opens_http_urls_in_the_browser(url: str, opened: list[str], ctx: ToolContext) -> None:
    result = call(OpenTargetTool(), {"target": url}, ctx)
    assert result.ok
    assert opened == [url]
    assert "default web browser" in result.content


def test_opens_existing_file_and_folder_by_resolved_path(workspace: Path, opened: list[str], ctx: ToolContext) -> None:
    (workspace / "docs").mkdir()
    (workspace / "docs" / "report.pdf").write_bytes(b"%PDF-1.4")
    file_result = call(OpenTargetTool(), {"target": "docs/report.pdf"}, ctx)
    folder_result = call(OpenTargetTool(), {"target": str(workspace / "docs")}, ctx)
    assert file_result.ok and folder_result.ok
    assert opened == [str((workspace / "docs" / "report.pdf").resolve()), str((workspace / "docs").resolve())]
    assert folder_result.content.startswith("Opened folder ")


def test_explicit_launcher_argument_is_used(workspace: Path, ctx: ToolContext) -> None:
    calls: list[str] = []
    (workspace / "a.txt").write_text("x", encoding="utf-8")
    call(OpenTargetTool(launcher=calls.append), {"target": "a.txt"}, ctx)
    assert calls == [str((workspace / "a.txt").resolve())]


def test_missing_path_is_an_error(opened: list[str], ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="Nothing exists"):
        call(OpenTargetTool(), {"target": "nope.txt"}, ctx)
    with pytest.raises(ToolError, match="include http:// or https://"):
        call(OpenTargetTool(), {"target": "www.example.com"}, ctx)
    assert opened == []


@pytest.mark.parametrize(
    "name",
    ["setup.exe", "RUN.BAT", "build.cmd", "deploy.ps1", "macro.vbs", "x.js", "installer.msi", "shortcut.lnk",
     "tweak.reg", "app.hta", "script.py", "tool.jar", "run.sh", "invoice.pdf.exe", "page.url", "panel.cpl", "x.scr"],
)  # fmt: skip
def test_executables_and_scripts_are_refused(name: str, workspace: Path, opened: list[str], ctx: ToolContext) -> None:
    (workspace / name).write_bytes(b"payload")
    with pytest.raises(ToolError, match="Refusing to open"):
        call(OpenTargetTool(), {"target": name}, ctx)
    assert opened == []


def test_blocklist_contains_the_required_suffixes() -> None:
    required = ".exe .bat .cmd .com .ps1 .vbs .js .jse .wsf .wsh .msi .msc .scr .lnk .reg .hta .cpl .jar .sh".split()
    assert set(required) <= BLOCKED_SUFFIXES


def test_symlink_disguising_an_executable_is_refused(workspace: Path, opened: list[str], ctx: ToolContext) -> None:
    from builtin_helpers import make_symlink

    (workspace / "evil.exe").write_bytes(b"MZ")
    make_symlink(workspace / "harmless.txt", workspace / "evil.exe")
    with pytest.raises(ToolError, match="Refusing to open"):
        call(OpenTargetTool(), {"target": "harmless.txt"}, ctx)
    assert opened == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX executable bit")
def test_executable_bit_is_refused_on_posix(workspace: Path, opened: list[str], ctx: ToolContext) -> None:
    script = workspace / "tool"
    script.write_text("#!/bin/sh\necho hi\n", encoding="utf-8")
    script.chmod(0o755)
    with pytest.raises(ToolError, match="executable"):
        call(OpenTargetTool(), {"target": "tool"}, ctx)
    assert opened == []


@pytest.mark.parametrize(
    "target",
    ["file:///C:/Windows/System32/calc.exe", "javascript:alert(1)", "mailto:someone@example.com",
     "ms-settings:privacy", "ftp://example.com/x", "vbscript:msgbox(1)", "search-ms:query=x",
     "https://user:pass@example.com/", "http:///nohost"],
)  # fmt: skip
def test_non_http_schemes_and_credentials_are_refused(target: str, opened: list[str], ctx: ToolContext) -> None:
    tool = OpenTargetTool()
    with pytest.raises(ToolArgumentError):
        tool.validate({"target": target})
    with pytest.raises(ToolArgumentError):
        tool.run({"target": target}, ctx)  # run() re-checks unvalidated input
    assert opened == []


def test_launcher_failure_becomes_tool_error(workspace: Path, ctx: ToolContext) -> None:
    def broken(target: str) -> None:
        raise OSError(1155, "No application is associated with the specified file")

    (workspace / "data.xyz").write_text("x", encoding="utf-8")
    with pytest.raises(ToolError, match="No application is associated"):
        call(OpenTargetTool(launcher=broken), {"target": "data.xyz"}, ctx)


@pytest.mark.parametrize("args", [{}, {"target": ""}, {"target": "  "}, {"target": 3}, {"target": "a.txt", "x": 1}])
def test_argument_validation(args: dict) -> None:
    with pytest.raises(ToolArgumentError):
        OpenTargetTool().validate(args)


def test_describe_call_shows_what_will_be_opened(workspace: Path, ctx: ToolContext) -> None:
    tool = OpenTargetTool()
    (workspace / "docs").mkdir()
    url = "https://example.com/a/very/long/path?with=query&and=" + "x" * 200
    assert tool.describe_call({"target": url}) == f"Open web page in the default browser: {url}"
    assert tool.describe_call({"target": "docs"}, ctx) == f"Open folder {(workspace / 'docs').resolve()}"
    assert tool.describe_call({"target": "docs/a.pdf"}, ctx) == (
        f"Open file {(workspace / 'docs' / 'a.pdf').resolve()} with its default application"
    )


def test_drive_letter_paths_are_paths_not_urls(workspace: Path, opened: list[str], ctx: ToolContext) -> None:
    target = workspace / "notes.md"
    target.write_text("x", encoding="utf-8")
    call(OpenTargetTool(), {"target": str(target)}, ctx)  # "C:\\..." must not be parsed as scheme "c"
    assert opened == [str(target.resolve())]
