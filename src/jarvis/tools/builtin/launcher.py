"""Open a web page in the browser or an existing file/folder with its default
application (always requires the user's confirmation).

Executables and scripts are never launched from here; ``run_shell_command``
exists for that and shows the full command in its confirmation dialog.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from jarvis.errors import ToolError
from jarvis.tools.base import RiskLevel, ToolContext, ToolResult
from jarvis.tools.builtin._display import reveal_hidden
from jarvis.tools.builtin._paths import (
    WorkspaceAwareTool,
    check_path_argument,
    display_path,
    resolve_path,
    try_resolve,
)
from jarvis.tools.builtin.web import check_web_url

# Files that execute code (or point at code) when "opened" on at least one platform.
BLOCKED_SUFFIXES = frozenset(
    {
        # required minimum
        ".exe", ".bat", ".cmd", ".com", ".ps1", ".vbs", ".js", ".jse", ".wsf", ".wsh", ".msi", ".msc",
        ".scr", ".lnk", ".reg", ".hta", ".cpl", ".jar", ".sh",
        # further Windows launchers / shortcuts / installers
        ".pif", ".scf", ".url", ".website", ".application", ".appref-ms", ".gadget", ".msp", ".mst",
        ".psm1", ".psd1", ".ps1xml", ".psc1", ".vbe", ".ws", ".wsc", ".sct", ".shb", ".chm", ".inf",
        ".ins", ".isp", ".settingcontent-ms", ".xbap", ".dll", ".sys", ".diagcab", ".appx", ".msix",
        ".appxbundle", ".msixbundle", ".library-ms", ".searchconnector-ms",
        # interpreters registered as default handlers
        ".py", ".pyw", ".pyz", ".pyzw", ".pyc", ".pl", ".rb", ".php", ".tcl",
        # macOS / Linux
        ".command", ".app", ".desktop", ".appimage", ".run", ".bash", ".zsh", ".csh", ".ksh", ".bin",
    }
)  # fmt: skip

Launcher = Callable[[str], None]


def launch_target(target: str) -> None:
    """Hand ``target`` (an http(s) URL or an absolute path) to the OS default handler."""
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]  # noqa: S606
        return
    opener = ["open"] if sys.platform == "darwin" else ["xdg-open"]
    subprocess.Popen(  # noqa: S603
        [*opener, target],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _is_url(text: str) -> bool:
    scheme = urlsplit(text).scheme
    # One letter is a Windows drive ("C:\\..."), not a URL scheme.
    return len(scheme) > 1


def _blocked_reason(path: Path) -> str | None:
    if path.is_dir():
        return None
    # Windows ignores trailing dots/spaces ("evil.exe." runs as evil.exe).
    suffixes = {path.suffix.lower(), os.path.splitext(path.name.rstrip(". "))[1].lower()}
    blocked = sorted(suffixes & BLOCKED_SUFFIXES)
    if blocked:
        return f"files of type {blocked[0]} can run code"
    if os.name != "nt":
        try:
            mode = path.stat().st_mode
        except OSError:
            return None
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            return "the file is marked as executable"
    return None


class OpenTargetTool(WorkspaceAwareTool):
    name = "open_target"
    description = (
        "Open an http/https URL in the user's default web browser, or open an existing file or folder with its "
        "default application (e.g. a PDF viewer or File Explorer). Requires the user's approval. Programs and "
        "scripts (.exe, .bat, .ps1, .py, shortcuts, ...) are refused; use run_shell_command for those."
    )
    parameters = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "An http(s) URL, or a path to an existing file/folder (relative to the workspace or absolute).",
                "minLength": 1,
                "maxLength": 2048,
            }
        },
        "required": ["target"],
        "additionalProperties": False,
    }
    risk = RiskLevel.CONSEQUENTIAL

    def __init__(self, launcher: Launcher | None = None) -> None:
        self._launcher = launcher

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        result = super().validate(args)
        target = result["target"].strip()
        if _is_url(target):
            result["target"] = check_web_url(target, argument="target")
        else:
            result["target"] = check_path_argument(target, "target")
        return result

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        target = str(args.get("target", "")).strip()
        if _is_url(target):
            return f"Open web page in the default browser: {reveal_hidden(target)}"
        workspace = self._display_workspace(ctx)
        resolved = try_resolve(target, workspace)
        shown = display_path(target, workspace)
        if resolved is not None and resolved.is_dir():
            return f"Open folder {shown}"
        return f"Open file {shown} with its default application"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = args["target"].strip()
        if _is_url(target):
            url = check_web_url(target, argument="target")
            self._launch(url)
            return ToolResult.success(f"Opened {url} in the default web browser.")

        path = resolve_path(target, ctx.workspace, "target")
        if not path.exists():
            looks_like_domain = "." in target and not any(sep in target for sep in ("/", "\\"))
            hint = " If this is a web address, include http:// or https://." if looks_like_domain else ""
            raise ToolError(f"Nothing exists at {path}.{hint}")
        reason = _blocked_reason(path)
        if reason:
            raise ToolError(
                f"Refusing to open {path}: {reason}. Use run_shell_command if you really need to run it."
            )
        self._launch(str(path))
        kind = "folder" if path.is_dir() else "file"
        return ToolResult.success(f"Opened {kind} {path}.")

    def _launch(self, target: str) -> None:
        launcher = self._launcher or launch_target
        try:
            launcher(target)
        except ToolError:
            raise
        except OSError as exc:
            raise ToolError(f"Could not open {target}: {exc.strerror or exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Could not open {target}: {exc}") from exc


__all__ = ["BLOCKED_SUFFIXES", "OpenTargetTool", "launch_target"]
