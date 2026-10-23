"""Path handling shared by the built-in tools.

Rules:

* relative paths resolve against the workspace, ``~`` expands to the home folder;
* paths are resolved with :meth:`Path.resolve` (symlinks/junctions followed,
  ``..`` collapsed), so a link inside the workspace that points elsewhere
  counts as *outside*;
* "inside the workspace" means the resolved path is the workspace itself or
  below it, compared case-insensitively on Windows (``os.path.normcase``).

Network (UNC) and device paths are rejected on Windows before anything touches
the file system: merely resolving ``\\\\host\\share`` would make Windows open an
SMB connection (and offer the user's NTLM credentials) to that host.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from jarvis.errors import ToolArgumentError, ToolError
from jarvis.tools.base import RiskLevel, Tool, ToolContext
from jarvis.tools.builtin._display import reveal_hidden


def check_path_argument(value: Any, name: str = "path") -> str:
    """Syntactic checks on a raw path argument; returns it stripped."""
    if not isinstance(value, str):
        raise ToolArgumentError(f"Argument '{name}' must be a string.")
    text = value.strip()
    if not text:
        raise ToolArgumentError(f"Argument '{name}' must not be empty.")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        # Also keeps line breaks out of the confirmation dialog, where they could fake extra lines.
        raise ToolArgumentError(f"Argument '{name}' must not contain control characters (such as NUL or line breaks).")
    if os.name == "nt" and text.replace("/", "\\").startswith("\\\\"):
        raise ToolArgumentError(f"Network (UNC) and device paths are not supported: {text}")
    return text


def resolve_workspace(workspace: Path | str) -> Path:
    return Path(workspace).expanduser().resolve()


def resolve_path(raw: Any, workspace: Path | str, name: str = "path") -> Path:
    """Resolve a user/model supplied path to an absolute, symlink-free path."""
    text = check_path_argument(raw, name)
    candidate = Path(os.path.expanduser(text))
    if not candidate.is_absolute():
        candidate = Path(workspace).expanduser() / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise ToolError(f"Cannot resolve path {text!r}: {exc}") from exc


def is_within(path: Path | str, root: Path | str) -> bool:
    """True if ``path`` equals ``root`` or lies below it (both already resolved)."""
    p = os.path.normcase(os.path.normpath(str(path)))
    r = os.path.normcase(os.path.normpath(str(root)))
    if p == r:
        return True
    if not r.endswith(os.sep):
        r += os.sep
    return p.startswith(r)


def is_inside_workspace(path: Path, workspace: Path | str) -> bool:
    return is_within(path, resolve_workspace(workspace))


def try_resolve(raw: Any, workspace: Path | None) -> Path | None:
    """Best-effort resolution for display purposes; never raises."""
    try:
        text = check_path_argument(raw)
        candidate = Path(os.path.expanduser(text))
        if not candidate.is_absolute():
            if workspace is None:
                return None
            candidate = Path(workspace).expanduser() / candidate
        return candidate.resolve()
    except Exception:  # noqa: BLE001 - display helper must not fail
        return None


def display_path(raw: Any, workspace: Path | None) -> str:
    """Absolute path for the confirmation dialog; never raises, never hides anything."""
    resolved = try_resolve(raw, workspace)
    if resolved is not None:
        return reveal_hidden(str(resolved))
    try:
        text = check_path_argument(raw)
    except ToolArgumentError:
        return f"{raw!r} (invalid path)"
    return f"{reveal_hidden(text)} (relative to the workspace folder)"


def format_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("bytes", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "bytes":
                return "1 byte" if num_bytes == 1 else f"{num_bytes} bytes"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover - loop always returns


class WorkspaceAwareTool(Tool):
    """Base class for tools whose confirmation text needs the workspace.

    :meth:`Tool.describe_call` receives no :class:`ToolContext`, so these tools
    accept an optional ``ctx`` and otherwise fall back to the workspace seen by
    the most recent :meth:`assess_risk` call (the agent always assesses a call
    before describing it). If neither is known, relative paths are shown as
    relative to the workspace instead of being guessed.
    """

    _workspace_hint: Path | None = None

    def assess_risk(self, args: dict[str, Any], ctx: ToolContext) -> RiskLevel:
        self._workspace_hint = Path(ctx.workspace)
        try:
            assessed = self._assess(args, ctx)
        except Exception:  # noqa: BLE001 - fail closed
            return RiskLevel.CONSEQUENTIAL
        if RiskLevel.CONSEQUENTIAL in (assessed, self.risk):
            return RiskLevel.CONSEQUENTIAL
        return RiskLevel.SAFE

    def _assess(self, args: dict[str, Any], ctx: ToolContext) -> RiskLevel:
        return self.risk

    def _display_workspace(self, ctx: ToolContext | None) -> Path | None:
        if ctx is not None:
            return Path(ctx.workspace)
        return self._workspace_hint


class PathReadTool(WorkspaceAwareTool):
    """A read-only tool: SAFE inside the workspace, CONSEQUENTIAL outside."""

    risk = RiskLevel.SAFE
    path_argument: str = "path"

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        result = super().validate(args)
        if self.path_argument in result:
            result[self.path_argument] = check_path_argument(result[self.path_argument], self.path_argument)
        return result

    def _assess(self, args: dict[str, Any], ctx: ToolContext) -> RiskLevel:
        target = resolve_path(args.get(self.path_argument, "."), ctx.workspace, self.path_argument)
        return RiskLevel.SAFE if is_inside_workspace(target, ctx.workspace) else RiskLevel.CONSEQUENTIAL
