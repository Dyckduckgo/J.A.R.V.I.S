"""Helpers for the built-in tool tests (imported directly; not a test module)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from jarvis.tools.base import Tool, ToolContext, ToolResult


def call(tool: Tool, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    """Validate then run, exactly like the agent does."""
    return tool.run(tool.validate(args), ctx)


def make_symlink(link: Path, target: Path) -> None:
    """Create a symlink or skip the test when this machine does not allow it."""
    try:
        os.symlink(target, link, target_is_directory=target.is_dir())
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"cannot create symlinks on this machine: {exc}")


def make_junction(link: Path, target: Path) -> None:
    """Create an NTFS junction (no admin rights needed) or skip."""
    if os.name != "nt":
        pytest.skip("junctions are Windows-only")
    try:
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    except (ImportError, AttributeError, OSError) as exc:
        pytest.skip(f"cannot create junctions on this machine: {exc}")
