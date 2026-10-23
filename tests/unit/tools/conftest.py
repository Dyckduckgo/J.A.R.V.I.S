"""Fixtures shared by the built-in tool tests (helpers live in builtin_helpers.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jarvis.tools.base import ToolContext


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, shell_timeout=20, http_timeout=5.0)


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A folder next to (not inside) the workspace, holding a 'secret' file."""
    path = tmp_path / "outside"
    path.mkdir()
    (path / "secret.txt").write_text("top secret", encoding="utf-8")
    return path
