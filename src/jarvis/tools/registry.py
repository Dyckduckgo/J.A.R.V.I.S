"""Tool registry."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from jarvis.tools.base import Tool

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if not _NAME_RE.match(tool.name):
            raise ValueError(f"Invalid tool name {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def all(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: t.name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, disabled: Iterable[str] = ()) -> list[dict[str, Any]]:
        excluded = set(disabled)
        return [tool.schema() for tool in self.all() if tool.name not in excluded]
