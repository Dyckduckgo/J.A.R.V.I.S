"""The assistant's built-in tools.

Risk levels (see :mod:`jarvis.tools.policy`):

=====================  ==========================================================
SAFE (runs on its own)  get_datetime, calculate, system_info, save_note,
                        search_notes; list_directory / read_file / search_files
                        while the path stays inside the workspace
CONSEQUENTIAL (asks)    write_file, run_shell_command, fetch_url, open_target;
                        the read tools above for paths outside the workspace
=====================  ==========================================================
"""

from __future__ import annotations

from jarvis.tools.base import Tool
from jarvis.tools.builtin.calculator import CalculateTool
from jarvis.tools.builtin.files import ListDirectoryTool, ReadFileTool, SearchFilesTool, WriteFileTool
from jarvis.tools.builtin.launcher import OpenTargetTool
from jarvis.tools.builtin.notes import SaveNoteTool, SearchNotesTool
from jarvis.tools.builtin.shell import RunShellCommandTool
from jarvis.tools.builtin.system import GetDateTimeTool, SystemInfoTool
from jarvis.tools.builtin.web import FetchUrlTool
from jarvis.tools.registry import ToolRegistry


def builtin_tools() -> list[Tool]:
    """Fresh instances of every built-in tool."""
    return [
        GetDateTimeTool(),
        CalculateTool(),
        SystemInfoTool(),
        ListDirectoryTool(),
        ReadFileTool(),
        SearchFilesTool(),
        WriteFileTool(),
        RunShellCommandTool(),
        FetchUrlTool(),
        OpenTargetTool(),
        SaveNoteTool(),
        SearchNotesTool(),
    ]


def create_default_registry() -> ToolRegistry:
    return ToolRegistry(builtin_tools())


__all__ = [
    "CalculateTool",
    "FetchUrlTool",
    "GetDateTimeTool",
    "ListDirectoryTool",
    "OpenTargetTool",
    "ReadFileTool",
    "RunShellCommandTool",
    "SaveNoteTool",
    "SearchFilesTool",
    "SearchNotesTool",
    "SystemInfoTool",
    "WriteFileTool",
    "builtin_tools",
    "create_default_registry",
]
