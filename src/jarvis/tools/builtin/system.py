"""Date/time and system information tools (read-only)."""

from __future__ import annotations

import os
import platform
import shutil
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from jarvis.tools.base import RiskLevel, Tool, ToolContext, ToolResult, truncate

_NO_ARGS: dict[str, Any] = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def format_utc_offset(offset: timedelta) -> str:
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


class GetDateTimeTool(Tool):
    name = "get_datetime"
    description = "Get the current local date and time (ISO 8601), the weekday, the time zone/UTC offset and the current UTC time."
    parameters = _NO_ARGS
    risk = RiskLevel.SAFE

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        now = datetime.now().astimezone()
        utc = now.astimezone(timezone.utc)
        offset = now.utcoffset() or timedelta(0)
        zone = now.tzname() or "local time"
        lines = [
            f"Local date and time: {now.isoformat(timespec='seconds')}",
            f"Weekday: {_WEEKDAYS[now.weekday()]}",
            f"Time zone: {zone} (UTC{format_utc_offset(offset)})",
            f"UTC: {utc.isoformat(timespec='seconds')}",
        ]
        return ToolResult.success("\n".join(lines))


# --- system information ---------------------------------------------------


def _gib(num_bytes: int) -> str:
    return f"{num_bytes / 1024**3:.1f} GB"


def _windows_memory() -> tuple[int, int] | None:
    import ctypes
    from ctypes import wintypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
    kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys), int(status.ullAvailPhys)


def _posix_memory() -> tuple[int | None, int | None]:
    total: int | None = None
    available: int | None = None
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        total = page * os.sysconf("SC_PHYS_PAGES")
        try:
            available = page * os.sysconf("SC_AVPHYS_PAGES")
        except (ValueError, OSError):
            available = None
    except (ValueError, OSError, AttributeError):
        return None, None
    # Linux: MemAvailable (includes reclaimable cache) is more meaningful than free pages.
    try:
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    available = int(line.split()[1]) * 1024
                    break
    except (OSError, ValueError):
        pass
    return total, available


def memory_info() -> tuple[int | None, int | None]:
    """(total, available) physical memory in bytes; ``None`` where unknown."""
    try:
        if os.name == "nt":
            result = _windows_memory()
            return result if result else (None, None)
        return _posix_memory()
    except Exception:  # noqa: BLE001 - best effort
        return None, None


def _os_description() -> str:
    system = platform.system()
    if system == "Windows":
        release, version, _csd, _ptype = platform.win32_ver()
        edition = ""
        try:
            edition = platform.win32_edition() or ""
        except Exception:  # noqa: BLE001
            edition = ""
        return f"Windows {release} {edition}".strip() + f" (build {version})"
    if system == "Darwin":
        return f"macOS {platform.mac_ver()[0]} (Darwin {platform.release()})"
    pretty = ""
    try:
        pretty = platform.freedesktop_os_release().get("PRETTY_NAME", "")
    except (OSError, AttributeError):
        pretty = ""
    base = f"{system} {platform.release()}"
    return f"{pretty} ({base})" if pretty else base


def _existing_ancestor(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    return Path(path.anchor or os.sep)


class SystemInfoTool(Tool):
    name = "system_info"
    description = (
        "Get basic information about this computer: operating system and version, CPU architecture and core "
        "count, Python version, total/available memory and free disk space on the workspace drive."
    )
    parameters = _NO_ARGS
    risk = RiskLevel.SAFE

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        # Deliberately no user name, host name or file paths: this text is sent to a cloud LLM.
        lines = [
            f"Operating system: {_os_description()}",
            f"Architecture: {platform.machine() or 'unknown'}",
        ]
        if os.name == "nt":
            processor = os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
            if processor:
                lines.append(f"Processor: {processor}")
        lines.append(f"Logical CPUs: {os.cpu_count() or 'unknown'}")
        lines.append(
            f"Python: {platform.python_version()} ({platform.python_implementation()}, "
            f"{struct.calcsize('P') * 8}-bit)"
        )
        total, available = memory_info()
        if total:
            lines.append(f"Memory: {_gib(total)} total" + (f", {_gib(available)} available" if available else ""))
        try:
            workspace = Path(ctx.workspace).expanduser().resolve()
            usage = shutil.disk_usage(_existing_ancestor(workspace))
            label = f"drive {workspace.drive}" if workspace.drive else "workspace volume"
            lines.append(f"Disk ({label}): {_gib(usage.free)} free of {_gib(usage.total)}")
        except OSError:
            pass
        return ToolResult.success(truncate("\n".join(lines)))
