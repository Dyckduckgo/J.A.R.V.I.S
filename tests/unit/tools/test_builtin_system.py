"""get_datetime and system_info."""

from __future__ import annotations

import getpass
import os
import platform
import re
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from builtin_helpers import call

from jarvis.errors import ToolArgumentError
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.system import GetDateTimeTool, SystemInfoTool, format_utc_offset, memory_info


def _field(text: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}: (.+)$", text, re.MULTILINE)
    assert match, f"missing {label!r} in:\n{text}"
    return match.group(1)


def test_get_datetime_reports_consistent_local_and_utc_times(ctx: ToolContext) -> None:
    before = datetime.now(timezone.utc)
    result = call(GetDateTimeTool(), {}, ctx)
    after = datetime.now(timezone.utc)
    assert result.ok

    local = datetime.fromisoformat(_field(result.content, "Local date and time"))
    utc = datetime.fromisoformat(_field(result.content, "UTC"))
    assert local.tzinfo is not None and utc.utcoffset() == timedelta(0)
    assert local == utc  # the same instant, two representations
    assert before - timedelta(seconds=2) <= utc <= after + timedelta(seconds=2)
    weekdays = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
    assert _field(result.content, "Weekday") == weekdays[local.weekday()]
    assert f"UTC{format_utc_offset(local.utcoffset())}" in _field(result.content, "Time zone")


@pytest.mark.parametrize(
    ("offset", "expected"),
    [(timedelta(0), "+00:00"), (timedelta(hours=5, minutes=30), "+05:30"), (timedelta(hours=-3, minutes=-30), "-03:30")],
)
def test_format_utc_offset(offset: timedelta, expected: str) -> None:
    assert format_utc_offset(offset) == expected


def test_get_datetime_takes_no_arguments() -> None:
    with pytest.raises(ToolArgumentError):
        GetDateTimeTool().validate({"timezone": "UTC"})
    assert GetDateTimeTool().describe_call({}) == "get_datetime()"


def test_system_info_reports_basics(ctx: ToolContext) -> None:
    result = call(SystemInfoTool(), {}, ctx)
    assert result.ok
    text = result.content
    assert _field(text, "Operating system")
    assert _field(text, "Architecture") == (platform.machine() or "unknown")
    assert _field(text, "Logical CPUs") == str(os.cpu_count() or "unknown")
    assert platform.python_version() in _field(text, "Python")
    assert re.search(r"^Disk \(.+\): [\d.]+ GB free of [\d.]+ GB$", text, re.MULTILINE)
    if os.name == "nt":
        assert re.search(r"^Memory: [\d.]+ GB total, [\d.]+ GB available$", text, re.MULTILINE)


def test_system_info_does_not_leak_user_or_host_names(ctx: ToolContext, workspace: Path) -> None:
    text = call(SystemInfoTool(), {}, ctx).content
    secrets = {getpass.getuser(), socket.gethostname(), platform.node()}
    for secret in filter(None, secrets):
        assert not re.search(rf"(?<![A-Za-z0-9]){re.escape(secret)}(?![A-Za-z0-9])", text, re.IGNORECASE), secret
    assert str(workspace) not in text
    assert str(Path.home()) not in text


def test_system_info_handles_missing_workspace(tmp_path: Path) -> None:
    ctx = ToolContext(workspace=tmp_path / "does" / "not" / "exist")
    result = SystemInfoTool().run({}, ctx)
    assert result.ok and "Disk" in result.content


def test_memory_info_is_plausible() -> None:
    total, available = memory_info()
    if os.name == "nt":
        assert total and available
    if total and available:
        assert 0 < available <= total
