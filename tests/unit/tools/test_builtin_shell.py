"""run_shell_command against the real shell (PowerShell on Windows, /bin/sh elsewhere)."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from builtin_helpers import call

from jarvis.errors import ToolArgumentError, ToolError
from jarvis.tools.base import RiskLevel, ToolContext
from jarvis.tools.builtin.shell import SHELL_LABEL, RunShellCommandTool

WINDOWS = os.name == "nt"

if WINDOWS:
    ECHO_HELLO = 'Write-Output "hello world"'
    EXIT_3 = "exit 3"
    NATIVE_EXIT_5 = 'cmd /c "exit 5"'
    TO_STDERR = '[Console]::Error.WriteLine("oops on stderr"); Write-Output "out"'
    PRINT_CWD = "(Get-Location).Path"
    SLEEP_10 = "Start-Sleep -Seconds 10"
    SLEEP_30 = "Start-Sleep -Seconds 30"
    PRINT_KEY = 'Write-Output "key=[$env:GROQ_API_KEY]"'
    BIG_OUTPUT = "Write-Output ('x' * 300000)"
    UNICODE = 'Write-Output "café ✓ 日本"'
    # A grandchild process whose PID is written to a file, then the shell itself sleeps.
    SPAWN_CHILD = (
        "$p = Start-Process -FilePath powershell.exe -ArgumentList '-NoProfile','-Command','Start-Sleep -Seconds 60' "
        "-WindowStyle Hidden -PassThru; Set-Content -Path child.pid -Value $p.Id -Encoding ascii; Start-Sleep -Seconds 60"
    )
else:
    ECHO_HELLO = 'echo "hello world"'
    EXIT_3 = "exit 3"
    NATIVE_EXIT_5 = "sh -c 'exit 5'"
    TO_STDERR = "echo 'oops on stderr' 1>&2; echo out"
    PRINT_CWD = "pwd -P"
    SLEEP_10 = "sleep 10"
    SLEEP_30 = "sleep 30"
    PRINT_KEY = 'echo "key=[$GROQ_API_KEY]"'
    BIG_OUTPUT = "head -c 300000 /dev/zero | tr '\\0' x"
    UNICODE = "printf 'café ✓ 日本\\n'"
    SPAWN_CHILD = "sleep 60 & echo $! > child.pid; wait"


def _process_alive(pid: int) -> bool:
    if WINDOWS:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_until_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_alive(pid):
            return True
        time.sleep(0.1)
    return False


def test_successful_command(ctx: ToolContext) -> None:
    result = call(RunShellCommandTool(), {"command": ECHO_HELLO}, ctx)
    assert result.ok
    assert result.content.startswith("Exit code: 0")
    assert "--- stdout ---\nhello world" in result.content
    assert "stderr" not in result.content


def test_unicode_output_is_decoded(ctx: ToolContext) -> None:
    result = call(RunShellCommandTool(), {"command": UNICODE}, ctx)
    assert result.ok
    assert "café ✓ 日本" in result.content


def test_non_zero_exit_is_a_failure(ctx: ToolContext) -> None:
    result = call(RunShellCommandTool(), {"command": EXIT_3}, ctx)
    assert not result.ok
    assert result.content.startswith("Exit code: 3")


def test_exit_code_of_native_program_is_propagated(ctx: ToolContext) -> None:
    result = call(RunShellCommandTool(), {"command": NATIVE_EXIT_5}, ctx)
    assert not result.ok
    assert result.content.startswith("Exit code: 5")


def test_stderr_is_captured_separately(ctx: ToolContext) -> None:
    result = call(RunShellCommandTool(), {"command": TO_STDERR}, ctx)
    assert "--- stdout ---\nout" in result.content
    assert "--- stderr ---\noops on stderr" in result.content


def test_runs_in_workspace_or_given_working_directory(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "sub").mkdir()
    default = call(RunShellCommandTool(), {"command": PRINT_CWD}, ctx)
    assert os.path.normcase(str(workspace.resolve())) in os.path.normcase(default.content)
    sub = call(RunShellCommandTool(), {"command": PRINT_CWD, "working_directory": "sub"}, ctx)
    assert os.path.normcase(str((workspace / "sub").resolve())) in os.path.normcase(sub.content)


def test_missing_working_directory_is_an_error(ctx: ToolContext) -> None:
    with pytest.raises(ToolError, match="does not exist"):
        call(RunShellCommandTool(), {"command": ECHO_HELLO, "working_directory": "nope"}, ctx)


def test_large_output_does_not_deadlock_and_is_truncated(ctx: ToolContext) -> None:
    started = time.monotonic()
    result = call(RunShellCommandTool(), {"command": BIG_OUTPUT}, ctx)
    assert result.ok
    assert "truncated" in result.content
    assert len(result.content) < 25_000
    assert time.monotonic() - started < 15


def test_timeout_kills_the_command_quickly(workspace: Path) -> None:
    ctx = ToolContext(workspace=workspace, shell_timeout=1)
    started = time.monotonic()
    result = call(RunShellCommandTool(), {"command": SLEEP_10}, ctx)
    elapsed = time.monotonic() - started
    assert not result.ok
    assert "timed out after 1 s" in result.content
    assert elapsed < 8, elapsed


def test_timeout_kills_the_whole_process_tree(workspace: Path) -> None:
    ctx = ToolContext(workspace=workspace, shell_timeout=4)
    result = call(RunShellCommandTool(), {"command": SPAWN_CHILD}, ctx)
    assert not result.ok and "timed out" in result.content
    pid_file = workspace / "child.pid"
    assert pid_file.exists(), result.content
    child_pid = int(pid_file.read_text(encoding="ascii").strip())
    assert _wait_until_dead(child_pid), f"grandchild {child_pid} survived the timeout"


def test_cancel_event_stops_the_command(workspace: Path) -> None:
    ctx = ToolContext(workspace=workspace, shell_timeout=60)
    timer = threading.Timer(1.0, ctx.cancel_event.set)
    timer.start()
    started = time.monotonic()
    try:
        result = call(RunShellCommandTool(), {"command": SLEEP_30}, ctx)
    finally:
        timer.cancel()
    assert not result.ok
    assert "cancelled by the user" in result.content
    assert time.monotonic() - started < 10


def test_app_secrets_are_not_passed_to_commands(ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_not_a_real_key")
    result = call(RunShellCommandTool(), {"command": PRINT_KEY}, ctx)
    assert "key=[]" in result.content
    assert "gsk_test_not_a_real_key" not in result.content


@pytest.mark.parametrize(
    "args",
    [{}, {"command": ""}, {"command": "   "}, {"command": "echo\x00hi"}, {"command": 5},
     {"command": "x" * 8001}, {"command": "echo", "working_directory": ""}, {"command": "echo", "timeout": 5}],
)  # fmt: skip
def test_argument_validation(args: dict) -> None:
    with pytest.raises(ToolArgumentError):
        RunShellCommandTool().validate(args)


def test_describe_call_shows_full_command_and_directory(workspace: Path, ctx: ToolContext) -> None:
    tool = RunShellCommandTool()
    command = "Get-ChildItem -Recurse | Remove-Item -WhatIf; " + "x" * 300
    args = tool.validate({"command": command, "working_directory": "build"})
    text = tool.describe_call(args, ctx)
    assert text == f"Run in {SHELL_LABEL} (working directory: {(workspace / 'build').resolve()}): {command}"
    plain = tool.validate({"command": "dir"})
    assert tool.describe_call(plain) == f"Run in {SHELL_LABEL} (working directory: the workspace folder): dir"
    assert tool.assess_risk(plain, ctx) is RiskLevel.CONSEQUENTIAL
    assert str(workspace.resolve()) in tool.describe_call(plain)
