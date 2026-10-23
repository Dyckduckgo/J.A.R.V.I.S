"""Run a shell command (always requires the user's confirmation).

Windows uses Windows PowerShell, everything else ``/bin/sh``. The child gets
no stdin, its own process group, and is killed together with all of its
descendants when the timeout expires or the user presses Stop.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from typing import IO, Any

from jarvis.errors import ToolArgumentError, ToolError
from jarvis.tools.base import RiskLevel, ToolContext, ToolResult, truncate
from jarvis.tools.builtin._display import reveal_hidden
from jarvis.tools.builtin._paths import (
    WorkspaceAwareTool,
    check_path_argument,
    display_path,
    resolve_path,
    resolve_workspace,
)

IS_WINDOWS = os.name == "nt"
SHELL_LABEL = "Windows PowerShell" if IS_WINDOWS else "/bin/sh"
MAX_COMMAND_LENGTH = 8000
MAX_CAPTURE_BYTES = 1024 * 1024  # per stream; the rest is drained and discarded
STDOUT_CHARS = 12_000
STDERR_CHARS = 6_000
POLL_INTERVAL = 0.05
READER_GRACE_SECONDS = 2.0
# Environment variables never passed to commands (the app's own secrets).
SCRUBBED_ENV_VARS = ("GROQ_API_KEY",)

# One line before the user's command: UTF-8 output (decoded as such below), no
# progress records (PowerShell would serialise them to stderr as CLIXML) and a
# wide console buffer so tables and long paths are not cut off with "...".
_PS_PREAMBLE = (
    "$ProgressPreference = 'SilentlyContinue'; "
    "try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}; "
    "try { $__jarvis_ui = $Host.UI.RawUI; $__jarvis_ui.BufferSize = "
    "New-Object System.Management.Automation.Host.Size(400, $__jarvis_ui.BufferSize.Height) } catch {}\n"
)
# One line after it: `powershell -Command` only reports 0/1, so propagate the
# exit code of a failing native program (the documented PowerShell idiom).
_PS_TRAILER = "\n$__jarvis_ok = $?; if (-not $__jarvis_ok) { if ($LASTEXITCODE) { exit $LASTEXITCODE } else { exit 1 } }"


def _system32(*parts: str) -> str:
    root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
    return os.path.join(root, "System32", *parts)


def _powershell_executable() -> str:
    # An absolute path avoids picking up a planted powershell.exe from the current directory.
    candidate = _system32("WindowsPowerShell", "v1.0", "powershell.exe")
    return candidate if os.path.isfile(candidate) else "powershell"


def build_invocation(command: str) -> tuple[list[str], dict[str, Any]]:
    if IS_WINDOWS:
        argv = [_powershell_executable(), "-NoProfile", "-NonInteractive", "-Command", _PS_PREAMBLE + command + _PS_TRAILER]
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        return argv, {"creationflags": flags}
    return ["/bin/sh", "-c", command], {"start_new_session": True}


def kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill ``proc`` and all of its descendants."""
    if IS_WINDOWS:
        taskkill = _system32("taskkill.exe")
        try:
            subprocess.run(
                [taskkill if os.path.isfile(taskkill) else "taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # start_new_session: pgid == pid
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


class _StreamCollector:
    """Drains a pipe on a daemon thread so the child can never block on a full pipe."""

    def __init__(self, stream: IO[bytes], limit: int = MAX_CAPTURE_BYTES) -> None:
        self._limit = limit
        self._chunks: list[bytes] = []
        self._size = 0
        self.dropped = 0
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._pump, args=(stream,), daemon=True, name="shell-output-reader")
        self._thread.start()

    def _pump(self, stream: IO[bytes]) -> None:
        read = getattr(stream, "read1", stream.read)
        try:
            while True:
                chunk = read(65536)
                if not chunk:
                    break
                with self._lock:
                    room = self._limit - self._size
                    if room > 0:
                        kept = chunk[:room]
                        self._chunks.append(kept)
                        self._size += len(kept)
                    self.dropped += len(chunk) - max(0, min(room, len(chunk)))
        except (OSError, ValueError):
            pass
        finally:
            try:
                stream.close()
            except OSError:
                pass

    def join(self, timeout: float) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def text(self) -> str:
        with self._lock:
            data = b"".join(self._chunks)
            dropped = self.dropped
        text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
        text = "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")
        if dropped:
            text += f"\n... [{dropped:,} more bytes of output discarded]"
        return text


class RunShellCommandTool(WorkspaceAwareTool):
    name = "run_shell_command"
    description = (
        f"Run a command in {SHELL_LABEL} and return its exit code, stdout and stderr. Always requires the "
        "user's approval. Non-interactive (no input is possible); the command runs in the workspace folder "
        "unless working_directory is given and is killed (with any child processes) if it exceeds the "
        "configured timeout."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": f"The complete command line to run in {SHELL_LABEL}.",
                "minLength": 1,
                "maxLength": MAX_COMMAND_LENGTH,
            },
            "working_directory": {
                "type": "string",
                "description": "Existing folder to run in; relative to the workspace or absolute. Default: the workspace.",
                "maxLength": 1024,
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    risk = RiskLevel.CONSEQUENTIAL

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        result = super().validate(args)
        if not result["command"].strip():
            raise ToolArgumentError("Argument 'command' must not be blank.")
        if "\x00" in result["command"]:
            raise ToolArgumentError("Argument 'command' must not contain NUL characters.")
        if "working_directory" in result:
            result["working_directory"] = check_path_argument(result["working_directory"], "working_directory")
        return result

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        workspace = self._display_workspace(ctx)
        if args.get("working_directory"):
            where = display_path(args["working_directory"], workspace)
        elif workspace is not None:
            where = str(resolve_workspace(workspace))
        else:
            where = "the workspace folder"
        return f"Run in {SHELL_LABEL} (working directory: {where}): {reveal_hidden(str(args.get('command', '')))}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command: str = args["command"]
        if args.get("working_directory"):
            cwd = resolve_path(args["working_directory"], ctx.workspace, "working_directory")
        else:
            cwd = resolve_workspace(ctx.workspace)
        if not cwd.is_dir():
            raise ToolError(f"Working directory does not exist or is not a folder: {cwd}")

        argv, platform_kwargs = build_invocation(command)
        env = dict(os.environ)
        for key in list(env):
            if key.upper() in SCRUBBED_ENV_VARS:
                del env[key]
        try:
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                **platform_kwargs,
            )
        except FileNotFoundError as exc:
            raise ToolError(f"{SHELL_LABEL} was not found on this system.") from exc
        except OSError as exc:
            raise ToolError(f"Could not start {SHELL_LABEL}: {exc}") from exc

        assert proc.stdout is not None and proc.stderr is not None
        stdout = _StreamCollector(proc.stdout)
        stderr = _StreamCollector(proc.stderr)
        timeout = max(1, int(ctx.shell_timeout))
        started = time.monotonic()
        stopped: str | None = None
        while True:
            try:
                proc.wait(timeout=POLL_INTERVAL)
                break
            except subprocess.TimeoutExpired:
                pass
            if ctx.cancel_event.is_set():
                stopped = "cancelled"
                break
            if time.monotonic() - started >= timeout:
                stopped = "timeout"
                break
        if stopped:
            kill_process_tree(proc)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        elapsed = time.monotonic() - started
        streams_closed = stdout.join(READER_GRACE_SECONDS) & stderr.join(READER_GRACE_SECONDS)

        out_text = stdout.text()
        err_text = stderr.text()
        sections: list[str] = []
        if stopped == "timeout":
            sections.append(f"Command timed out after {timeout} s and was stopped (process tree killed).")
        elif stopped == "cancelled":
            sections.append(f"Command cancelled by the user after {elapsed:.1f} s (process tree killed).")
        else:
            sections.append(f"Exit code: {proc.returncode}")
        if not streams_closed:
            sections.append("Note: a background process is still holding the output open; output may be incomplete.")
        if out_text:
            sections.append("--- stdout ---\n" + truncate(out_text, STDOUT_CHARS))
        if err_text:
            sections.append("--- stderr ---\n" + truncate(err_text, STDERR_CHARS))
        if not out_text and not err_text:
            sections.append("(no output)")
        content = truncate("\n".join(sections))
        if stopped or proc.returncode != 0:
            return ToolResult.failure(content)
        return ToolResult.success(content)


__all__ = ["RunShellCommandTool", "build_invocation", "kill_process_tree"]
