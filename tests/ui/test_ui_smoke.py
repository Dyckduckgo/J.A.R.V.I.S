"""Launch the real application headless in a subprocess (it owns its own QApplication)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gui

SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def _env(home: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("GROQ_API_KEY", None)
    env.update(
        QT_QPA_PLATFORM="offscreen",
        JARVIS_HOME=str(home),
        # Never touch the real OS credential store from tests.
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        PYTHONPATH=os.pathsep.join(p for p in (str(SRC_DIR), env.get("PYTHONPATH", "")) if p),
    )
    return env


def test_version_flag(tmp_path: Path):
    result = subprocess.run(
        [sys.executable, "-m", "jarvis", "--version"],
        env=_env(tmp_path / "home"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "J.A.R.V.I.S." in result.stdout


def test_smoke_test_launches_and_exits_cleanly(tmp_path: Path):
    home = tmp_path / "home"
    result = subprocess.run(
        [sys.executable, "-m", "jarvis", "--smoke-test"],
        env=_env(home),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    log_text = (home / "logs" / "jarvis.log").read_text(encoding="utf-8")
    assert "[smoke test]" in log_text
    assert "exited with code 0" in log_text
    assert "Traceback" not in log_text


def test_startup_failure_returns_1_without_hanging(tmp_path: Path):
    script = (
        "import sys\n"
        "from jarvis import app_context\n"
        "from jarvis.errors import StorageError\n"
        "def boom(*a, **k):\n"
        "    raise StorageError('db broken', user_message='The conversation database could not be created.')\n"
        "app_context.AppContext.create = classmethod(lambda cls, *a, **k: boom())\n"
        "from jarvis.app import main\n"
        "sys.exit(main(['--smoke-test']))\n"
    )
    home = tmp_path / "home"
    result = subprocess.run(
        [sys.executable, "-c", script], env=_env(home), capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 1, result.stderr
    assert "Start-up failed" in (home / "logs" / "jarvis.log").read_text(encoding="utf-8")


def test_exception_hooks_log_and_report(caplog):
    import threading

    from jarvis.app import _install_exception_hooks

    reports: list[str] = []
    previous = (sys.excepthook, threading.excepthook)
    restore = _install_exception_hooks(reports.append)
    try:
        try:
            raise ValueError("ui bug")
        except ValueError:
            sys.excepthook(*sys.exc_info())
        thread = threading.Thread(target=lambda: 1 / 0, name="bg-worker")
        thread.start()
        thread.join()
    finally:
        restore()
    assert (sys.excepthook, threading.excepthook) == previous
    assert reports == ["ValueError in the user interface.", "ZeroDivisionError in background thread 'bg-worker'."]
    assert "Unhandled exception" in caplog.text and "ui bug" in caplog.text
