"""Shared pytest configuration.

* Qt runs with the ``offscreen`` platform so GUI tests need no display.
* Every test gets an isolated ``JARVIS_HOME`` so nothing touches real user data.
* ``GROQ_API_KEY`` is captured once for the opt-in live tests and then removed
  from the environment so ordinary tests never hit the network by accident.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_LIVE_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

from fakes import MemoryKeyring  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "jarvis-home"
    monkeypatch.setenv("JARVIS_HOME", str(home))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    return home


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def memory_keyring() -> MemoryKeyring:
    return MemoryKeyring()


@pytest.fixture
def live_api_key() -> str:
    if not _LIVE_API_KEY:
        pytest.skip("GROQ_API_KEY not set; live Groq tests skipped")
    return _LIVE_API_KEY
