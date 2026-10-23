"""Filesystem locations used by the application.

All per-user state lives outside the source tree. Setting the ``JARVIS_HOME``
environment variable relocates everything under a single directory, which is
used by the test-suite and for portable installs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir, user_documents_dir, user_log_dir

APP_DIR_NAME = "JARVIS"
HOME_ENV_VAR = "JARVIS_HOME"


@dataclass(frozen=True)
class AppPaths:
    config_dir: Path
    data_dir: Path
    log_dir: Path

    @property
    def settings_file(self) -> Path:
        return self.config_dir / "settings.json"

    @property
    def database_file(self) -> Path:
        return self.data_dir / "jarvis.db"

    @property
    def log_file(self) -> Path:
        return self.log_dir / "jarvis.log"

    def ensure(self) -> AppPaths:
        for directory in (self.config_dir, self.data_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @classmethod
    def under(cls, base: Path) -> AppPaths:
        base = Path(base).expanduser().resolve()
        return cls(base / "config", base / "data", base / "logs")

    @classmethod
    def default(cls) -> AppPaths:
        home = os.environ.get(HOME_ENV_VAR)
        if home:
            return cls.under(Path(home))
        return cls(
            config_dir=Path(user_config_dir(APP_DIR_NAME, appauthor=False, roaming=True)),
            data_dir=Path(user_data_dir(APP_DIR_NAME, appauthor=False)),
            log_dir=Path(user_log_dir(APP_DIR_NAME, appauthor=False)),
        )


def default_workspace_dir() -> Path:
    """Directory the file tools may write into without leaving their sandbox."""
    home = os.environ.get(HOME_ENV_VAR)
    if home:
        return Path(home).expanduser().resolve() / "workspace"
    return Path(user_documents_dir()) / "JARVIS Workspace"
