"""User-configurable application settings.

Settings are an immutable, validated pydantic model persisted as JSON. The API
key is deliberately *not* part of this model; see :mod:`jarvis.config.secrets`.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from jarvis.errors import ConfigError
from jarvis.paths import default_workspace_dir

log = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
# Production chat models listed by Groq that support tool use. The settings
# dialog can also fetch the live list from the API.
SUGGESTED_MODELS: tuple[str, ...] = (
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
)


class Theme(str, Enum):
    DARK = "dark"
    LIGHT = "light"


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", protected_namespaces=())

    model: str = Field(default=DEFAULT_MODEL, min_length=1, max_length=200)
    temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=64, le=32768)
    max_tool_iterations: int = Field(default=8, ge=1, le=30)
    max_history_messages: int = Field(default=50, ge=2, le=500)
    request_timeout_seconds: float = Field(default=60.0, ge=5.0, le=600.0)
    shell_timeout_seconds: int = Field(default=60, ge=1, le=3600)
    stream_responses: bool = True
    workspace_dir: str = Field(default_factory=lambda: str(default_workspace_dir()))
    disabled_tools: tuple[str, ...] = ()
    custom_instructions: str = Field(default="", max_length=4000)
    theme: Theme = Theme.DARK

    @field_validator("model")
    @classmethod
    def _strip_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model must not be empty")
        return value

    @field_validator("workspace_dir")
    @classmethod
    def _check_workspace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("workspace_dir must not be empty")
        return str(Path(value).expanduser())

    @field_validator("disabled_tools", mode="before")
    @classmethod
    def _dedupe_tools(cls, value: Any) -> Any:
        if isinstance(value, (list, tuple, set)):
            return tuple(sorted({str(v).strip() for v in value if str(v).strip()}))
        return value

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace_dir).expanduser().resolve()

    def with_changes(self, **changes: Any) -> Settings:
        """Return a validated copy with ``changes`` applied (raises ConfigError)."""
        data = self.model_dump(mode="json")
        data.update(changes)
        try:
            return Settings.model_validate(data)
        except ValidationError as exc:
            raise ConfigError(str(exc), user_message=_friendly_validation_error(exc)) from exc


def _friendly_validation_error(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        field = ".".join(str(p) for p in err.get("loc", ())) or "value"
        parts.append(f"{field}: {err.get('msg', 'invalid value')}")
    return "Invalid settings - " + "; ".join(parts)


class SettingsManager:
    """Loads, validates and atomically persists :class:`Settings`.

    A corrupt or invalid settings file never prevents start-up: it is moved
    aside (``settings.json.invalid-<timestamp>``) and defaults are used. The
    reason is exposed via :attr:`recovery_notice` so the GUI can tell the user.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._settings = Settings()
        self.recovery_notice: str | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def settings(self) -> Settings:
        with self._lock:
            return self._settings

    def load(self) -> Settings:
        with self._lock:
            self.recovery_notice = None
            if not self._path.exists():
                log.info("No settings file at %s; using defaults", self._path)
                self._settings = Settings()
                return self._settings
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    raise ValueError("settings file does not contain a JSON object")
                self._settings = Settings.model_validate(raw)
            except (OSError, ValueError, ValidationError) as exc:
                backup = self._quarantine()
                log.warning("Settings file invalid (%s); reset to defaults, backup at %s", exc, backup)
                self.recovery_notice = (
                    "Your settings file could not be read and was reset to defaults."
                    + (f" The old file was saved as {backup.name}." if backup else "")
                )
                self._settings = Settings()
            return self._settings

    def save(self, settings: Settings) -> None:
        with self._lock:
            self._write_atomic(settings)
            self._settings = settings
        log.info("Settings saved")

    def update(self, **changes: Any) -> Settings:
        new = self.settings.with_changes(**changes)
        self.save(new)
        return new

    def _write_atomic(self, settings: Settings) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(settings.model_dump(mode="json"), indent=2, sort_keys=True)
            fd, tmp_name = tempfile.mkstemp(prefix=".settings-", suffix=".tmp", dir=self._path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, self._path)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise ConfigError(
                f"Could not write settings to {self._path}: {exc}",
                user_message=f"Could not save settings: {exc.strerror or exc}",
            ) from exc

    def _quarantine(self) -> Path | None:
        target = self._path.with_name(f"{self._path.name}.invalid-{time.strftime('%Y%m%d-%H%M%S')}")
        try:
            os.replace(self._path, target)
            return target
        except OSError:
            log.exception("Could not move invalid settings file aside")
            return None
