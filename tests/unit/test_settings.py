from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.config.settings import DEFAULT_MODEL, Settings, SettingsManager, Theme
from jarvis.errors import ConfigError


def test_defaults_are_valid(isolated_environment: Path) -> None:
    settings = Settings()
    assert settings.model == DEFAULT_MODEL
    assert settings.theme is Theme.DARK
    # JARVIS_HOME relocates the default workspace too
    assert Path(settings.workspace_dir) == isolated_environment.resolve() / "workspace"


def test_with_changes_validates_and_is_immutable() -> None:
    base = Settings()
    changed = base.with_changes(temperature=1.2, disabled_tools=["b", "a", "a", " "])
    assert changed.temperature == 1.2
    assert changed.disabled_tools == ("a", "b")
    assert base.temperature == 0.4
    with pytest.raises(Exception):
        base.temperature = 1.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "changes",
    [{"temperature": 3}, {"max_tokens": 10}, {"model": "   "}, {"workspace_dir": ""}, {"theme": "neon"}],
)
def test_invalid_changes_raise_config_error(changes: dict) -> None:
    with pytest.raises(ConfigError) as info:
        Settings().with_changes(**changes)
    assert info.value.user_message.startswith("Invalid settings")


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "cfg" / "settings.json"
    manager = SettingsManager(path)
    saved = manager.update(model="openai/gpt-oss-20b", custom_instructions="Be brief.")
    assert path.exists()
    assert not list(path.parent.glob(".settings-*.tmp")), "temp file must not linger"

    reloaded = SettingsManager(path).load()
    assert reloaded == saved
    assert reloaded.custom_instructions == "Be brief."
    assert manager.recovery_notice is None


def test_missing_file_uses_defaults(tmp_path: Path) -> None:
    manager = SettingsManager(tmp_path / "nope.json")
    assert manager.load() == Settings()
    assert manager.recovery_notice is None


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", json.dumps({"temperature": "hot"})])
def test_corrupt_file_is_quarantined_and_defaults_used(tmp_path: Path, content: str) -> None:
    path = tmp_path / "settings.json"
    path.write_text(content, encoding="utf-8")
    manager = SettingsManager(path)
    assert manager.load() == Settings()
    assert manager.recovery_notice and "reset to defaults" in manager.recovery_notice
    backups = list(tmp_path.glob("settings.json.invalid-*"))
    assert len(backups) == 1 and backups[0].read_text(encoding="utf-8") == content
    assert not path.exists()


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"model": "x-model", "legacy_option": True}), encoding="utf-8")
    assert SettingsManager(path).load().model == "x-model"


def test_save_failure_raises_config_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    manager = SettingsManager(blocker / "settings.json")  # parent is a file -> cannot create dir
    with pytest.raises(ConfigError):
        manager.save(Settings())
