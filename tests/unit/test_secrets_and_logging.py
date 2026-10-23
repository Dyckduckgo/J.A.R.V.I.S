from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fakes import MemoryKeyring

from jarvis.config.secrets import API_KEY_USERNAME, SERVICE_NAME, KeySource, SecretStore, mask_secret
from jarvis.errors import SecretStoreError
from jarvis.logging_setup import configure_logging, redact, register_secret

KEY = "gsk_" + "A1b2C3d4" * 6


def test_keyring_roundtrip(memory_keyring: MemoryKeyring) -> None:
    store = SecretStore(backend=memory_keyring, env={})
    assert store.get_api_key() is None
    assert store.source() is KeySource.NONE
    store.set_api_key(f"  {KEY}  ")
    assert memory_keyring.store[(SERVICE_NAME, API_KEY_USERNAME)] == KEY
    assert store.get_api_key() == KEY
    assert store.source() is KeySource.KEYRING
    store.delete_api_key()
    assert store.get_api_key() is None
    store.delete_api_key()  # idempotent


def test_environment_overrides_keyring(memory_keyring: MemoryKeyring) -> None:
    memory_keyring.set_password(SERVICE_NAME, API_KEY_USERNAME, KEY)
    store = SecretStore(backend=memory_keyring, env={"GROQ_API_KEY": "gsk_fromenvironment000000000"})
    assert store.get_api_key() == "gsk_fromenvironment000000000"
    assert store.source() is KeySource.ENVIRONMENT
    assert store.has_stored_key() is True
    assert SecretStore(backend=MemoryKeyring(), env={"GROQ_API_KEY": "gsk_fromenvironment000000000"}).has_stored_key() is False


@pytest.mark.parametrize("bad", ["", "   ", "gsk_short", "gsk_has space_000000000000000"])
def test_invalid_keys_rejected(memory_keyring: MemoryKeyring, bad: str) -> None:
    with pytest.raises(SecretStoreError):
        SecretStore(backend=memory_keyring, env={}).set_api_key(bad)
    assert memory_keyring.store == {}


def test_unavailable_backend_degrades_gracefully() -> None:
    store = SecretStore(backend=MemoryKeyring(fail=True), env={})
    assert store.get_api_key() is None  # read failure is not fatal
    with pytest.raises(SecretStoreError) as info:
        store.set_api_key(KEY)
    assert "GROQ_API_KEY" in info.value.user_message
    assert KEY not in str(info.value) and KEY not in info.value.user_message


def test_mask_secret() -> None:
    assert mask_secret(None) == ""
    assert mask_secret("abc") == "***"
    masked = mask_secret(KEY)
    assert masked.startswith("gsk_") and masked.endswith(KEY[-4:]) and KEY not in masked


def test_redact_patterns() -> None:
    assert "gsk_" not in redact(f"key={KEY}")
    assert redact("Authorization: Bearer abc.def.ghi") == "Authorization: Bearer [REDACTED]"
    assert redact('{"api_key": "sekret-value"}') == '{"api_key": "[REDACTED]"}'
    register_secret("custom-secret-value-123")
    assert redact("leak custom-secret-value-123 here") == "leak [REDACTED] here"


def test_log_file_never_contains_key(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "jarvis.log"
    configure_logging(log_file, console=False)
    logger = logging.getLogger("jarvis.test")
    logger.info("using key %s", KEY)
    try:
        raise RuntimeError(f"request failed with {KEY}")
    except RuntimeError:
        logger.exception("boom")
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = log_file.read_text(encoding="utf-8")
    assert "boom" in text and "RuntimeError" in text
    assert KEY not in text
    configure_logging(None, console=False)  # detach file handler so tmp dir can be removed


def test_redact_generic_secret_shapes() -> None:
    assert "sk-NOTYET" not in redact("startup key = sk-NOTYETREGISTERED-123456789")
    assert redact("password=hunter2") == "password=[REDACTED]"
    assert redact('{"token": "abc"}') == '{"token": "[REDACTED]"}'
    assert redact("usage prompt_tokens=12 total_tokens=15") == "usage prompt_tokens=12 total_tokens=15"
