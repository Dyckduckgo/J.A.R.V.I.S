"""API key storage.

Resolution order for the Groq API key:

1. ``GROQ_API_KEY`` environment variable (explicit override, useful for CI).
2. The operating-system credential store via :mod:`keyring`
   (Windows Credential Manager, macOS Keychain, Secret Service on Linux).

The key is never written to the settings file or the log.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from enum import Enum
from typing import Any

from jarvis.errors import SecretStoreError
from jarvis.logging_setup import register_secret

log = logging.getLogger(__name__)

SERVICE_NAME = "JARVIS-Desktop"
API_KEY_USERNAME = "groq_api_key"
API_KEY_ENV_VAR = "GROQ_API_KEY"


class KeySource(str, Enum):
    ENVIRONMENT = "environment"
    KEYRING = "keyring"
    NONE = "none"


class SecretStore:
    """Thin wrapper around a keyring-compatible backend.

    ``backend`` must provide ``get_password``, ``set_password`` and
    ``delete_password`` (the :mod:`keyring` module itself satisfies this).
    """

    def __init__(self, backend: Any | None = None, env: Mapping[str, str] | None = None) -> None:
        if backend is None:
            import keyring

            backend = keyring
        self._backend = backend
        self._env = env if env is not None else os.environ

    def _env_key(self) -> str | None:
        value = (self._env.get(API_KEY_ENV_VAR) or "").strip()
        return value or None

    def _keyring_key(self) -> str | None:
        try:
            value = self._backend.get_password(SERVICE_NAME, API_KEY_USERNAME)
        except Exception as exc:  # keyring raises many backend-specific types
            log.warning("Credential store unavailable while reading API key: %s", type(exc).__name__)
            return None
        return value.strip() if value and value.strip() else None

    def get_api_key(self) -> str | None:
        key = self._env_key() or self._keyring_key()
        register_secret(key)
        return key

    def has_stored_key(self) -> bool:
        """True if a key is saved in the credential store (regardless of the env var)."""
        return self._keyring_key() is not None

    def source(self) -> KeySource:
        if self._env_key():
            return KeySource.ENVIRONMENT
        if self._keyring_key():
            return KeySource.KEYRING
        return KeySource.NONE

    def set_api_key(self, key: str) -> None:
        key = (key or "").strip()
        validate_api_key_format(key)
        try:
            self._backend.set_password(SERVICE_NAME, API_KEY_USERNAME, key)
        except Exception as exc:
            raise SecretStoreError(
                f"Could not store API key in credential store: {type(exc).__name__}",
                user_message=(
                    "The operating-system credential store is not available, so the key "
                    f"could not be saved. Set the {API_KEY_ENV_VAR} environment variable instead."
                ),
            ) from exc
        register_secret(key)
        log.info("API key saved to the OS credential store")

    def delete_api_key(self) -> None:
        try:
            if self._backend.get_password(SERVICE_NAME, API_KEY_USERNAME) is None:
                return
            self._backend.delete_password(SERVICE_NAME, API_KEY_USERNAME)
        except Exception as exc:
            raise SecretStoreError(
                f"Could not remove API key from credential store: {type(exc).__name__}",
                user_message="The API key could not be removed from the credential store.",
            ) from exc
        log.info("API key removed from the OS credential store")


def validate_api_key_format(key: str) -> None:
    """Cheap sanity checks; the real validation is the provider accepting it."""
    if not key:
        raise SecretStoreError("empty key", user_message="The API key is empty.")
    if any(ch.isspace() for ch in key):
        raise SecretStoreError("whitespace in key", user_message="The API key must not contain spaces.")
    if len(key) < 20 or len(key) > 512:
        raise SecretStoreError("bad key length", user_message="That does not look like a valid API key.")


def mask_secret(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * 8}{value[-4:]}"
