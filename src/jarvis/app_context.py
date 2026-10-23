"""Composition root: builds and wires every non-GUI service.

The GUI receives a single :class:`AppContext`; tests build one with fakes for
the external boundaries (credential store, LLM client).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from jarvis.agent.agent import Agent
from jarvis.agent.service import ChatService
from jarvis.config.secrets import SecretStore
from jarvis.config.settings import Settings, SettingsManager
from jarvis.errors import MissingAPIKeyError
from jarvis.llm.base import ChatClient
from jarvis.paths import AppPaths
from jarvis.storage.database import ConversationStore
from jarvis.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class GroqClientFactory:
    """Creates (and caches) a :class:`GroqChatClient` for the current key and timeout."""

    def __init__(self, secrets: SecretStore, settings_provider: Callable[[], Settings]) -> None:
        self._secrets = secrets
        self._settings_provider = settings_provider
        self._lock = threading.Lock()
        self._cache_key: tuple[str, float] | None = None
        self._client: ChatClient | None = None

    def __call__(self) -> ChatClient:
        from jarvis.llm.groq_client import GroqChatClient

        api_key = self._secrets.get_api_key()
        if not api_key:
            raise MissingAPIKeyError()
        timeout = self._settings_provider().request_timeout_seconds
        fingerprint = (hashlib.sha256(api_key.encode()).hexdigest(), timeout)
        with self._lock:
            if self._client is None or self._cache_key != fingerprint:
                self._close_cached()
                self._client = GroqChatClient(api_key, timeout=timeout)
                self._cache_key = fingerprint
            return self._client

    def invalidate(self) -> None:
        with self._lock:
            self._close_cached()

    def close(self) -> None:
        """Close the cached client's connection pool (application shutdown only)."""
        with self._lock:
            client, self._client, self._cache_key = self._client, None, None
        if client is not None and hasattr(client, "close"):
            try:
                client.close()  # type: ignore[union-attr]
            except Exception:
                log.debug("Error closing Groq client", exc_info=True)

    def _close_cached(self) -> None:
        # Only drop the reference: a worker thread may still be streaming from
        # the old client, and closing it underneath would abort that request.
        self._client = None
        self._cache_key = None


@dataclass
class AppContext:
    paths: AppPaths
    settings: SettingsManager
    secrets: SecretStore
    store: ConversationStore
    registry: ToolRegistry
    client_factory: Callable[[], ChatClient]
    agent: Agent
    chat: ChatService
    notices: list[str] = field(default_factory=list)
    """Recovery messages (corrupt settings/database) to show the user at start-up."""

    @classmethod
    def create(
        cls,
        paths: AppPaths | None = None,
        *,
        secrets: SecretStore | None = None,
        client_factory: Callable[[], ChatClient] | None = None,
        registry: ToolRegistry | None = None,
    ) -> AppContext:
        paths = (paths or AppPaths.default()).ensure()
        settings = SettingsManager(paths.settings_file)
        settings.load()
        secrets = secrets or SecretStore()
        store = ConversationStore(paths.database_file)
        if registry is None:
            from jarvis.tools.builtin import create_default_registry

            registry = create_default_registry()
        factory = client_factory or GroqClientFactory(secrets, lambda: settings.settings)
        agent = Agent(factory, registry, lambda: settings.settings, notes=store)
        notices = [n for n in (settings.recovery_notice, store.recovery_notice) if n]
        return cls(paths, settings, secrets, store, registry, factory, agent, ChatService(store, agent), notices)

    def invalidate_client(self) -> None:
        """Call after the API key or request timeout changes."""
        invalidate = getattr(self.client_factory, "invalidate", None)
        if callable(invalidate):
            invalidate()

    def close(self) -> None:
        close = getattr(self.client_factory, "close", None)
        if callable(close):
            close()
        self.store.close()
