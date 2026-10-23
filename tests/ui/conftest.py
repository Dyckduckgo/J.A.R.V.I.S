"""Fixtures for GUI tests: a real AppContext wired to fakes at the external boundaries."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from fakes import MemoryKeyring, ScriptedChatClient
from ui_support import TEST_API_KEY, CountingFactory, EchoTool, Harness, WriteMarkerTool

from jarvis.app_context import AppContext
from jarvis.config.secrets import SecretStore
from jarvis.paths import AppPaths
from jarvis.tools.registry import ToolRegistry


@pytest.fixture
def make_harness(tmp_path: Path) -> Iterator[Callable[..., Harness]]:
    created: list[AppContext] = []

    def factory(
        *,
        with_key: bool = True,
        real_client_factory: bool = False,
        notices: Sequence[str] = (),
        script: Sequence[Any] = (),
        extra_tools: Sequence[Any] = (),
        default_registry: bool = False,
    ) -> Harness:
        keyring = MemoryKeyring()
        secrets = SecretStore(backend=keyring, env={})
        if with_key:
            secrets.set_api_key(TEST_API_KEY)
        client = ScriptedChatClient(script=list(script), chunk_size=3)
        echo, writer = EchoTool(), WriteMarkerTool()
        counting = CountingFactory(client)
        ctx = AppContext.create(
            AppPaths.under(tmp_path / f"app{len(created)}"),
            secrets=secrets,
            client_factory=None if real_client_factory else counting,
            registry=None if default_registry else ToolRegistry([echo, writer, *extra_tools]),
        )
        ctx.notices.extend(notices)
        created.append(ctx)
        return Harness(ctx, client, keyring, echo, writer, ctx.client_factory)

    yield factory
    for ctx in created:
        ctx.close()


@pytest.fixture
def make_window(qtbot) -> Callable[..., Any]:
    from jarvis.ui.main_window import MainWindow

    def factory(ctx: AppContext, **kwargs: Any) -> MainWindow:
        window = MainWindow(ctx, **kwargs)
        qtbot.addWidget(window)
        with qtbot.waitExposed(window):
            window.show()
        with qtbot.waitActive(window):  # keyboard shortcuts (Esc, Ctrl+N) need an active window
            window.activateWindow()
        return window

    return factory
