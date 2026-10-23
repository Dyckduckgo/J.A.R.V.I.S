"""End-to-end tests of the main window with a real AppContext and a scripted LLM."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox
from ui_support import send, wait_for_approval, wait_idle

from jarvis.agent.agent import CANCELLED_MESSAGE, DECLINED_MESSAGE, ERROR_PREFIX
from jarvis.errors import LLMRateLimitError, MissingAPIKeyError
from jarvis.llm.base import ChatMessage, LLMResponse, Role, ToolCall
from jarvis.ui.worker import GENERIC_ERROR_MESSAGE

pytestmark = pytest.mark.gui


def tool_call_response(call_id: str, name: str, arguments: str) -> LLMResponse:
    return LLMResponse(content="", tool_calls=(ToolCall(call_id, name, arguments),), finish_reason="tool_calls")


# --------------------------------------------------------------------------- start-up


def test_startup_shows_notices_and_no_key_prompt(qtbot, make_harness, make_window):
    notice = "Your settings file could not be read and was reset to defaults."
    h = make_harness(with_key=False, notices=[notice])
    window = make_window(h.ctx)

    assert any(b.text() == notice for b in window.banners())
    assert window.no_key_banner.isVisible()
    assert "not set" in window.key_label.text()

    # The prompt opens Settings on the API key tab.
    window.no_key_banner.action_button.click()
    dialog = window.settings_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "API key"
    dialog.reject()
    qtbot.waitUntil(lambda: window.settings_dialog is None)


def test_startup_with_key_hides_prompt_and_shows_model(qtbot, make_harness, make_window):
    h = make_harness()
    window = make_window(h.ctx)
    assert not window.no_key_banner.isVisible()
    assert window.banners() == []
    assert h.ctx.settings.settings.model in window.model_label.text()
    assert "stored" in window.key_label.text()
    assert window.windowTitle() == "J.A.R.V.I.S."


# --------------------------------------------------------------------------- chatting


def test_send_streams_text_and_persists_messages(qtbot, make_harness, make_window):
    reply = "Hello **there**! Here is a list:\n\n- one\n- two"
    h = make_harness(script=[LLMResponse(reply)])
    window = make_window(h.ctx)
    deltas: list[str] = []
    window.runner.text_received.connect(deltas.append)

    send(qtbot, window, "hi there")
    assert window.is_busy()
    assert window.input.toPlainText() == ""
    assert window.send_button.text() == "Stop"
    wait_idle(qtbot, window)

    assert len(deltas) > 3 and "".join(deltas) == reply  # streamed in chunks
    assert window.chat_view.message_texts("user") == ["hi there"]
    assert window.chat_view.message_texts("assistant") == [reply]
    assert window.send_button.text() == "Send"

    conversation_id = window.conversation_id
    assert conversation_id is not None
    stored = h.messages(conversation_id)
    assert [m.role for m in stored] == [Role.USER, Role.ASSISTANT]
    assert stored[0].content == "hi there" and stored[1].content == reply
    assert window.conversation_titles() == ["hi there"]  # auto-titled from the first message
    assert h.client.calls[0].stream is True


def test_enter_sends_but_shift_enter_inserts_newline(qtbot, make_harness, make_window):
    h = make_harness(script=[LLMResponse("ok")])
    window = make_window(h.ctx)
    window.input.setFocus()
    qtbot.keyClicks(window.input, "line one")
    qtbot.keyClick(window.input, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
    qtbot.keyClicks(window.input, "line two")
    assert window.input.toPlainText() == "line one\nline two"
    assert not window.is_busy()
    qtbot.keyClick(window.input, Qt.Key.Key_Return)
    wait_idle(qtbot, window)
    assert window.chat_view.message_texts("user") == ["line one\nline two"]


def test_safe_tool_runs_automatically_and_shows_card(qtbot, make_harness, make_window):
    h = make_harness(
        script=[tool_call_response("call_1", "echo_text", '{"text": "ping"}'), LLMResponse("The tool said ping.")]
    )
    window = make_window(h.ctx)
    approvals: list[object] = []
    window.runner.approval_requested.connect(approvals.append)

    send(qtbot, window, "please echo ping")
    wait_idle(qtbot, window)

    assert approvals == []
    assert h.echo.calls == [{"text": "ping"}]
    cards = window.chat_view.tool_cards()
    assert len(cards) == 1
    card = cards[0]
    assert card.status == "succeeded"
    assert card.decision_text == "Auto"
    assert "echo: ping" in (card.result_text or "")
    card.details_button.click()
    assert card.details.isVisible() and "echo: ping" in card.details.toPlainText()
    assert window.chat_view.message_texts("assistant") == ["The tool said ping."]
    stored = h.messages(window.conversation_id)
    assert [m.role for m in stored] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    assert stored[2].content == "echo: ping"


def test_consequential_tool_allowed_runs(qtbot, make_harness, make_window):
    h = make_harness(
        script=[
            tool_call_response("call_w", "write_marker", '{"filename": "allowed.txt", "content": "yes"}'),
            LLMResponse("Created the file."),
        ]
    )
    window = make_window(h.ctx)
    send(qtbot, window, "create a file")

    dialog = wait_for_approval(qtbot, window)
    assert dialog.request.tool_name == "write_marker"
    assert dialog.summary_view.toPlainText() == f"Create the file {h.workspace / 'allowed.txt'} with content 'yes'"
    assert '"filename": "allowed.txt"' in dialog.arguments_view.toPlainText()
    card = window.chat_view.tool_card("call_w")
    assert card is not None and card.status == "awaiting" and card.decision_text == "Needs approval"
    assert h.writer.calls == []  # nothing runs before the answer

    qtbot.mouseClick(dialog.allow_button, Qt.MouseButton.LeftButton)
    wait_idle(qtbot, window)

    assert h.writer.calls == [{"filename": "allowed.txt", "content": "yes"}]
    assert (h.workspace / "allowed.txt").read_text(encoding="utf-8") == "yes"
    assert card.status == "succeeded"
    assert window.approval_dialog is None
    tool_messages = [m for m in h.messages(window.conversation_id) if m.role is Role.TOOL]
    assert [m.content for m in tool_messages] == ["Wrote allowed.txt"]


@pytest.mark.parametrize("how", ["escape", "deny_button"])
def test_consequential_tool_denied_does_not_run(qtbot, make_harness, make_window, how):
    h = make_harness(
        script=[
            tool_call_response("call_w", "write_marker", '{"filename": "denied.txt"}'),
            LLMResponse("Understood, I did not create it."),
        ]
    )
    window = make_window(h.ctx)
    send(qtbot, window, "create a file")
    dialog = wait_for_approval(qtbot, window)

    if how == "escape":
        qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    else:
        qtbot.mouseClick(dialog.deny_button, Qt.MouseButton.LeftButton)
    wait_idle(qtbot, window)

    assert h.writer.calls == []
    assert not (h.workspace / "denied.txt").exists()
    tool_messages = [m for m in h.messages(window.conversation_id) if m.role is Role.TOOL]
    assert [m.content for m in tool_messages] == [ERROR_PREFIX + DECLINED_MESSAGE]
    assert window.chat_view.tool_card("call_w").status == "declined"
    assert window.chat_view.message_texts("assistant") == ["Understood, I did not create it."]


def test_stop_during_pending_approval_denies_and_finishes(qtbot, make_harness, make_window):
    h = make_harness(script=[tool_call_response("call_w", "write_marker", '{"filename": "stopped.txt"}')])
    window = make_window(h.ctx)
    send(qtbot, window, "create a file")
    wait_for_approval(qtbot, window)

    # While the turn runs the active conversation cannot be switched or deleted.
    assert not window.conversation_list.isEnabled()
    assert not window.new_chat_button.isEnabled()
    assert window.delete_conversation(window.conversation_id) is False

    window.send_button.click()  # the Send button is "Stop" while busy
    wait_idle(qtbot, window)

    assert window.approval_dialog is None
    assert h.writer.calls == []
    assert not (h.workspace / "stopped.txt").exists()
    tool_messages = [m for m in h.messages(window.conversation_id) if m.role is Role.TOOL]
    assert [m.content for m in tool_messages] == [ERROR_PREFIX + CANCELLED_MESSAGE]
    assert window.chat_view.tool_card("call_w").status == "cancelled"
    assert "Stopped." in window.chat_view.message_texts("notice")
    assert len(h.client.calls) == 1  # no further model call after Stop
    assert window.conversation_list.isEnabled()


def test_escape_key_stops_streaming_turn(qtbot, make_harness, make_window):
    import threading

    release = threading.Event()

    def slow_reply(call):
        release.wait(5)
        return LLMResponse("too late")

    h = make_harness(script=[slow_reply])
    window = make_window(h.ctx)
    send(qtbot, window, "take your time")
    assert window.is_busy()
    qtbot.keyClick(window, Qt.Key.Key_Escape)
    release.set()
    wait_idle(qtbot, window)
    # The scripted client honours cancellation before streaming, so nothing is shown or stored.
    assert window.chat_view.message_texts("assistant") == []
    assert [m.role for m in h.messages(window.conversation_id)] == [Role.USER]


def test_llm_error_shows_friendly_message_and_keeps_input(qtbot, make_harness, make_window):
    friendly = "Groq's rate limit was reached. Please wait a moment and try again."
    h = make_harness(script=[LLMRateLimitError("HTTP 429 raw body xyz", user_message=friendly)])
    window = make_window(h.ctx)
    send(qtbot, window, "hello")
    wait_idle(qtbot, window)

    assert window.chat_view.message_texts("error") == [friendly]
    assert all("raw body" not in t for t in window.chat_view.message_texts())
    assert window.input.toPlainText() == "hello"
    assert window.send_button.text() == "Send"


def test_missing_api_key_path(qtbot, make_harness, make_window):
    h = make_harness(with_key=False, real_client_factory=True)
    window = make_window(h.ctx)
    send(qtbot, window, "hello?")
    wait_idle(qtbot, window)

    assert window.chat_view.message_texts("error") == [MissingAPIKeyError().user_message]
    assert window.no_key_banner.isVisible()
    assert window.input.toPlainText() == "hello?"
    assert h.messages(window.conversation_id) == []  # nothing persisted before the key check


def test_unexpected_error_shows_generic_message(qtbot, make_harness, make_window):
    h = make_harness(script=[RuntimeError("internal detail 42")])
    window = make_window(h.ctx)
    send(qtbot, window, "hello")
    wait_idle(qtbot, window)
    assert window.chat_view.message_texts("error") == [GENERIC_ERROR_MESSAGE]
    assert all("internal detail" not in t for t in window.chat_view.message_texts())


def test_model_html_is_not_rendered(qtbot, make_harness, make_window):
    h = make_harness(script=[LLMResponse("<b>bold?</b> <img src='file:///C:/x.png'> [x](javascript:alert(1))")])
    window = make_window(h.ctx)
    send(qtbot, window, "html please")
    wait_idle(qtbot, window)
    bubble = [b for b in window.chat_view.bubbles() if b.kind == "assistant"][0]
    plain = bubble.text_view.toPlainText()
    assert "<b>bold?</b>" in plain and "<img" in plain


# --------------------------------------------------------------------------- conversations


def test_conversation_new_rename_delete(qtbot, make_harness, make_window, monkeypatch):
    h = make_harness(script=[LLMResponse("first answer"), LLMResponse("second answer")])
    window = make_window(h.ctx)
    send(qtbot, window, "first question")
    wait_idle(qtbot, window)
    first_id = window.conversation_id

    window.new_chat_action.trigger()
    assert window.conversation_id is None
    assert window.chat_view.bubbles() == []
    send(qtbot, window, "second question")
    wait_idle(qtbot, window)
    second_id = window.conversation_id
    assert second_id != first_id
    assert window.conversation_titles() == ["second question", "first question"]

    # Rename
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Renamed chat", True)))
    assert window.rename_conversation(first_id) is True
    assert h.ctx.store.get_conversation(first_id).title == "Renamed chat"
    assert "Renamed chat" in window.conversation_titles()

    # Delete: "No" keeps it; the default button must be No.
    asked: list[tuple] = []

    def answer(value):
        def question(*args, **kwargs):
            asked.append(args)
            return value

        return staticmethod(question)

    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.No))
    assert window.delete_conversation(second_id) is False
    assert h.ctx.store.get_conversation(second_id) is not None
    assert asked[-1][4] == QMessageBox.StandardButton.No  # default button

    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.Yes))
    assert window.delete_conversation(second_id) is True
    assert h.ctx.store.get_conversation(second_id) is None
    assert window.conversation_titles() == ["Renamed chat"]
    # The deleted conversation was active, so the remaining one is shown.
    assert window.conversation_id == first_id
    assert window.chat_view.message_texts("user") == ["first question"]


def test_switching_conversations_and_search(qtbot, make_harness, make_window):
    h = make_harness()
    store = h.ctx.store
    alpha = store.create_conversation("Alpha plans")
    store.append_message(alpha.id, ChatMessage.user("talk about alpha"))
    beta = store.create_conversation("Beta notes")
    store.append_message(beta.id, ChatMessage.user("beta content"))
    window = make_window(h.ctx)

    assert window.conversation_id == beta.id  # most recent opens at start-up
    item = window._item_for(alpha.id)
    window.conversation_list.setCurrentItem(item)
    assert window.conversation_id == alpha.id
    assert window.chat_view.message_texts("user") == ["talk about alpha"]

    qtbot.keyClicks(window.search_edit, "beta")
    qtbot.waitUntil(lambda: window.conversation_titles() == ["Beta notes"])
    window.search_edit.clear()
    qtbot.waitUntil(lambda: len(window.conversation_titles()) == 2)


def test_reloaded_conversation_renders_tool_cards(qtbot, make_harness, make_window):
    h = make_harness()
    store = h.ctx.store
    conv = store.create_conversation("Stored chat")
    echo_call = ToolCall("c1", "echo_text", '{"text": "stored"}')
    write_call = ToolCall("c2", "write_marker", '{"filename": "x.txt"}')
    gone_call = ToolCall("c3", "removed_tool", "{}")
    for message in [
        ChatMessage.user("do things"),
        ChatMessage.assistant("Working on it.", [echo_call, write_call, gone_call]),
        ChatMessage.tool(echo_call, "echo: stored"),
        ChatMessage.tool(write_call, ERROR_PREFIX + DECLINED_MESSAGE),
        ChatMessage.tool(gone_call, ERROR_PREFIX + "Unknown tool 'removed_tool'."),
        ChatMessage.assistant("Done, <script>x</script>."),
    ]:
        store.append_message(conv.id, message)

    window = make_window(h.ctx)
    assert window.conversation_id == conv.id
    view = window.chat_view
    assert view.message_texts("user") == ["do things"]
    assert view.message_texts("assistant") == ["Working on it.", "Done, <script>x</script>."]
    cards = {c.call.id: c for c in view.tool_cards()}
    assert set(cards) == {"c1", "c2", "c3"}
    assert (cards["c1"].status, cards["c1"].decision_text) == ("succeeded", "Auto")
    assert (cards["c2"].status, cards["c2"].decision_text) == ("declined", "Needs approval")
    # Stored calls are described with a ToolContext, so relative paths render as absolute.
    assert cards["c2"].summary == f"Create the file {h.workspace / 'x.txt'} with content 'marker'"
    assert (cards["c3"].status, cards["c3"].decision_text) == ("blocked", "Denied")
    assert cards["c1"].result_text == "echo: stored"


def test_export_conversation(qtbot, make_harness, make_window, monkeypatch, tmp_path: Path):
    h = make_harness(script=[LLMResponse("exported answer")])
    window = make_window(h.ctx)
    send(qtbot, window, "export me")
    wait_idle(qtbot, window)
    target = tmp_path / "export.md"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(target), "")))
    assert window.export_conversation() == target
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# export me") and "exported answer" in text


def test_close_while_waiting_for_approval_denies(qtbot, make_harness, make_window):
    h = make_harness(script=[tool_call_response("call_w", "write_marker", '{"filename": "closing.txt"}')])
    window = make_window(h.ctx)
    send(qtbot, window, "create a file")
    wait_for_approval(qtbot, window)

    window.close()

    assert window.runner.wait_for_workers(5.0)  # the worker finished; no deadlock
    assert h.writer.calls == []
    assert not (h.workspace / "closing.txt").exists()


def test_keyboard_shortcuts(qtbot, make_harness, make_window):
    h = make_harness(script=[LLMResponse("answer")])
    window = make_window(h.ctx)
    send(qtbot, window, "question")
    wait_idle(qtbot, window)
    assert window.conversation_id is not None

    qtbot.keyClick(window, Qt.Key.Key_N, Qt.KeyboardModifier.ControlModifier)
    assert window.conversation_id is None and window.chat_view.bubbles() == []

    qtbot.keyClick(window, Qt.Key.Key_Comma, Qt.KeyboardModifier.ControlModifier)
    dialog = window.settings_dialog
    assert dialog is not None and dialog.isVisible()
    dialog.reject()
    qtbot.waitUntil(lambda: window.settings_dialog is None)


def test_theme_change_from_settings_is_applied(qtbot, make_harness, make_window):
    from PySide6.QtWidgets import QApplication

    from jarvis.ui.theme import DARK, LIGHT

    h = make_harness(script=[LLMResponse("Some *markdown* with [a link](https://example.com)")])
    window = make_window(h.ctx)
    send(qtbot, window, "hello")
    wait_idle(qtbot, window)
    assert DARK.window in QApplication.instance().styleSheet()

    dialog = window.open_settings()
    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData("light"))
    dialog.save_button.click()
    qtbot.waitUntil(lambda: window.settings_dialog is None)
    assert LIGHT.window in QApplication.instance().styleSheet()
    assert h.ctx.settings.settings.theme.value == "light"
    assert window.chat_view.message_texts("assistant") == ["Some *markdown* with [a link](https://example.com)"]


def test_stored_call_with_failing_describe_falls_back(qtbot, make_harness, make_window):
    from jarvis.tools.base import Tool, ToolResult

    class FragileTool(Tool):
        name = "fragile"
        description = "describe_call breaks on old arguments"
        parameters = {"type": "object", "properties": {"x": {"type": "string"}}}

        def describe_call(self, args, ctx=None):
            raise KeyError("old schema")

        def run(self, args, ctx):
            return ToolResult.success("ran")

    h = make_harness(extra_tools=[FragileTool()])
    conv = h.ctx.store.create_conversation("Old chat")
    call = ToolCall("f1", "fragile", '{"x": "1"}')
    for message in [ChatMessage.user("go"), ChatMessage.assistant(None, [call]), ChatMessage.tool(call, "ran")]:
        h.ctx.store.append_message(conv.id, message)
    window = make_window(h.ctx)
    card = window.chat_view.tool_card("f1")
    assert card is not None
    assert card.summary == 'fragile({"x": "1"})'
    assert (card.status, card.decision_text) == ("succeeded", "Auto")


def test_window_with_real_default_registry(qtbot, make_harness, make_window):
    h = make_harness(default_registry=True)
    names = h.ctx.registry.names()
    assert "read_file" in names and "write_file" in names
    conv = h.ctx.store.create_conversation("Files")
    call = ToolCall("r1", "read_file", '{"path": "notes.txt"}')
    for message in [ChatMessage.user("read it"), ChatMessage.assistant(None, [call]), ChatMessage.tool(call, "hi")]:
        h.ctx.store.append_message(conv.id, message)
    window = make_window(h.ctx)
    card = window.chat_view.tool_card("r1")
    assert card is not None and card.status == "succeeded"
    assert str(h.workspace / "notes.txt") in card.summary  # described with a ToolContext: absolute path
    dialog = window.open_settings(tab="Tools")
    assert dialog.tools_tree.topLevelItemCount() == len(names)
    dialog.reject()
