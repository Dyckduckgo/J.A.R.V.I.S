"""Worker threading/approval semantics and chat view rendering."""

from __future__ import annotations

import threading

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

from jarvis.agent.agent import CANCELLED_MESSAGE, DECLINED_MESSAGE, ERROR_PREFIX
from jarvis.agent.events import ApprovalRequest
from jarvis.llm.base import ChatMessage, LLMResponse, Role, ToolCall
from jarvis.tools.base import RiskLevel, ToolResult
from jarvis.tools.policy import Decision
from jarvis.ui import chat_view as chat_view_module
from jarvis.ui.chat_view import ChatView, classify_result, open_external_link
from jarvis.ui.worker import PendingApproval, TurnRunner

pytestmark = pytest.mark.gui


def approval_request() -> ApprovalRequest:
    call = ToolCall("c", "write_marker", "{}")
    return ApprovalRequest(call, "write_marker", "summary", RiskLevel.CONSEQUENTIAL, "reason", {})


# --------------------------------------------------------------------------- PendingApproval


def test_pending_approval_first_answer_wins():
    pending = PendingApproval(approval_request())
    assert pending.resolve(False, by="cancel") is True
    assert pending.resolve(True) is False  # a late "allow" cannot override Stop
    assert pending.approved is False and pending.resolved_by == "cancel"


def test_pending_approval_wait_returns_false_on_cancel_and_timeout():
    cancel = threading.Event()
    pending = PendingApproval(approval_request())
    threading.Timer(0.1, cancel.set).start()
    assert pending.wait(cancel, timeout=10) is False
    assert pending.resolved_by == "cancel"

    pending = PendingApproval(approval_request())
    assert pending.wait(threading.Event(), timeout=0.15) is False
    assert pending.resolved_by == "timeout"

    pending = PendingApproval(approval_request())
    threading.Timer(0.05, lambda: pending.resolve(True)).start()
    assert pending.wait(threading.Event(), timeout=10) is True


# --------------------------------------------------------------------------- TurnRunner


def test_runner_denies_when_no_answer_arrives(qtbot, make_harness):
    h = make_harness(
        script=[
            LLMResponse("", tool_calls=(ToolCall("w1", "write_marker", '{"filename": "t.txt"}'),)),
            LLMResponse("ok, not done"),
        ]
    )
    conversation = h.ctx.store.create_conversation()
    runner = TurnRunner(h.ctx.chat, approval_timeout=0.3)
    requested: list[PendingApproval] = []
    closed: list[PendingApproval] = []
    outcomes: list[object] = []
    runner.approval_requested.connect(requested.append)  # nobody answers
    runner.approval_closed.connect(closed.append)
    runner.turn_finished.connect(outcomes.append)

    assert runner.start(conversation.id, "write it") is True
    assert runner.start(conversation.id, "second turn") is False  # one turn at a time
    qtbot.waitUntil(lambda: bool(outcomes), timeout=8000)

    assert len(requested) == 1 and closed == requested
    assert requested[0].resolved_by == "timeout"
    assert h.writer.calls == []
    tool_messages = [m for m in h.ctx.store.get_messages(conversation.id) if m.role is Role.TOOL]
    assert [m.content for m in tool_messages] == [ERROR_PREFIX + DECLINED_MESSAGE]
    assert outcomes[0].ok and not runner.is_busy()


def test_runner_signals_arrive_on_gui_thread(qtbot, make_harness):
    h = make_harness(script=[LLMResponse("threaded reply")])
    conversation = h.ctx.store.create_conversation()
    runner = TurnRunner(h.ctx.chat)
    seen_threads: set[int] = set()
    record = lambda *_: seen_threads.add(threading.get_ident())  # noqa: E731
    for signal in (runner.message_received, runner.text_received, runner.llm_started, runner.turn_finished):
        signal.connect(record)
    with qtbot.waitSignal(runner.turn_finished, timeout=8000):
        runner.start(conversation.id, "hello")
    assert seen_threads == {threading.get_ident()}


def test_runner_shutdown_cancels_pending_approval(qtbot, make_harness):
    h = make_harness(script=[LLMResponse("", tool_calls=(ToolCall("w1", "write_marker", '{"filename": "s.txt"}'),))])
    conversation = h.ctx.store.create_conversation()
    runner = TurnRunner(h.ctx.chat)
    requested: list[PendingApproval] = []
    runner.approval_requested.connect(requested.append)
    runner.start(conversation.id, "write it")
    qtbot.waitUntil(lambda: bool(requested), timeout=8000)
    assert runner.shutdown(5.0) is True
    assert requested[0].approved is False
    assert h.writer.calls == []
    tool_messages = [m for m in h.ctx.store.get_messages(conversation.id) if m.role is Role.TOOL]
    assert [m.content for m in tool_messages] == [ERROR_PREFIX + CANCELLED_MESSAGE]


# --------------------------------------------------------------------------- chat view


def test_classify_result():
    assert classify_result(True, "fine") == "succeeded"
    assert classify_result(False, DECLINED_MESSAGE) == "declined"
    assert classify_result(False, ERROR_PREFIX + DECLINED_MESSAGE) == "declined"
    assert classify_result(False, CANCELLED_MESSAGE) == "cancelled"
    assert classify_result(False, "The tool 'x' is disabled in Settings.", Decision.DENY) == "blocked"
    assert classify_result(False, "disk full") == "failed"


def test_links_open_only_for_safe_schemes(monkeypatch):
    opened: list[str] = []
    monkeypatch.setattr(QDesktopServices, "openUrl", staticmethod(lambda url: opened.append(url.toString()) or True))
    assert open_external_link(QUrl("https://example.com/a")) is True
    assert open_external_link(QUrl("mailto:someone@example.com")) is True
    for bad in ("javascript:alert(1)", "file:///C:/Windows/System32/calc.exe", "ms-settings:privacy", "C:/x.exe"):
        assert open_external_link(QUrl(bad)) is False
    assert opened == ["https://example.com/a", "mailto:someone@example.com"]


def test_clicking_a_link_in_a_bubble_goes_through_the_safe_opener(qtbot, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(chat_view_module, "open_external_link", lambda url: calls.append(url.toString()))
    view = ChatView()
    qtbot.addWidget(view)
    bubble = view.add_assistant_text("see [docs](https://example.com)")
    bubble.text_view.anchorClicked.emit(QUrl("https://example.com"))
    assert calls == ["https://example.com"]
    assert bubble.text_view.source().isEmpty()  # the browser itself never navigates
    assert not bubble.text_view.openLinks() and not bubble.text_view.openExternalLinks()


def test_streaming_is_throttled_and_finalised(qtbot):
    view = ChatView()
    qtbot.addWidget(view)
    view.show()
    view.begin_model_step()
    for chunk in ["Hel", "lo ", "**wor", "ld**"]:
        view.append_stream_text(chunk)
    bubble = view.bubbles()[0]
    assert bubble.raw_text == ""  # not re-rendered per token
    qtbot.waitUntil(lambda: bubble.raw_text == "Hello **world**", timeout=2000)
    assert "Hello world" in bubble.text_view.toPlainText()

    view.add_assistant_message(ChatMessage.assistant("Hello **world**!"))
    assert view.message_texts("assistant") == ["Hello **world**!"]
    assert not view.is_streaming()


def test_assistant_message_with_only_tool_calls_removes_empty_stream_bubble(qtbot):
    view = ChatView()
    qtbot.addWidget(view)
    view.begin_model_step()
    view.append_stream_text("  ")
    view.add_assistant_message(ChatMessage.assistant(None, [ToolCall("t", "echo_text", "{}")]))
    assert view.bubbles() == []
    card = view.tool_started(ToolCall("t", "echo_text", "{}"), "echo_text()", Decision.AUTO)
    view.tool_finished(ToolCall("t", "echo_text", "{}"), ToolResult.failure("boom"))
    assert card.status == "failed" and card.status_text == "Failed"


def test_synthetic_notice_without_deltas_is_shown(qtbot):
    view = ChatView()
    qtbot.addWidget(view)
    view.begin_model_step()
    view.add_assistant_message(ChatMessage.assistant("I stopped after 8 tool steps without finishing."))
    assert view.message_texts("assistant") == ["I stopped after 8 tool steps without finishing."]


def test_untrusted_html_is_escaped(qtbot):
    view = ChatView()
    qtbot.addWidget(view)
    bubble = view.add_assistant_text("<b>x</b><script>alert(1)</script><img src=x onerror=y>")
    text = bubble.text_view.toPlainText()
    assert "<b>x</b>" in text and "<script>" in text and "<img" in text
    user = view.add_user_message("<i>me</i>")
    assert user.text_view.toPlainText() == "<i>me</i>"


def test_autoscroll_only_when_at_bottom(qtbot):
    view = ChatView()
    qtbot.addWidget(view)
    view.resize(500, 300)
    with qtbot.waitExposed(view):
        view.show()
    for i in range(30):
        view.add_assistant_text(f"Message number {i}\n\nwith a second paragraph")
    bar = view.verticalScrollBar()
    qtbot.waitUntil(lambda: bar.maximum() > 0 and bar.value() == bar.maximum(), timeout=3000)

    bar.setValue(0)  # the user scrolls up to read
    previous_max = bar.maximum()
    view.add_assistant_text("new message while reading")
    qtbot.waitUntil(lambda: bar.maximum() > previous_max, timeout=3000)
    assert bar.value() == 0

    view.scroll_to_bottom()
    view.add_assistant_text("another message")
    qtbot.waitUntil(lambda: bar.value() == bar.maximum(), timeout=3000)
