"""Scrollable chat transcript: message bubbles and tool cards.

Security: model and tool output is untrusted. Markdown is parsed with
``MarkdownNoHTML`` so embedded HTML is shown as literal text, plain text is set
with ``setPlainText``, external resources (images) are never loaded, and links
only open on an explicit click, through :func:`open_external_link`, for
``http``/``https``/``mailto`` URLs.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from PySide6.QtCore import QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QTextCharFormat, QTextCursor, QTextDocument, QTextOption
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from jarvis import APP_DISPLAY_NAME
from jarvis.agent.agent import CANCELLED_MESSAGE, DECLINED_MESSAGE, ERROR_PREFIX
from jarvis.llm.base import ChatMessage, Role, ToolCall
from jarvis.tools.base import ToolResult
from jarvis.tools.policy import Decision
from jarvis.ui.theme import ThemeColors, current_colors, set_tone

log = logging.getLogger(__name__)

SAFE_URL_SCHEMES = frozenset({"http", "https", "mailto"})
STREAM_RENDER_INTERVAL_MS = 80
MONOSPACE_FAMILIES = ["Cascadia Mono", "Consolas", "Courier New", "monospace"]
_MARKDOWN_FEATURES = (
    QTextDocument.MarkdownFeature.MarkdownDialectGitHub | QTextDocument.MarkdownFeature.MarkdownNoHTML
)

# Tool card states -> (label, tone)
STATUS_LABELS: dict[str, tuple[str, str]] = {
    "running": ("Running…", "info"),
    "awaiting": ("Awaiting approval", "warning"),
    "succeeded": ("Succeeded", "success"),
    "failed": ("Failed", "danger"),
    "declined": ("Declined", "warning"),
    "cancelled": ("Cancelled", "neutral"),
    "blocked": ("Blocked", "danger"),
    "no_result": ("No result", "neutral"),
}
DECISION_LABELS: dict[Decision, tuple[str, str]] = {
    Decision.AUTO: ("Auto", "info"),
    Decision.CONFIRM: ("Needs approval", "warning"),
    Decision.DENY: ("Denied", "danger"),
}
_CARD_TONES = {"succeeded": "success", "failed": "danger", "blocked": "danger", "declined": "warning",
               "awaiting": "warning"}

ToolDescriber = Callable[[ToolCall], tuple[str, Decision | None]]


def open_external_link(url: QUrl) -> bool:
    """Open ``url`` in the system browser if (and only if) its scheme is safe."""
    scheme = url.scheme().lower()
    if scheme not in SAFE_URL_SCHEMES or not url.isValid():
        log.info("Refused to open link with scheme %r", scheme)
        return False
    return bool(QDesktopServices.openUrl(url))


def classify_result(ok: bool, content: str, decision: Decision | None = None) -> str:
    """Map a tool result to a card status."""
    if ok:
        return "succeeded"
    text = content[len(ERROR_PREFIX):] if content.startswith(ERROR_PREFIX) else content
    if text.strip() == DECLINED_MESSAGE:
        return "declined"
    if text.startswith(CANCELLED_MESSAGE) or text.startswith("Cancelled"):
        return "cancelled"
    if decision is Decision.DENY:
        return "blocked"
    return "failed"


def render_markdown(document: QTextDocument, text: str, colors: ThemeColors) -> None:
    """Parse ``text`` as Markdown (raw HTML disabled) and apply theme styling."""
    document.setMarkdown(text, _MARKDOWN_FEATURES)
    code_bg = QColor(colors.code_bg)
    link = QColor(colors.link)
    # One edit block for all format changes: without it every merge triggers an
    # incremental re-layout of a widget-attached document (~10x slower on long text).
    edit = QTextCursor(document)
    edit.beginEditBlock()
    try:
        _style_fragments(document, code_bg, link)
    finally:
        edit.endEditBlock()


def _style_fragments(document: QTextDocument, code_bg: QColor, link: QColor) -> None:
    block = document.begin()
    while block.isValid():
        block_format = block.blockFormat()
        is_code_block = block_format.nonBreakableLines()
        if is_code_block:
            # Wrap long code lines instead of growing the bubble sideways.
            block_format.setNonBreakableLines(False)
            block_format.setBackground(code_bg)
            QTextCursor(block).setBlockFormat(block_format)
        it = block.begin()
        while not it.atEnd():
            fragment = it.fragment()
            if fragment.isValid():
                fmt = fragment.charFormat()
                patch = QTextCharFormat()
                changed = False
                if fmt.isAnchor():
                    patch.setForeground(link)
                    patch.setFontUnderline(True)
                    changed = True
                families = fmt.fontFamilies() or []
                if fmt.fontFixedPitch() or is_code_block or "monospace" in families:
                    patch.setFontFamilies(MONOSPACE_FAMILIES)
                    if not is_code_block:
                        patch.setBackground(code_bg)
                    changed = True
                if changed:
                    cursor = QTextCursor(document)
                    cursor.setPosition(fragment.position())
                    cursor.setPosition(fragment.position() + fragment.length(), QTextCursor.MoveMode.KeepAnchor)
                    cursor.mergeCharFormat(patch)
            it += 1
        block = block.next()


class MessageText(QTextBrowser):
    """Read-only, selectable text that grows to fit its content (no inner scrolling)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("MessageText")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        self.setWordWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(40)
        self.viewport().setAutoFillBackground(False)
        self.document().setDocumentMargin(2)
        self.anchorClicked.connect(self._open_link)
        self.document().documentLayout().documentSizeChanged.connect(self._sync_height)
        self._sync_height()

    def _open_link(self, url: QUrl) -> None:
        open_external_link(url)

    # Never fetch images or other resources referenced by untrusted text.
    def loadResource(self, type: int, name: QUrl) -> Any:  # noqa: A002 - Qt signature
        return None

    def wheelEvent(self, event: Any) -> None:
        event.ignore()  # let the surrounding chat scroll area scroll

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._sync_height()

    def _sync_height(self, *_: Any) -> None:
        height = math.ceil(self.document().size().height()) + 2 * self.frameWidth()
        margins = self.contentsMargins()
        height += margins.top() + margins.bottom()
        if self.height() != height or self.minimumHeight() != height:
            self.setFixedHeight(max(height, 18))

    def natural_width(self) -> int:
        """Width needed to show the content without wrapping."""
        clone = self.document().clone(self)
        clone.setTextWidth(-1)
        width = math.ceil(clone.idealWidth()) + 2 * self.frameWidth() + 6
        clone.deleteLater()
        return width


class MessageBubble(QFrame):
    """One chat message (user, assistant, error or notice)."""

    OBJECT_NAMES = {"user": "UserBubble", "assistant": "AssistantBubble", "error": "ErrorBubble",
                    "notice": "NoticeBubble"}

    def __init__(self, kind: str, author: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.setObjectName(self.OBJECT_NAMES[kind])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 7, 12, 8)
        layout.setSpacing(2)
        self.author_label = QLabel(author)
        self.author_label.setObjectName("BubbleAuthor")
        self.author_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.author_label)
        self.text_view = MessageText(self)
        layout.addWidget(self.text_view)
        self._raw = ""
        self._markdown = False

    @property
    def raw_text(self) -> str:
        return self._raw

    def set_markdown(self, text: str) -> None:
        self._raw = text
        self._markdown = True
        render_markdown(self.text_view.document(), text, current_colors())
        self._after_content_change()

    def set_plain(self, text: str) -> None:
        self._raw = text
        self._markdown = False
        self.text_view.setPlainText(text)
        self._after_content_change()

    def rerender(self) -> None:
        if self._markdown:
            self.set_markdown(self._raw)

    def _after_content_change(self) -> None:
        if self.kind == "user":
            # Hug short messages; long ones wrap at the available width.
            margins = self.layout().contentsMargins()
            width = max(self.text_view.natural_width(), self.author_label.sizeHint().width())
            self.setMaximumWidth(width + margins.left() + margins.right() + 4)


class Badge(QLabel):
    def __init__(self, text: str = "", tone: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setObjectName("Badge")
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setProperty("tone", tone)

    def set(self, text: str, tone: str) -> None:
        self.setText(text)
        set_tone(self, tone)


class ToolCard(QFrame):
    """Shows one tool call: summary, policy decision, status and a result preview."""

    def __init__(
        self,
        call: ToolCall,
        summary: str,
        decision: Decision | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ToolCard")
        self.call = call
        self.summary = summary
        self.decision: Decision | None = None
        self.status = "running"
        self.result_text: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.name_label = QLabel(f"Tool · {call.name}")
        self.name_label.setObjectName("ToolName")
        self.name_label.setTextFormat(Qt.TextFormat.PlainText)
        header.addWidget(self.name_label)
        self.decision_badge = Badge()
        header.addWidget(self.decision_badge)
        self.status_badge = Badge()
        header.addWidget(self.status_badge)
        header.addStretch(1)
        self.details_button = QToolButton()
        self.details_button.setObjectName("DisclosureButton")
        self.details_button.setCheckable(True)
        self.details_button.setText("Details ▸")
        self.details_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.details_button.toggled.connect(self._toggle_details)
        header.addWidget(self.details_button)
        layout.addLayout(header)

        self.summary_view = MessageText(self)
        self.summary_view.setPlainText(summary)
        layout.addWidget(self.summary_view)

        self.details = QPlainTextEdit()
        self.details.setObjectName("ToolDetails")
        self.details.setReadOnly(True)
        self.details.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.details.setMaximumHeight(240)
        self.details.setVisible(False)
        layout.addWidget(self.details)

        self.set_decision(decision)
        self.set_status("running" if decision is not Decision.CONFIRM else "awaiting")
        self._refresh_details()

    # -- state -------------------------------------------------------------------

    def set_decision(self, decision: Decision | None) -> None:
        self.decision = decision
        if decision is None:
            self.decision_badge.setVisible(False)
            return
        text, tone = DECISION_LABELS.get(decision, (str(decision), "neutral"))
        self.decision_badge.set(text, tone)
        self.decision_badge.setVisible(True)

    def set_status(self, status: str) -> None:
        self.status = status
        text, tone = STATUS_LABELS.get(status, (status, "neutral"))
        self.status_badge.set(text, tone)
        set_tone(self, _CARD_TONES.get(status, "neutral"))

    def set_result(self, ok: bool, content: str) -> None:
        self.result_text = content
        self.set_status(classify_result(ok, content, self.decision))
        self._refresh_details()

    @property
    def decision_text(self) -> str:
        return self.decision_badge.text() if self.decision is not None else ""

    @property
    def status_text(self) -> str:
        return self.status_badge.text()

    def _refresh_details(self) -> None:
        try:
            pretty_args = json.dumps(self.call.parsed_arguments(), indent=2, ensure_ascii=False)
        except Exception:
            pretty_args = self.call.arguments or "{}"
        parts = ["Arguments:", pretty_args]
        if self.result_text is not None:
            parts += ["", "Result:", self.result_text]
        self.details.setPlainText("\n".join(parts))

    def _toggle_details(self, expanded: bool) -> None:
        self.details.setVisible(expanded)
        self.details_button.setText("Details ▾" if expanded else "Details ▸")


class ChatView(QScrollArea):
    """The conversation transcript."""

    content_changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ChatView")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self._container = QWidget()
        self._container.setObjectName("ChatContainer")
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(20, 16, 20, 16)
        self._layout.setSpacing(10)
        self._empty_label = QLabel(f"{APP_DISPLAY_NAME}\nHow can I help you today?")
        self._empty_label.setObjectName("EmptyState")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._empty_label, 1)
        self._layout.addStretch(0)
        self.setWidget(self._container)

        self._rows: list[QWidget] = []
        self._bubbles: list[MessageBubble] = []
        self._cards: dict[str, ToolCard] = {}
        self._stream_bubble: MessageBubble | None = None
        self._stream_chunks: list[str] = []
        self._stream_dirty = False
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(STREAM_RENDER_INTERVAL_MS)
        self._render_timer.timeout.connect(self._flush_stream)

        self._stick_to_bottom = True
        bar = self.verticalScrollBar()
        bar.valueChanged.connect(self._on_scroll_value)
        bar.rangeChanged.connect(self._on_scroll_range)

    # -- introspection (used by the window and tests) ---------------------------

    def bubbles(self) -> list[MessageBubble]:
        return list(self._bubbles)

    def message_texts(self, kind: str | None = None) -> list[str]:
        return [b.raw_text for b in self._bubbles if kind is None or b.kind == kind]

    def tool_cards(self) -> list[ToolCard]:
        return list(self._cards.values())

    def tool_card(self, call_id: str) -> ToolCard | None:
        return self._cards.get(call_id)

    def is_streaming(self) -> bool:
        return self._stream_bubble is not None

    # -- building blocks -----------------------------------------------------------

    def clear(self) -> None:
        self._render_timer.stop()
        self._stream_bubble = None
        self._stream_chunks = []
        for row in self._rows:
            self._layout.removeWidget(row)
            row.deleteLater()
        self._rows.clear()
        self._bubbles.clear()
        self._cards.clear()
        self._empty_label.setVisible(True)
        self._stick_to_bottom = True

    def _add_row(self, widget: QWidget, align: str) -> QWidget:
        row = QWidget()
        hbox = QHBoxLayout(row)
        hbox.setSpacing(0)
        if align == "right":
            hbox.setContentsMargins(64, 0, 0, 0)
            hbox.addStretch(1)
            hbox.addWidget(widget, 100)
        else:
            hbox.setContentsMargins(0, 0, 32, 0)
            hbox.addWidget(widget, 100)
            hbox.addStretch(1)
        self._layout.insertWidget(self._layout.count() - 1, row)
        row.show()  # layouts otherwise show new children asynchronously
        self._rows.append(row)
        self._empty_label.setVisible(False)
        self.content_changed.emit()
        return row

    def _remove_widget_row(self, widget: QWidget) -> None:
        for row in list(self._rows):
            if row.isAncestorOf(widget):
                self._rows.remove(row)
                self._layout.removeWidget(row)
                row.deleteLater()
        if isinstance(widget, MessageBubble) and widget in self._bubbles:
            self._bubbles.remove(widget)
        self._empty_label.setVisible(not self._rows)

    def _new_bubble(self, kind: str, author: str) -> MessageBubble:
        bubble = MessageBubble(kind, author)
        self._bubbles.append(bubble)
        self._add_row(bubble, "right" if kind == "user" else "left")
        return bubble

    def add_user_message(self, text: str) -> MessageBubble:
        bubble = self._new_bubble("user", "You")
        bubble.set_plain(text)
        self.scroll_to_bottom()
        return bubble

    def add_assistant_text(self, text: str) -> MessageBubble:
        bubble = self._new_bubble("assistant", APP_DISPLAY_NAME)
        bubble.set_markdown(text)
        return bubble

    def add_error(self, text: str) -> MessageBubble:
        bubble = self._new_bubble("error", "Error")
        bubble.set_plain(text)
        self.scroll_to_bottom()
        return bubble

    def add_notice(self, text: str) -> MessageBubble:
        bubble = self._new_bubble("notice", "Notice")
        bubble.set_plain(text)
        return bubble

    # -- live turn -----------------------------------------------------------------

    def begin_model_step(self) -> None:
        """A model request started: the next text deltas go to a fresh bubble."""
        if self._stream_bubble is not None:
            self._flush_stream()
            self._stream_bubble = None
        self._stream_chunks = []
        self._render_timer.setInterval(STREAM_RENDER_INTERVAL_MS)

    def append_stream_text(self, delta: str) -> None:
        if not delta:
            return
        if self._stream_bubble is None:
            self._stream_bubble = self._new_bubble("assistant", APP_DISPLAY_NAME)
            self._stream_chunks = []
        self._stream_chunks.append(delta)
        self._stream_dirty = True
        if not self._render_timer.isActive():
            self._render_timer.start()

    def _flush_stream(self) -> None:
        if self._stream_bubble is not None and self._stream_dirty:
            self._stream_dirty = False
            started = time.perf_counter()
            self._stream_bubble.set_markdown("".join(self._stream_chunks))
            # Adaptive throttle: re-rendering costs grow with the answer's length,
            # so keep the GUI thread at most ~25% busy with stream renders.
            elapsed_ms = (time.perf_counter() - started) * 1000
            self._render_timer.setInterval(max(STREAM_RENDER_INTERVAL_MS, int(elapsed_ms * 3)))

    def add_assistant_message(self, message: ChatMessage) -> None:
        """Authoritative assistant message: replaces any streamed preview."""
        self._render_timer.stop()
        self._stream_dirty = False
        content = message.content or ""
        bubble, self._stream_bubble = self._stream_bubble, None
        self._stream_chunks = []
        if bubble is not None:
            if content.strip():
                bubble.set_markdown(content)
            else:
                self._remove_widget_row(bubble)
        elif content.strip():
            self.add_assistant_text(content)

    def abort_stream(self, note: str = "*[response interrupted]*") -> None:
        """The turn ended without an authoritative message for the streamed text."""
        self._render_timer.stop()
        bubble, self._stream_bubble = self._stream_bubble, None
        text = "".join(self._stream_chunks)
        self._stream_chunks = []
        self._stream_dirty = False
        if bubble is None:
            return
        if text.strip():
            bubble.set_markdown(f"{text}\n\n{note}")
        else:
            self._remove_widget_row(bubble)

    def tool_started(self, call: ToolCall, summary: str, decision: Decision | None) -> ToolCard:
        card = self._cards.get(call.id)
        if card is None:
            card = ToolCard(call, summary, decision)
            self._cards[call.id] = card
            self._add_row(card, "left")
        else:
            card.set_decision(decision)
        return card

    def tool_awaiting_approval(self, call_id: str) -> None:
        card = self._cards.get(call_id)
        if card is not None:
            card.set_status("awaiting")

    def tool_approval_answered(self, call_id: str, approved: bool) -> None:
        card = self._cards.get(call_id)
        if card is not None and card.status == "awaiting":
            card.set_status("running" if approved else "declined")

    def tool_finished(self, call: ToolCall, result: ToolResult) -> None:
        card = self._cards.get(call.id)
        if card is None:
            card = self.tool_started(call, call.name, None)
        card.set_result(result.ok, result.content)

    # -- history ---------------------------------------------------------------------

    def load_messages(self, messages: Sequence[ChatMessage], describe: ToolDescriber | None = None) -> None:
        """Render a stored conversation; tool calls are paired with results by id."""
        self.clear()
        results = {m.tool_call_id: m for m in messages if m.role is Role.TOOL and m.tool_call_id}
        shown: set[str] = set()
        for message in messages:
            if message.role is Role.USER:
                bubble = self._new_bubble("user", "You")
                bubble.set_plain(message.content or "")
            elif message.role is Role.ASSISTANT:
                if (message.content or "").strip():
                    self.add_assistant_text(message.content or "")
                for call in message.tool_calls:
                    summary, decision = describe(call) if describe else (call.name, None)
                    result = results.get(call.id)
                    if result is not None and decision is not Decision.DENY:
                        text = result.content or ""
                        body = text[len(ERROR_PREFIX):] if text.startswith(ERROR_PREFIX) else text
                        if body.strip() == DECLINED_MESSAGE:
                            decision = Decision.CONFIRM
                    card = ToolCard(call, summary, decision)
                    self._cards[call.id] = card
                    self._add_row(card, "left")
                    if result is None:
                        card.set_status("no_result")
                    else:
                        content = result.content or ""
                        ok = not content.startswith(ERROR_PREFIX)
                        card.set_result(ok, content)
                        shown.add(call.id)
            elif message.role is Role.TOOL and message.tool_call_id not in shown:
                # Orphaned tool result (its call was not stored): still show it.
                call = ToolCall(message.tool_call_id or "?", message.name or "tool", "{}")
                if call.id in self._cards:
                    continue
                card = ToolCard(call, message.name or "tool", None)
                self._cards[call.id] = card
                self._add_row(card, "left")
                content = message.content or ""
                card.set_result(not content.startswith(ERROR_PREFIX), content)
        self.scroll_to_bottom()

    def rerender(self) -> None:
        """Re-apply theme colours to rendered Markdown."""
        for bubble in self._bubbles:
            bubble.rerender()

    # -- scrolling -------------------------------------------------------------------

    def scroll_to_bottom(self) -> None:
        self._stick_to_bottom = True
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def is_at_bottom(self) -> bool:
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - 24

    def _on_scroll_value(self, _value: int) -> None:
        self._stick_to_bottom = self.is_at_bottom()

    def _on_scroll_range(self, _minimum: int, maximum: int) -> None:
        if self._stick_to_bottom:
            self.verticalScrollBar().setValue(maximum)
