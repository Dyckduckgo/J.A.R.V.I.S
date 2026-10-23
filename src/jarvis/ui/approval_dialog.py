"""Modal confirmation for consequential tool calls.

Safety properties (covered by tests):

* **Deny** is the default button and has initial focus, so Enter denies;
* Escape, the window close button and any other dismissal deny;
* the call is approved only by explicitly clicking **Allow once**;
* the full summary and the exact JSON arguments are shown (never truncated,
  never interpreted as HTML).
"""

from __future__ import annotations

import json
import math
from typing import Any

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontDatabase, QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from jarvis import APP_DISPLAY_NAME
from jarvis.agent.events import ApprovalRequest
from jarvis.tools.base import RiskLevel
from jarvis.ui.theme import set_tone


class _FitTextView(QPlainTextEdit):
    """Read-only monospace text that is as tall as its content, up to ``max_lines``
    (longer content scrolls - nothing is ever cut off)."""

    def __init__(self, text: str, max_lines: int) -> None:
        super().__init__()
        self._max_lines = max_lines
        self.setObjectName("Monospace")
        self.setReadOnly(True)
        self.setPlainText(text)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.setTabChangesFocus(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.document().documentLayout().documentSizeChanged.connect(lambda *_: self.updateGeometry())

    def _height_for_lines(self, lines: int) -> int:
        margins = self.contentsMargins()
        extra = 2 * self.frameWidth() + margins.top() + margins.bottom() + int(2 * self.document().documentMargin())
        return lines * self.fontMetrics().lineSpacing() + extra + 4

    def _line_count(self) -> int:
        # Before the widget has its final width the layout counts unwrapped
        # lines only, so also estimate wrapping (~60 monospace columns at the
        # dialog's minimum width) to avoid a 3-line box with a scrollbar.
        laid_out = int(self.document().documentLayout().documentSize().height())
        estimated = sum(max(1, math.ceil(len(line) / 60)) for line in self.toPlainText().splitlines() or [""])
        return max(1, laid_out, estimated)

    def sizeHint(self) -> QSize:
        return QSize(super().sizeHint().width(), self._height_for_lines(min(self._line_count(), self._max_lines)))

    def minimumSizeHint(self) -> QSize:
        # Security-relevant text: keep the whole thing visible up to max_lines.
        return QSize(120, self._height_for_lines(min(self._line_count(), self._max_lines)))


def format_arguments(arguments: Any) -> str:
    try:
        return json.dumps(arguments, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(arguments)


class ApprovalDialog(QDialog):
    """Ask the user to allow or deny one tool call. ``approved`` is the answer."""

    def __init__(self, request: ApprovalRequest, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.request = request
        self._approved = False
        self.setWindowTitle(f"{APP_DISPLAY_NAME} – Approve action?")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setSizeGripEnabled(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title = QLabel("The assistant wants to perform an action")
        title.setObjectName("SectionTitle")
        title.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(6)
        grid.addWidget(self._caption("Tool"), 0, 0)
        self.tool_label = QLabel(request.tool_name)
        self.tool_label.setTextFormat(Qt.TextFormat.PlainText)
        self.tool_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(self.tool_label, 0, 1)

        grid.addWidget(self._caption("Risk"), 1, 0)
        risk_value = request.risk.value if isinstance(request.risk, RiskLevel) else str(request.risk)
        self.risk_label = QLabel(risk_value.capitalize())
        self.risk_label.setObjectName("RiskBadge")
        self.risk_label.setTextFormat(Qt.TextFormat.PlainText)
        set_tone(self.risk_label, "warning" if risk_value == RiskLevel.CONSEQUENTIAL.value else "success")
        risk_row = QHBoxLayout()
        risk_row.addWidget(self.risk_label)
        risk_row.addStretch(1)
        grid.addLayout(risk_row, 1, 1)

        grid.addWidget(self._caption("Why"), 2, 0, Qt.AlignmentFlag.AlignTop)
        self.reason_label = QLabel(request.reason)
        self.reason_label.setTextFormat(Qt.TextFormat.PlainText)
        self.reason_label.setWordWrap(True)
        self.reason_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        grid.addWidget(self.reason_label, 2, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        layout.addWidget(self._caption("What will happen"))
        self.summary_view = _FitTextView(request.summary, max_lines=10)
        layout.addWidget(self.summary_view)

        layout.addWidget(self._caption("Arguments (JSON)"))
        self.arguments_view = _FitTextView(format_arguments(request.arguments), max_lines=14)
        layout.addWidget(self.arguments_view)
        layout.addStretch(1)

        hint = QLabel("Only allow actions you expected. Closing this window or pressing Esc denies the action.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        layout.addWidget(line)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.allow_button = QPushButton("Allow once")
        self.allow_button.setProperty("variant", "danger")
        self.allow_button.setAutoDefault(False)
        self.allow_button.setDefault(False)
        self.allow_button.clicked.connect(self._allow)
        self.deny_button = QPushButton("Deny")
        self.deny_button.setProperty("variant", "primary")
        self.deny_button.setAutoDefault(True)
        self.deny_button.setDefault(True)
        self.deny_button.clicked.connect(self.reject)
        buttons.addWidget(self.allow_button)
        buttons.addWidget(self.deny_button)
        layout.addLayout(buttons)

        self.deny_button.setFocus(Qt.FocusReason.OtherFocusReason)

    @staticmethod
    def _caption(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("Hint")
        return label

    @property
    def approved(self) -> bool:
        return self._approved and self.result() == QDialog.DialogCode.Accepted

    def _allow(self) -> None:
        self._approved = True
        self.accept()

    def reject(self) -> None:  # Escape, close button, Deny
        self._approved = False
        super().reject()

    def done(self, result: int) -> None:
        if result != QDialog.DialogCode.Accepted:
            self._approved = False
        super().done(result)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.focusWidget() is self.allow_button:
            # Keyboard activation of "Allow" must be a deliberate Space press, not a stray Enter.
            event.accept()
            return
        super().keyPressEvent(event)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self.deny_button.setFocus(Qt.FocusReason.OtherFocusReason)
