"""Main application window: conversation sidebar, chat transcript and composer."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, QStandardPaths, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeyEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from jarvis import APP_DISPLAY_NAME, __version__
from jarvis.agent.service import MAX_INPUT_CHARS
from jarvis.app_context import AppContext
from jarvis.config.secrets import API_KEY_ENV_VAR, KeySource
from jarvis.config.settings import Settings
from jarvis.errors import JarvisError
from jarvis.llm.base import ChatMessage, Role, ToolCall
from jarvis.tools.base import RiskLevel, ToolContext, ToolResult
from jarvis.tools.policy import Decision
from jarvis.ui.approval_dialog import ApprovalDialog
from jarvis.ui.chat_view import ChatView
from jarvis.ui.settings_dialog import KEY_SOURCE_TEXT, SettingsDialog
from jarvis.ui.theme import apply_theme, set_tone
from jarvis.ui.worker import DEFAULT_APPROVAL_TIMEOUT_S, PendingApproval, TurnOutcome, TurnRunner

log = logging.getLogger(__name__)

CONVERSATION_ID_ROLE = Qt.ItemDataRole.UserRole
SHUTDOWN_WAIT_S = 3.0
NO_KEY_TEXT = (
    "No Groq API key is configured yet. Add your key in Settings to start chatting "
    f"(or set the {API_KEY_ENV_VAR} environment variable)."
)


class ChatInput(QPlainTextEdit):
    """Multi-line composer: Enter sends, Shift+Enter inserts a newline."""

    submitted = Signal()

    MAX_VISIBLE_LINES = 8

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ChatInput")
        self.setPlaceholderText(f"Message {APP_DISPLAY_NAME}…")
        self.setTabChangesFocus(True)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.document().documentLayout().documentSizeChanged.connect(self._fit_height)
        self._fit_height()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.insertPlainText("\n")
                event.accept()
                return
            if not (event.modifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)):
                event.accept()
                self.submitted.emit()
                return
        super().keyPressEvent(event)

    def _fit_height(self, *_: Any) -> None:
        lines = max(1, min(int(self.document().size().height()), self.MAX_VISIBLE_LINES))
        line_height = self.fontMetrics().lineSpacing()
        margins = self.contentsMargins()
        doc_margin = int(self.document().documentMargin() * 2)
        height = lines * line_height + margins.top() + margins.bottom() + doc_margin + 6
        self.setFixedHeight(max(height, 38))


class Banner(QFrame):
    """Inline message strip with an optional action button and a close button."""

    action_triggered = Signal()

    def __init__(
        self, text: str, tone: str = "info", action_text: str | None = None, closable: bool = True
    ) -> None:
        super().__init__()
        self.setObjectName("Banner")
        self.setProperty("tone", tone)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 8, 8)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.PlainText)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.label, 1)
        self.action_button: QPushButton | None = None
        if action_text:
            self.action_button = QPushButton(action_text)
            self.action_button.setProperty("variant", "primary")
            self.action_button.clicked.connect(self.action_triggered)
            layout.addWidget(self.action_button)
        if closable:
            close = QToolButton()
            close.setText("✕")
            close.setToolTip("Dismiss")
            close.setAutoRaise(True)
            close.clicked.connect(self.hide)
            layout.addWidget(close)

    def text(self) -> str:
        return self.label.text()


class MainWindow(QMainWindow):
    def __init__(
        self,
        ctx: AppContext,
        parent: QWidget | None = None,
        *,
        approval_timeout: float | None = DEFAULT_APPROVAL_TIMEOUT_S,
    ) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._conversation_id: int | None = None
        self._populating = False
        self._closed = False
        self._approval_dialog: ApprovalDialog | None = None
        self._approval_pending: PendingApproval | None = None
        self._settings_dialog: SettingsDialog | None = None
        self.approval_dialog_class: type[ApprovalDialog] = ApprovalDialog

        self._runner = TurnRunner(ctx.chat, self, approval_timeout=approval_timeout)
        self._runner.busy_changed.connect(self._on_busy_changed)
        self._runner.message_received.connect(self._on_message)
        self._runner.llm_started.connect(self._on_llm_started)
        self._runner.text_received.connect(self._on_text)
        self._runner.tool_started.connect(self._on_tool_started)
        self._runner.tool_finished.connect(self._on_tool_finished)
        self._runner.approval_requested.connect(self._on_approval_requested)
        self._runner.approval_closed.connect(self._on_approval_closed)
        self._runner.turn_finished.connect(self._on_turn_finished)

        self.setWindowTitle(APP_DISPLAY_NAME)
        self.resize(1180, 780)
        self.setMinimumSize(720, 480)
        self._build_ui()
        self._build_menus()
        self._build_status_bar()
        self._restore_geometry()

        self._apply_settings(ctx.settings.settings, rerender=False)
        self._refresh_key_state()
        for notice in ctx.notices:
            self.add_banner(notice, tone="warning")
        self.refresh_conversations()
        self._open_initial_conversation()
        self.input.setFocus()

    # ------------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(1)

        # Sidebar
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setMinimumWidth(200)
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(12, 14, 12, 12)
        side.setSpacing(10)
        title = QLabel(APP_DISPLAY_NAME)
        title.setObjectName("AppTitle")
        side.addWidget(title)
        self.new_chat_button = QPushButton("+  New chat")
        self.new_chat_button.setObjectName("NewChatButton")
        self.new_chat_button.setProperty("variant", "primary")
        self.new_chat_button.setToolTip("Start a new conversation (Ctrl+N)")
        self.new_chat_button.clicked.connect(self.new_chat)
        side.addWidget(self.new_chat_button)
        self.search_edit = QLineEdit()
        self.search_edit.setObjectName("SearchField")
        self.search_edit.setPlaceholderText("Search conversations")
        self.search_edit.setClearButtonEnabled(True)
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(200)
        self._search_timer.timeout.connect(self.refresh_conversations)
        self.search_edit.textChanged.connect(lambda _text: self._search_timer.start())
        side.addWidget(self.search_edit)
        self.conversation_list = QListWidget()
        self.conversation_list.setObjectName("ConversationList")
        self.conversation_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.conversation_list.customContextMenuRequested.connect(self._show_conversation_menu)
        self.conversation_list.currentItemChanged.connect(self._on_conversation_selected)
        self.conversation_list.setUniformItemSizes(True)
        self.conversation_list.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.conversation_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side.addWidget(self.conversation_list, 1)
        delete_shortcut = QShortcut(QKeySequence(QKeySequence.StandardKey.Delete), self.conversation_list)
        delete_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        delete_shortcut.activated.connect(lambda: self.delete_conversation(self._selected_conversation_id()))
        rename_shortcut = QShortcut(QKeySequence(Qt.Key.Key_F2), self.conversation_list)
        rename_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        rename_shortcut.activated.connect(lambda: self.rename_conversation(self._selected_conversation_id()))

        # Chat area
        central = QWidget()
        main = QVBoxLayout(central)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        self.banner_area = QWidget()
        self.banner_layout = QVBoxLayout(self.banner_area)
        self.banner_layout.setContentsMargins(16, 10, 16, 4)
        self.banner_layout.setSpacing(6)
        self.no_key_banner = Banner(NO_KEY_TEXT, tone="warning", action_text="Open Settings", closable=False)
        self.no_key_banner.action_triggered.connect(lambda: self.open_settings(tab="API key"))
        self.no_key_banner.setVisible(False)
        self.banner_layout.addWidget(self.no_key_banner)
        main.addWidget(self.banner_area)

        self.chat_view = ChatView()
        main.addWidget(self.chat_view, 1)

        composer_wrap = QWidget()
        wrap = QVBoxLayout(composer_wrap)
        wrap.setContentsMargins(16, 6, 16, 10)
        wrap.setSpacing(4)
        self.composer = QFrame()
        self.composer.setObjectName("Composer")
        comp = QHBoxLayout(self.composer)
        comp.setContentsMargins(8, 6, 6, 6)
        comp.setSpacing(6)
        self.input = ChatInput()
        self.input.submitted.connect(self.send_message)
        comp.addWidget(self.input, 1)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("SendButton")
        self.send_button.setProperty("variant", "primary")
        self.send_button.setProperty("mode", "send")
        self.send_button.setToolTip("Send (Enter)")
        self.send_button.clicked.connect(self._on_send_clicked)
        comp.addWidget(self.send_button, 0, Qt.AlignmentFlag.AlignBottom)
        wrap.addWidget(self.composer)
        hint = QLabel("Enter to send · Shift+Enter for a new line · Esc to stop")
        hint.setObjectName("Hint")
        wrap.addWidget(hint)
        main.addWidget(composer_wrap)

        self.splitter.addWidget(self.sidebar)
        self.splitter.addWidget(central)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([260, 920])
        self.setCentralWidget(self.splitter)

    def _build_menus(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self.new_chat_action = QAction("&New Chat", self)
        self.new_chat_action.setShortcut(QKeySequence("Ctrl+N"))
        self.new_chat_action.triggered.connect(self.new_chat)
        file_menu.addAction(self.new_chat_action)
        self.export_action = QAction("&Export Conversation…", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+E"))
        self.export_action.triggered.connect(lambda: self.export_conversation())
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        self.settings_action = QAction("&Settings…", self)
        self.settings_action.setShortcuts([QKeySequence("Ctrl+,"), QKeySequence(QKeySequence.StandardKey.Preferences)])
        self.settings_action.triggered.connect(lambda: self.open_settings())
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()
        self.quit_action = QAction("&Quit", self)
        self.quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        self.quit_action.triggered.connect(self.close)
        file_menu.addAction(self.quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        about = QAction(f"&About {APP_DISPLAY_NAME}", self)
        about.triggered.connect(self.show_about)
        help_menu.addAction(about)
        logs = QAction("Open &Log Folder", self)
        logs.triggered.connect(self.open_log_folder)
        help_menu.addAction(logs)

        self.stop_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self.stop_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self.stop_shortcut.activated.connect(self.stop_turn)

    def _build_status_bar(self) -> None:
        bar = self.statusBar()
        bar.setSizeGripEnabled(True)
        self.busy_label = QLabel("")
        self.busy_indicator = QProgressBar()
        self.busy_indicator.setRange(0, 0)
        self.busy_indicator.setFixedWidth(90)
        self.busy_indicator.setTextVisible(False)
        self.busy_indicator.setVisible(False)
        self.model_label = QLabel()
        self.key_label = QLabel()
        bar.addPermanentWidget(self.busy_label)
        bar.addPermanentWidget(self.busy_indicator)
        bar.addPermanentWidget(self.model_label)
        bar.addPermanentWidget(self.key_label)

    # ------------------------------------------------------------------ state

    @property
    def ctx(self) -> AppContext:
        return self._ctx

    @property
    def runner(self) -> TurnRunner:
        return self._runner

    @property
    def conversation_id(self) -> int | None:
        return self._conversation_id

    @property
    def approval_dialog(self) -> ApprovalDialog | None:
        return self._approval_dialog

    @property
    def settings_dialog(self) -> SettingsDialog | None:
        return self._settings_dialog

    def is_busy(self) -> bool:
        return self._runner.is_busy()

    def add_banner(self, text: str, tone: str = "info") -> Banner:
        banner = Banner(text, tone=tone)
        self.banner_layout.addWidget(banner)
        return banner

    def banners(self) -> list[Banner]:
        return [b for b in self.banner_area.findChildren(Banner) if b.isVisible()]

    # ------------------------------------------------------------------ settings

    def _apply_settings(self, settings: Settings, rerender: bool = True) -> None:
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, settings.theme)
        if rerender:
            self.chat_view.rerender()
        self.model_label.setText(f"Model: {settings.model}")
        self.model_label.setToolTip("Change the model in Settings")

    def _refresh_key_state(self) -> None:
        try:
            source = self._ctx.secrets.source()
        except Exception:  # never let a credential-store hiccup break the window
            log.exception("Could not determine API key source")
            source = KeySource.NONE
        if source is KeySource.NONE:
            self.key_label.setText("API key: not set")
            set_tone(self.key_label, "danger")
        else:
            self.key_label.setText("API key: " + ("environment" if source is KeySource.ENVIRONMENT else "stored"))
            set_tone(self.key_label, "success")
        self.key_label.setToolTip(KEY_SOURCE_TEXT.get(source, ""))
        self.no_key_banner.setVisible(source is KeySource.NONE)

    def open_settings(self, tab: str | None = None) -> SettingsDialog:
        if self._settings_dialog is not None:
            self._settings_dialog.raise_()
            self._settings_dialog.activateWindow()
            return self._settings_dialog
        dialog = SettingsDialog(self._ctx, self)
        if tab:
            for index in range(dialog.tabs.count()):
                if dialog.tabs.tabText(index) == tab:
                    dialog.tabs.setCurrentIndex(index)
        dialog.settings_saved.connect(self._apply_settings)
        dialog.api_key_changed.connect(self._refresh_key_state)
        dialog.finished.connect(self._on_settings_closed)
        self._settings_dialog = dialog
        dialog.open()
        return dialog

    def _on_settings_closed(self, _result: int) -> None:
        dialog, self._settings_dialog = self._settings_dialog, None
        self._refresh_key_state()
        if dialog is not None:
            dialog.deleteLater()

    # ------------------------------------------------------------------ conversations

    def refresh_conversations(self) -> None:
        search = self.search_edit.text()
        try:
            conversations = self._ctx.store.list_conversations(search)
        except JarvisError as exc:
            self.statusBar().showMessage(exc.user_message, 8000)
            return
        self._populating = True
        try:
            self.conversation_list.clear()
            for conv in conversations:
                item = QListWidgetItem(conv.title)
                item.setData(CONVERSATION_ID_ROLE, conv.id)
                stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(conv.updated_at))
                item.setToolTip(f"{conv.title}\nUpdated {stamp} · {conv.message_count} messages")
                self.conversation_list.addItem(item)
                if conv.id == self._conversation_id:
                    self.conversation_list.setCurrentItem(item)
            if self._conversation_id is None:
                self.conversation_list.setCurrentItem(None)
        finally:
            self._populating = False

    def conversation_titles(self) -> list[str]:
        return [self.conversation_list.item(i).text() for i in range(self.conversation_list.count())]

    def _item_for(self, conversation_id: int) -> QListWidgetItem | None:
        for i in range(self.conversation_list.count()):
            item = self.conversation_list.item(i)
            if item.data(CONVERSATION_ID_ROLE) == conversation_id:
                return item
        return None

    def _selected_conversation_id(self) -> int | None:
        item = self.conversation_list.currentItem()
        return int(item.data(CONVERSATION_ID_ROLE)) if item is not None else None

    def _open_initial_conversation(self) -> None:
        try:
            conversations = self._ctx.store.list_conversations()
        except JarvisError as exc:
            self.chat_view.add_error(exc.user_message)
            return
        if conversations:
            self.open_conversation(conversations[0].id)

    def _on_conversation_selected(self, current: QListWidgetItem | None, _previous: Any) -> None:
        if self._populating or current is None:
            return
        conversation_id = int(current.data(CONVERSATION_ID_ROLE))
        if conversation_id == self._conversation_id:
            return
        if self.is_busy():
            self.statusBar().showMessage("Wait for the current response to finish (or press Esc to stop).", 5000)
            self._reselect_current()
            return
        self.open_conversation(conversation_id)

    def _reselect_current(self) -> None:
        self._populating = True
        try:
            item = self._item_for(self._conversation_id) if self._conversation_id is not None else None
            self.conversation_list.setCurrentItem(item)
        finally:
            self._populating = False

    def open_conversation(self, conversation_id: int) -> bool:
        if self.is_busy():
            return False
        try:
            conversation = self._ctx.store.get_conversation(conversation_id)
            messages = self._ctx.store.get_messages(conversation_id) if conversation else []
        except JarvisError as exc:
            self.chat_view.add_error(exc.user_message)
            return False
        if conversation is None:
            self.refresh_conversations()
            return False
        self._conversation_id = conversation_id
        self.chat_view.load_messages(messages, describe=self.describe_call)
        self._update_window_title(conversation.title)
        self._reselect_current()
        return True

    def new_chat(self) -> None:
        if self.is_busy():
            self.statusBar().showMessage("Wait for the current response to finish (or press Esc to stop).", 5000)
            return
        self._conversation_id = None
        self.chat_view.clear()
        self._update_window_title(None)
        self._reselect_current()
        self.input.setFocus()

    def _update_window_title(self, title: str | None) -> None:
        self.setWindowTitle(f"{title} – {APP_DISPLAY_NAME}" if title else APP_DISPLAY_NAME)

    def _show_conversation_menu(self, pos: Any) -> None:
        item = self.conversation_list.itemAt(pos)
        if item is None:
            return
        conversation_id = int(item.data(CONVERSATION_ID_ROLE))
        locked = self.is_busy() and conversation_id == self._runner.conversation_id
        menu = QMenu(self)
        rename = menu.addAction("Rename…")
        export = menu.addAction("Export to Markdown…")
        menu.addSeparator()
        delete = menu.addAction("Delete…")
        rename.setEnabled(not locked)
        delete.setEnabled(not locked)
        chosen = menu.exec(self.conversation_list.viewport().mapToGlobal(pos))
        if chosen is rename:
            self.rename_conversation(conversation_id)
        elif chosen is export:
            self.export_conversation(conversation_id)
        elif chosen is delete:
            self.delete_conversation(conversation_id)

    def rename_conversation(self, conversation_id: int | None) -> bool:
        if conversation_id is None:
            return False
        if self.is_busy() and conversation_id == self._runner.conversation_id:
            self.statusBar().showMessage("This conversation is busy; try again when the response is complete.", 5000)
            return False
        conversation = self._ctx.store.get_conversation(conversation_id)
        if conversation is None:
            return False
        text, ok = QInputDialog.getText(
            self, "Rename conversation", "New title:", QLineEdit.EchoMode.Normal, conversation.title
        )
        if not ok or not text.strip():
            return False
        try:
            self._ctx.store.rename_conversation(conversation_id, text)
        except JarvisError as exc:
            QMessageBox.warning(self, "Rename failed", exc.user_message)
            return False
        if conversation_id == self._conversation_id:
            self._update_window_title(text.strip())
        self.refresh_conversations()
        return True

    def export_conversation(self, conversation_id: int | None = None) -> Path | None:
        conversation_id = conversation_id if conversation_id is not None else self._conversation_id
        if conversation_id is None:
            QMessageBox.information(self, "Export conversation", "There is no conversation to export yet.")
            return None
        try:
            conversation = self._ctx.store.get_conversation(conversation_id)
            if conversation is None:
                return None
            content = self._ctx.store.export_markdown(conversation_id)
        except JarvisError as exc:
            QMessageBox.warning(self, "Export failed", exc.user_message)
            return None
        safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", conversation.title).strip(" ._") or "conversation"
        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        suggested = str(Path(documents or Path.home()) / f"{safe_name[:80]}.md")
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export conversation", suggested, "Markdown files (*.md);;All files (*)"
        )
        if not path:
            return None
        target = Path(path)
        try:
            target.write_text(content, encoding="utf-8")
        except OSError as exc:
            log.warning("Export to %s failed: %s", target, exc)
            QMessageBox.warning(self, "Export failed", f"Could not write the file:\n{exc.strerror or exc}")
            return None
        self.statusBar().showMessage(f"Exported to {target}", 6000)
        return target

    def delete_conversation(self, conversation_id: int | None) -> bool:
        if conversation_id is None:
            return False
        if self.is_busy() and conversation_id == self._runner.conversation_id:
            self.statusBar().showMessage("This conversation is busy; stop the response before deleting it.", 5000)
            return False
        conversation = self._ctx.store.get_conversation(conversation_id)
        if conversation is None:
            self.refresh_conversations()
            return False
        answer = QMessageBox.question(
            self,
            "Delete conversation",
            f"Permanently delete “{conversation.title}”?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return False
        try:
            self._ctx.store.delete_conversation(conversation_id)
        except JarvisError as exc:
            QMessageBox.warning(self, "Delete failed", exc.user_message)
            return False
        if conversation_id == self._conversation_id:
            self._conversation_id = None
            self.chat_view.clear()
            self._update_window_title(None)
            self._open_initial_conversation()
        self.refresh_conversations()
        self.statusBar().showMessage("Conversation deleted.", 4000)
        return True

    def describe_call(self, call: ToolCall) -> tuple[str, Decision | None]:
        """Summary and (best-effort) policy decision for a stored tool call."""
        tool = self._ctx.registry.get(call.name)
        if tool is None:
            return f"{call.name}({call.arguments})", Decision.DENY
        try:
            args = tool.validate(call.parsed_arguments())
        except JarvisError:
            return f"{call.name}(<invalid arguments>)", Decision.DENY
        except Exception:
            log.exception("Could not validate stored call to %s", call.name)
            return f"{call.name}({call.arguments})", None
        # A context lets tools render relative paths as absolute ones, as in the approval dialog.
        tool_ctx = ToolContext(workspace=self._ctx.settings.settings.workspace_path)
        try:  # best effort: re-assess against the current workspace (e.g. reads outside it needed approval)
            risks = {tool.risk, tool.assess_risk(args, tool_ctx)}
        except Exception:
            risks = {RiskLevel.CONSEQUENTIAL}
        decision = Decision.CONFIRM if RiskLevel.CONSEQUENTIAL in risks else Decision.AUTO
        try:
            summary = tool.describe_call(args, tool_ctx)
        except Exception:  # old stored arguments may no longer describe cleanly
            log.warning("describe_call failed for stored call to %s", call.name, exc_info=True)
            summary = f"{call.name}({call.arguments})"
        return summary, decision

    # ------------------------------------------------------------------ sending

    def _on_send_clicked(self) -> None:
        if self.is_busy():
            self.stop_turn()
        else:
            self.send_message()

    def send_message(self) -> bool:
        if self.is_busy():
            self.statusBar().showMessage("Still working on the previous message – press Esc to stop it.", 4000)
            return False
        raw = self.input.toPlainText()
        text = raw.strip()
        if not text:
            return False
        if len(text) > MAX_INPUT_CHARS:
            self.chat_view.add_error(
                f"Your message is too long ({len(text):,} characters; the limit is {MAX_INPUT_CHARS:,})."
            )
            return False
        if self._conversation_id is None:
            try:
                conversation = self._ctx.store.create_conversation()
            except JarvisError as exc:
                self.chat_view.add_error(exc.user_message)
                return False
            self._conversation_id = conversation.id
            self.refresh_conversations()
        started = self._runner.start(self._conversation_id, text, model=self._ctx.settings.settings.model)
        if started:
            self.input.clear()
        return started

    def stop_turn(self) -> None:
        if not self.is_busy():
            return
        self.busy_label.setText("Stopping…")
        self._runner.stop()

    # ------------------------------------------------------------------ runner events

    def _on_busy_changed(self, busy: bool) -> None:
        self.send_button.setText("Stop" if busy else "Send")
        self.send_button.setToolTip("Stop the response (Esc)" if busy else "Send (Enter)")
        set_tone(self.send_button, "stop" if busy else "send", prop="mode")
        self.busy_indicator.setVisible(busy)
        self.busy_label.setText("Working…" if busy else "")
        self.new_chat_button.setEnabled(not busy)
        self.new_chat_action.setEnabled(not busy)
        self.conversation_list.setEnabled(not busy)

    def _on_message(self, message: ChatMessage) -> None:
        if message.role is Role.USER:
            self.chat_view.add_user_message(message.content or "")
            self.refresh_conversations()  # the first message renames the conversation
            try:
                conversation = self._ctx.store.get_conversation(self._conversation_id or -1)
            except JarvisError:
                conversation = None
            if conversation is not None:
                self._update_window_title(conversation.title)
        elif message.role is Role.ASSISTANT:
            self.chat_view.add_assistant_message(message)

    def _on_llm_started(self, iteration: int) -> None:
        self.chat_view.begin_model_step()
        self.busy_label.setText("Thinking…" if iteration == 1 else f"Thinking… (step {iteration})")

    def _on_text(self, delta: str) -> None:
        self.chat_view.append_stream_text(delta)
        self.busy_label.setText("Writing…")

    def _on_tool_started(self, call: ToolCall, summary: str, decision: Decision) -> None:
        self.chat_view.tool_started(call, summary, decision)
        if decision is Decision.CONFIRM:
            self.busy_label.setText("Waiting for your approval…")
        elif decision is Decision.AUTO:
            self.busy_label.setText(f"Running {call.name}…")

    def _on_tool_finished(self, call: ToolCall, result: ToolResult) -> None:
        self.chat_view.tool_finished(call, result)

    def _on_approval_requested(self, pending: PendingApproval) -> None:
        if self._closed or pending.resolved:
            pending.resolve(False, by="closed")
            return
        if self._approval_dialog is not None:  # never stack dialogs; agent asks one at a time
            log.warning("Approval requested while another dialog is open; denying")
            self._runner.answer_approval(pending, False)
            return
        self.chat_view.tool_awaiting_approval(pending.request.call.id)
        self.busy_label.setText("Waiting for your approval…")
        try:
            dialog = self.approval_dialog_class(pending.request, self)
        except Exception:  # fail closed: a broken dialog must never leave the worker waiting
            log.exception("Could not show the approval dialog; denying the action")
            self._runner.answer_approval(pending, False)
            self.chat_view.tool_approval_answered(pending.request.call.id, False)
            return
        self._approval_dialog = dialog
        self._approval_pending = pending
        dialog.finished.connect(lambda _code, d=dialog, p=pending: self._on_approval_dialog_finished(d, p))
        dialog.open()

    def _on_approval_dialog_finished(self, dialog: ApprovalDialog, pending: PendingApproval) -> None:
        self._runner.answer_approval(pending, dialog.approved)
        if self._approval_dialog is dialog:
            self._approval_dialog = None
            self._approval_pending = None
        self.chat_view.tool_approval_answered(pending.request.call.id, pending.approved)
        if self.is_busy():
            self.busy_label.setText("Working…" if pending.approved else "Declined")
        dialog.deleteLater()

    def _on_approval_closed(self, pending: PendingApproval) -> None:
        """The approval was resolved without the user (Stop, timeout): close its dialog."""
        if self._approval_pending is pending and self._approval_dialog is not None:
            self._approval_dialog.reject()
        if pending.resolved_by == "timeout":
            self.chat_view.add_notice("No answer was given in time, so the action was not performed.")

    def _close_approval_dialog(self) -> None:
        if self._approval_dialog is not None:
            self._approval_dialog.reject()

    def _on_turn_finished(self, outcome: TurnOutcome) -> None:
        self._close_approval_dialog()
        if outcome.ok:
            self.chat_view.abort_stream("*[stopped]*" if outcome.cancelled else "*[response interrupted]*")
            if outcome.cancelled:
                self.chat_view.add_notice("Stopped.")
                self.statusBar().showMessage("Response stopped.", 4000)
            elif outcome.result is not None and outcome.result.stopped_at_iteration_limit:
                self.statusBar().showMessage("Stopped at the tool step limit.", 6000)
        else:
            self.chat_view.abort_stream()
            # ChatService rolls back a user message that got no reply at all;
            # re-render so the view matches what is stored.
            if outcome.conversation_id == self._conversation_id:
                self.open_conversation(outcome.conversation_id)
            self.chat_view.add_error(outcome.error or "")
            if outcome.missing_api_key:
                self._refresh_key_state()
                self.no_key_banner.setVisible(True)
            # Give the text back so the user can retry or edit it.
            if not self.input.toPlainText().strip():
                self.input.setPlainText(outcome.text)
                cursor = self.input.textCursor()
                cursor.movePosition(cursor.MoveOperation.End)
                self.input.setTextCursor(cursor)
        self.refresh_conversations()
        if not self._closed:
            self.input.setFocus()

    # ------------------------------------------------------------------ help

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_DISPLAY_NAME}",
            f"<h3>{APP_DISPLAY_NAME}</h3>"
            f"<p>Version {__version__}</p>"
            "<p>A desktop AI assistant powered by the Groq API, with tool use and "
            "human-in-the-loop approval for every consequential action.</p>",
        )

    def open_log_folder(self) -> None:
        folder = self._ctx.paths.log_dir
        folder.mkdir(parents=True, exist_ok=True)
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            QMessageBox.information(self, "Log folder", f"The log files are in:\n{folder}")

    # ------------------------------------------------------------------ geometry / close

    def _window_settings(self) -> QSettings:
        return QSettings(str(self._ctx.paths.config_dir / "window.ini"), QSettings.Format.IniFormat)

    def _restore_geometry(self) -> None:
        try:
            qs = self._window_settings()
            geometry = qs.value("geometry")
            if geometry is not None:
                self.restoreGeometry(geometry)
            splitter = qs.value("splitter")
            if splitter is not None:
                self.splitter.restoreState(splitter)
        except Exception:
            log.warning("Could not restore window geometry", exc_info=True)

    def _save_geometry(self) -> None:
        try:
            qs = self._window_settings()
            qs.setValue("geometry", self.saveGeometry())
            qs.setValue("splitter", self.splitter.saveState())
            qs.sync()
        except Exception:
            log.warning("Could not save window geometry", exc_info=True)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._closed:
            event.accept()
            return
        self._closed = True
        if self._runner.is_busy():
            log.info("Window closing during a turn; cancelling it")
            self._runner.shutdown(SHUTDOWN_WAIT_S)  # cancels and denies any pending approval
        self._close_approval_dialog()
        # Late events from the worker must not touch the UI or the (closed) store.
        self._runner.blockSignals(True)
        if self._settings_dialog is not None:
            self._settings_dialog.reject()
        self._save_geometry()
        try:
            self._ctx.close()
        except Exception:
            log.exception("Error while closing the application context")
        event.accept()
        super().closeEvent(event)
