"""Settings dialog: General, API key, Tools and Workspace tabs.

All values are validated through :meth:`Settings.with_changes`; validation
errors are shown inline and nothing is saved until they are fixed. The API key
is stored only through :class:`SecretStore` (OS credential store) and is never
displayed in full - only its masked form and source are shown.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from jarvis import APP_DISPLAY_NAME
from jarvis.app_context import AppContext
from jarvis.config.secrets import API_KEY_ENV_VAR, KeySource, mask_secret
from jarvis.config.settings import SUGGESTED_MODELS, Settings, Theme
from jarvis.errors import ConfigError, JarvisError
from jarvis.tools.base import RiskLevel
from jarvis.ui.theme import repolish
from jarvis.ui.worker import BackgroundCall

log = logging.getLogger(__name__)

KEY_SOURCE_TEXT = {
    KeySource.ENVIRONMENT: f"Environment variable {API_KEY_ENV_VAR}",
    KeySource.KEYRING: "Operating-system credential store",
    KeySource.NONE: "Not configured",
}
CUSTOM_INSTRUCTIONS_LIMIT = 4000


def field_bounds(name: str, default: tuple[float, float]) -> tuple[float, float]:
    """Read ``ge``/``le`` constraints of a :class:`Settings` field."""
    low, high = default
    info = Settings.model_fields.get(name)
    for meta in getattr(info, "metadata", []) or []:
        if getattr(meta, "ge", None) is not None:
            low = meta.ge
        if getattr(meta, "le", None) is not None:
            high = meta.le
    return low, high


def _int_spin(name: str, value: int, suffix: str = "") -> QSpinBox:
    low, high = field_bounds(name, (0, 1_000_000))
    spin = QSpinBox()
    spin.setRange(int(low), int(high))
    spin.setValue(int(value))
    spin.setSuffix(suffix)
    spin.setAccelerated(True)
    return spin


def _float_spin(name: str, value: float, step: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
    low, high = field_bounds(name, (0.0, 1_000_000.0))
    spin = QDoubleSpinBox()
    spin.setRange(float(low), float(high))
    spin.setDecimals(decimals)
    spin.setSingleStep(step)
    spin.setValue(float(value))
    spin.setSuffix(suffix)
    return spin


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("Hint")
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    return label


class SettingsDialog(QDialog):
    settings_saved = Signal(object)  # Settings
    api_key_changed = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ctx = ctx
        self._initial = ctx.settings.settings
        self._models_call: BackgroundCall | None = None
        self._test_call: BackgroundCall | None = None
        self.setWindowTitle(f"{APP_DISPLAY_NAME} – Settings")
        self.setMinimumSize(640, 520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 12)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_general_tab(), "General")
        self.tabs.addTab(self._build_api_tab(), "API key")
        self.tabs.addTab(self._build_tools_tab(), "Tools")
        self.tabs.addTab(self._build_workspace_tab(), "Workspace")
        layout.addWidget(self.tabs, 1)

        self.error_label = QLabel()
        self.error_label.setObjectName("InlineError")
        self.error_label.setWordWrap(True)
        self.error_label.setTextFormat(Qt.TextFormat.PlainText)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.save_button = self.button_box.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setProperty("variant", "primary")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        self._refresh_key_status()

    # -- tabs ------------------------------------------------------------------------

    def _build_general_tab(self) -> QWidget:
        s = self._initial
        page = QWidget()
        form = QFormLayout(page)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        models = list(SUGGESTED_MODELS)
        if s.model not in models:
            models.insert(0, s.model)
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(s.model)
        self.refresh_models_button = QPushButton("Refresh")
        self.refresh_models_button.setToolTip("Fetch the models available to your Groq account")
        self.refresh_models_button.clicked.connect(self.refresh_models)
        model_row = QHBoxLayout()
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.refresh_models_button)
        self.models_status = _hint("")
        self.models_status.setVisible(False)
        model_box = QVBoxLayout()
        model_box.setSpacing(4)
        model_box.addLayout(model_row)
        model_box.addWidget(self.models_status)
        form.addRow("Model", model_box)

        self.temperature_spin = _float_spin("temperature", s.temperature, 0.1, 2)
        form.addRow("Temperature", self.temperature_spin)
        self.max_tokens_spin = _int_spin("max_tokens", s.max_tokens)
        self.max_tokens_spin.setSingleStep(256)
        form.addRow("Max response tokens", self.max_tokens_spin)
        self.max_iterations_spin = _int_spin("max_tool_iterations", s.max_tool_iterations)
        form.addRow("Max tool steps per message", self.max_iterations_spin)
        self.history_spin = _int_spin("max_history_messages", s.max_history_messages)
        form.addRow("History messages sent", self.history_spin)
        self.timeout_spin = _float_spin("request_timeout_seconds", s.request_timeout_seconds, 5.0, 0, " s")
        form.addRow("Request timeout", self.timeout_spin)
        self.stream_check = QCheckBox("Show responses while they are generated")
        self.stream_check.setChecked(s.stream_responses)
        form.addRow("Streaming", self.stream_check)

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Dark", Theme.DARK.value)
        self.theme_combo.addItem("Light", Theme.LIGHT.value)
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(s.theme.value)))
        form.addRow("Theme", self.theme_combo)

        self.instructions_edit = QPlainTextEdit()
        self.instructions_edit.setPlaceholderText("Optional: how the assistant should respond (tone, language, format)…")
        self.instructions_edit.setPlainText(s.custom_instructions)
        self.instructions_edit.setMinimumHeight(90)
        self.instructions_counter = _hint("")
        self.instructions_edit.textChanged.connect(self._update_instruction_counter)
        self._update_instruction_counter()
        form.addRow("Custom instructions", self.instructions_edit)
        form.addRow("", self.instructions_counter)
        return page

    def _build_api_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(
            _hint(
                "J.A.R.V.I.S. uses the Groq API. Create a key at console.groq.com/keys. The key is stored in "
                "the operating-system credential store and never written to settings or logs."
            )
        )
        form = QFormLayout()
        self.key_source_label = QLabel()
        self.key_source_label.setTextFormat(Qt.TextFormat.PlainText)
        form.addRow("Key source", self.key_source_label)
        self.key_masked_label = QLabel()
        self.key_masked_label.setTextFormat(Qt.TextFormat.PlainText)
        form.addRow("Current key", self.key_masked_label)
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_edit.setPlaceholderText("Paste a new API key (gsk_…)")
        self.key_edit.setClearButtonEnabled(True)
        self.key_edit.returnPressed.connect(self.save_api_key)
        form.addRow("New key", self.key_edit)
        layout.addLayout(form)

        row = QHBoxLayout()
        self.save_key_button = QPushButton("Save key")
        self.save_key_button.setProperty("variant", "primary")
        self.save_key_button.setAutoDefault(False)
        self.save_key_button.clicked.connect(self.save_api_key)
        self.remove_key_button = QPushButton("Remove key…")
        self.remove_key_button.setAutoDefault(False)
        self.remove_key_button.clicked.connect(self.remove_api_key)
        self.test_button = QPushButton("Test connection")
        self.test_button.setAutoDefault(False)
        self.test_button.clicked.connect(self.test_connection)
        row.addWidget(self.save_key_button)
        row.addWidget(self.remove_key_button)
        row.addStretch(1)
        row.addWidget(self.test_button)
        layout.addLayout(row)

        self.key_status = QLabel()
        self.key_status.setWordWrap(True)
        self.key_status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.key_status)
        self.env_note = _hint(
            f"The {API_KEY_ENV_VAR} environment variable is set and takes precedence over the stored key."
        )
        layout.addWidget(self.env_note)
        layout.addStretch(1)
        return page

    def _build_tools_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(
            _hint(
                "Unchecked tools are never offered to the assistant. Tools marked “Needs approval” always ask "
                "for your confirmation before they run."
            )
        )
        self.tools_tree = QTreeWidget()
        self.tools_tree.setColumnCount(3)
        self.tools_tree.setHeaderLabels(["Tool", "Risk", "Description"])
        self.tools_tree.setRootIsDecorated(False)
        self.tools_tree.setWordWrap(True)
        self.tools_tree.setUniformRowHeights(False)
        self.tools_tree.setAlternatingRowColors(True)
        header = self.tools_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        disabled = set(self._initial.disabled_tools)
        self._tool_items: dict[str, QTreeWidgetItem] = {}
        for tool in self._ctx.registry.all():
            consequential = tool.risk is RiskLevel.CONSEQUENTIAL
            item = QTreeWidgetItem([tool.name, "Needs approval" if consequential else "Safe", tool.description])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Unchecked if tool.name in disabled else Qt.CheckState.Checked)
            item.setToolTip(2, tool.description)
            self.tools_tree.addTopLevelItem(item)
            self._tool_items[tool.name] = item
        layout.addWidget(self.tools_tree, 1)
        return page

    def _build_workspace_tab(self) -> QWidget:
        s = self._initial
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(
            _hint(
                "The workspace is the folder where the assistant creates files. Reading inside it is automatic; "
                "anything outside it, and every change, requires your approval."
            )
        )
        form = QFormLayout()
        self.workspace_edit = QLineEdit(s.workspace_dir)
        browse = QPushButton("Browse…")
        browse.setAutoDefault(False)
        browse.clicked.connect(self._browse_workspace)
        row = QHBoxLayout()
        row.addWidget(self.workspace_edit, 1)
        row.addWidget(browse)
        form.addRow("Workspace folder", row)
        self.shell_timeout_spin = _int_spin("shell_timeout_seconds", s.shell_timeout_seconds, " s")
        form.addRow("Command timeout", self.shell_timeout_spin)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    # -- general helpers -------------------------------------------------------------

    def _update_instruction_counter(self) -> None:
        count = len(self.instructions_edit.toPlainText())
        self.instructions_counter.setText(f"{count} / {CUSTOM_INSTRUCTIONS_LIMIT} characters")

    def _browse_workspace(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose workspace folder", self.workspace_edit.text())
        if directory:
            self.workspace_edit.setText(directory)

    def collect_changes(self) -> dict[str, Any]:
        disabled = [n for n in self._initial.disabled_tools if n not in self._tool_items]  # keep unknown names
        disabled += [
            name for name, item in self._tool_items.items() if item.checkState(0) != Qt.CheckState.Checked
        ]
        return {
            "model": self.model_combo.currentText(),
            "temperature": round(self.temperature_spin.value(), 2),
            "max_tokens": self.max_tokens_spin.value(),
            "max_tool_iterations": self.max_iterations_spin.value(),
            "max_history_messages": self.history_spin.value(),
            "request_timeout_seconds": float(self.timeout_spin.value()),
            "stream_responses": self.stream_check.isChecked(),
            "theme": self.theme_combo.currentData() or Theme.DARK.value,
            "custom_instructions": self.instructions_edit.toPlainText(),
            "workspace_dir": self.workspace_edit.text(),
            "shell_timeout_seconds": self.shell_timeout_spin.value(),
            "disabled_tools": disabled,
        }

    def show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))

    def save(self) -> bool:
        """Validate and persist. Returns ``False`` (and shows why) on failure."""
        current = self._ctx.settings.settings
        try:
            new = current.with_changes(**self.collect_changes())
        except ConfigError as exc:
            self.show_error(exc.user_message)
            return False
        try:
            self._ctx.settings.save(new)
        except ConfigError as exc:
            self.show_error(exc.user_message)
            return False
        self.show_error("")
        if new.request_timeout_seconds != current.request_timeout_seconds:
            self._ctx.invalidate_client()
        self.settings_saved.emit(new)
        return True

    def accept(self) -> None:
        if self.save():
            super().accept()

    # -- models ----------------------------------------------------------------------

    def refresh_models(self) -> None:
        if self._models_call is not None and self._models_call.is_running():
            return
        self.refresh_models_button.setEnabled(False)
        self._set_models_status("Fetching available models…", error=False)
        factory = self._ctx.client_factory
        call = BackgroundCall(lambda: factory().list_models(), "Listing models")
        call.succeeded.connect(self._on_models_loaded)
        call.failed.connect(self._on_models_failed)
        self._models_call = call.start()

    def _on_models_loaded(self, models: Any) -> None:
        self.refresh_models_button.setEnabled(True)
        current = self.model_combo.currentText()
        names = [str(m) for m in (models or [])]
        if not names:
            self._set_models_status("The account returned no models; keeping the suggested list.", error=True)
            return
        self.model_combo.clear()
        self.model_combo.addItems(names)
        self.model_combo.setCurrentText(current)
        self._set_models_status(f"Loaded {len(names)} models from Groq.", error=False)

    def _on_models_failed(self, message: str) -> None:
        self.refresh_models_button.setEnabled(True)
        self._set_models_status(f"Could not fetch models: {message}", error=True)

    def _set_models_status(self, text: str, *, error: bool) -> None:
        self.models_status.setObjectName("InlineError" if error else "Hint")
        repolish(self.models_status)
        self.models_status.setText(text)
        self.models_status.setVisible(bool(text))

    # -- API key ---------------------------------------------------------------------

    def _refresh_key_status(self) -> None:
        secrets = self._ctx.secrets
        source = secrets.source()
        self.key_source_label.setText(KEY_SOURCE_TEXT.get(source, str(source)))
        self.key_masked_label.setText(mask_secret(secrets.get_api_key()) or "—")
        # With the environment variable set we cannot tell (no public API) whether a key is also
        # stored, so Remove stays available; deleting a missing key is a no-op.
        self.remove_key_button.setEnabled(secrets.has_stored_key())
        self.env_note.setVisible(source is KeySource.ENVIRONMENT)

    def _set_key_status(self, text: str, *, error: bool) -> None:
        self.key_status.setObjectName("InlineError" if error else "InlineSuccess")
        repolish(self.key_status)
        self.key_status.setText(text)

    def save_api_key(self) -> None:
        key = self.key_edit.text().strip()
        if not key:
            self._set_key_status("Enter a key first.", error=True)
            return
        try:
            self._ctx.secrets.set_api_key(key)
        except JarvisError as exc:
            self._set_key_status(exc.user_message, error=True)
            return
        self.key_edit.clear()
        self._ctx.invalidate_client()
        self._refresh_key_status()
        self._set_key_status("API key saved to the credential store.", error=False)
        self.api_key_changed.emit()

    def remove_api_key(self) -> None:
        answer = QMessageBox.question(
            self,
            "Remove API key",
            "Remove the stored Groq API key from the credential store?\n\n"
            "J.A.R.V.I.S. will not be able to answer until a new key is configured.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._ctx.secrets.delete_api_key()
        except JarvisError as exc:
            self._set_key_status(exc.user_message, error=True)
            return
        self._ctx.invalidate_client()
        self._refresh_key_status()
        still = self._ctx.secrets.source() is KeySource.ENVIRONMENT
        self._set_key_status(
            "Stored key removed." + (f" The {API_KEY_ENV_VAR} environment variable is still in use." if still else ""),
            error=False,
        )
        self.api_key_changed.emit()

    def test_connection(self) -> None:
        if self._test_call is not None and self._test_call.is_running():
            return
        if self.key_edit.text().strip():
            self._set_key_status("Save the new key first, then test the connection.", error=True)
            return
        self.test_button.setEnabled(False)
        self._set_key_status("Testing connection…", error=False)
        factory = self._ctx.client_factory
        call = BackgroundCall(lambda: factory().list_models(), "Testing connection")
        call.succeeded.connect(self._on_test_succeeded)
        call.failed.connect(self._on_test_failed)
        self._test_call = call.start()

    def _on_test_succeeded(self, models: Any) -> None:
        self.test_button.setEnabled(True)
        count = len(models or [])
        self._set_key_status(f"Connection OK – {count} models available.", error=False)

    def _on_test_failed(self, message: str) -> None:
        self.test_button.setEnabled(True)
        self._set_key_status(f"Connection failed: {message}", error=True)
