"""Approval dialog and settings dialog tests."""

from __future__ import annotations

import json

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QMessageBox
from ui_support import TEST_API_KEY, all_widget_texts

from jarvis.agent.events import ApprovalRequest
from jarvis.config.secrets import API_KEY_USERNAME, SERVICE_NAME, mask_secret
from jarvis.errors import LLMAuthenticationError
from jarvis.llm.base import ToolCall
from jarvis.tools.base import RiskLevel
from jarvis.ui.approval_dialog import ApprovalDialog
from jarvis.ui.settings_dialog import SettingsDialog

pytestmark = pytest.mark.gui


def make_request(summary: str = "Run a command", arguments: dict | None = None) -> ApprovalRequest:
    arguments = arguments if arguments is not None else {"command": "echo hi", "cwd": "C:/work"}
    call = ToolCall("call_1", "run_command", json.dumps(arguments))
    return ApprovalRequest(
        call, "run_command", summary, RiskLevel.CONSEQUENTIAL, "This action can run programs.", arguments
    )


# --------------------------------------------------------------------------- approval dialog


def test_approval_dialog_defaults_to_deny(qtbot):
    dialog = ApprovalDialog(make_request())
    qtbot.addWidget(dialog)
    assert dialog.deny_button.isDefault()
    assert not dialog.allow_button.isDefault() and not dialog.allow_button.autoDefault()
    with qtbot.waitExposed(dialog):
        dialog.open()
    assert dialog.focusWidget() is dialog.deny_button

    qtbot.keyClick(dialog, Qt.Key.Key_Escape)
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert dialog.approved is False


def test_approval_dialog_enter_denies(qtbot):
    dialog = ApprovalDialog(make_request())
    qtbot.addWidget(dialog)
    with qtbot.waitExposed(dialog):
        dialog.open()
    qtbot.keyClick(dialog.deny_button, Qt.Key.Key_Return)
    assert not dialog.isVisible()
    assert dialog.approved is False


def test_approval_dialog_close_denies(qtbot):
    dialog = ApprovalDialog(make_request())
    qtbot.addWidget(dialog)
    with qtbot.waitExposed(dialog):
        dialog.open()
    dialog.close()
    assert dialog.approved is False


def test_approval_dialog_allow_once(qtbot):
    dialog = ApprovalDialog(make_request())
    qtbot.addWidget(dialog)
    with qtbot.waitExposed(dialog):
        dialog.open()
    qtbot.mouseClick(dialog.allow_button, Qt.MouseButton.LeftButton)
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.approved is True


def test_approval_dialog_shows_full_summary_and_arguments_literally(qtbot):
    summary = "Write <b>file</b> " + "x" * 5000 + " END"
    arguments = {"path": "C:/tmp/<script>.txt", "content": "y" * 3000}
    dialog = ApprovalDialog(make_request(summary, arguments))
    qtbot.addWidget(dialog)
    assert dialog.summary_view.toPlainText() == summary  # no truncation, no HTML interpretation
    assert dialog.summary_view.isReadOnly() and dialog.arguments_view.isReadOnly()
    assert json.loads(dialog.arguments_view.toPlainText()) == arguments
    assert dialog.tool_label.text() == "run_command"
    assert dialog.risk_label.text() == "Consequential"
    assert dialog.reason_label.text() == "This action can run programs."


# --------------------------------------------------------------------------- settings dialog


def open_settings(qtbot, ctx) -> SettingsDialog:
    dialog = SettingsDialog(ctx)
    qtbot.addWidget(dialog)
    dialog.show()
    return dialog


def test_settings_validation_error_is_shown_and_nothing_saved(qtbot, make_harness):
    h = make_harness()
    dialog = open_settings(qtbot, h.ctx)
    dialog.model_combo.setCurrentText("   ")
    dialog.save_button.click()
    assert dialog.isVisible()
    assert dialog.error_label.isVisible()
    assert "model" in dialog.error_label.text()
    assert not h.ctx.settings.path.exists()

    dialog.model_combo.setCurrentText("llama-3.1-8b-instant")
    dialog.instructions_edit.setPlainText("z" * 4001)
    dialog.save_button.click()
    assert dialog.isVisible()
    assert "custom_instructions" in dialog.error_label.text()
    assert not h.ctx.settings.path.exists()


def test_settings_save_persists_to_json(qtbot, make_harness, tmp_path):
    h = make_harness()
    dialog = open_settings(qtbot, h.ctx)
    saved: list[object] = []
    dialog.settings_saved.connect(saved.append)
    workspace = tmp_path / "my-workspace"

    dialog.model_combo.setCurrentText("openai/gpt-oss-20b")
    dialog.temperature_spin.setValue(0.9)
    dialog.max_tokens_spin.setValue(2048)
    dialog.max_iterations_spin.setValue(5)
    dialog.history_spin.setValue(20)
    dialog.stream_check.setChecked(False)
    dialog.theme_combo.setCurrentIndex(dialog.theme_combo.findData("light"))
    dialog.instructions_edit.setPlainText("Answer in French.")
    dialog.workspace_edit.setText(str(workspace))
    dialog.shell_timeout_spin.setValue(30)
    dialog._tool_items["write_marker"].setCheckState(0, Qt.CheckState.Unchecked)
    dialog.save_button.click()

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert not dialog.error_label.isVisible()
    data = json.loads(h.ctx.settings.path.read_text(encoding="utf-8"))
    assert data["model"] == "openai/gpt-oss-20b"
    assert data["temperature"] == pytest.approx(0.9)
    assert data["max_tokens"] == 2048
    assert data["max_tool_iterations"] == 5
    assert data["max_history_messages"] == 20
    assert data["stream_responses"] is False
    assert data["theme"] == "light"
    assert data["custom_instructions"] == "Answer in French."
    assert data["workspace_dir"] == str(workspace)
    assert data["shell_timeout_seconds"] == 30
    assert data["disabled_tools"] == ["write_marker"]
    assert h.ctx.settings.settings.model == "openai/gpt-oss-20b"
    assert len(saved) == 1
    assert h.factory.invalidations == 0  # timeout unchanged


def test_settings_timeout_change_invalidates_client(qtbot, make_harness):
    h = make_harness()
    dialog = open_settings(qtbot, h.ctx)
    dialog.timeout_spin.setValue(120)
    assert dialog.save()
    assert h.ctx.settings.settings.request_timeout_seconds == 120
    assert h.factory.invalidations == 1


def test_settings_tools_tab_lists_registry(qtbot, make_harness):
    h = make_harness()
    dialog = open_settings(qtbot, h.ctx)
    rows = {
        dialog.tools_tree.topLevelItem(i).text(0): dialog.tools_tree.topLevelItem(i)
        for i in range(dialog.tools_tree.topLevelItemCount())
    }
    assert set(rows) == {"echo_text", "write_marker"}
    assert rows["echo_text"].text(1) == "Safe"
    assert rows["write_marker"].text(1) == "Needs approval"
    assert rows["write_marker"].text(2) == "Create a marker file in the workspace."
    assert all(item.checkState(0) == Qt.CheckState.Checked for item in rows.values())


def test_api_key_save_and_remove_never_shows_full_key(qtbot, make_harness, monkeypatch):
    h = make_harness(with_key=False)
    dialog = open_settings(qtbot, h.ctx)
    changed: list[bool] = []
    dialog.api_key_changed.connect(lambda: changed.append(True))
    assert dialog.key_source_label.text() == "Not configured"
    assert not dialog.remove_key_button.isEnabled()

    # An invalid key is rejected with a friendly message and nothing is stored.
    dialog.key_edit.setText("short")
    dialog.save_key_button.click()
    assert "valid API key" in dialog.key_status.text()
    assert h.keyring.store == {}

    dialog.key_edit.setText(TEST_API_KEY)
    assert dialog.key_edit.echoMode() == dialog.key_edit.EchoMode.Password
    dialog.save_key_button.click()
    assert h.keyring.store[(SERVICE_NAME, API_KEY_USERNAME)] == TEST_API_KEY
    assert dialog.key_edit.text() == ""
    assert dialog.key_source_label.text() == "Operating-system credential store"
    assert dialog.key_masked_label.text() == mask_secret(TEST_API_KEY)
    assert h.factory.invalidations == 1
    assert changed == [True]
    texts = all_widget_texts(dialog)
    assert not any(TEST_API_KEY in t for t in texts)
    assert not any(TEST_API_KEY[4:-4] in t for t in texts)

    # Remove asks first (default No); answering No keeps the key.
    asked: list[tuple] = []

    def answer(value):
        def question(*args, **kwargs):
            asked.append(args)
            return value

        return staticmethod(question)

    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.No))
    dialog.remove_key_button.click()
    assert h.keyring.store != {}
    assert asked[-1][4] == QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", answer(QMessageBox.StandardButton.Yes))
    dialog.remove_key_button.click()
    assert h.keyring.store == {}
    assert dialog.key_source_label.text() == "Not configured"
    assert h.factory.invalidations == 2
    assert not any(TEST_API_KEY in t for t in all_widget_texts(dialog))


def test_api_key_never_shown_in_main_window(qtbot, make_harness, make_window):
    h = make_harness()
    window = make_window(h.ctx)
    dialog = window.open_settings()
    for root in (window, dialog):
        assert not any(TEST_API_KEY in t for t in all_widget_texts(root))
    dialog.reject()


def test_test_connection_reports_success_and_failure(qtbot, make_harness):
    h = make_harness()
    dialog = open_settings(qtbot, h.ctx)
    dialog.test_button.click()
    qtbot.waitUntil(lambda: "Connection OK" in dialog.key_status.text(), timeout=5000)
    assert "2 models" in dialog.key_status.text()
    assert dialog.test_button.isEnabled()

    h.client.list_models_error = LLMAuthenticationError("401 raw", user_message="Groq rejected the API key.")
    dialog.test_button.click()
    qtbot.waitUntil(lambda: "Connection failed" in dialog.key_status.text(), timeout=5000)
    assert "Groq rejected the API key." in dialog.key_status.text()
    assert "401 raw" not in dialog.key_status.text()


def test_refresh_models_populates_combo_and_reports_errors(qtbot, make_harness):
    h = make_harness()
    h.client.models = ["model-a", "model-b", "model-c"]
    dialog = open_settings(qtbot, h.ctx)
    current = dialog.model_combo.currentText()
    dialog.refresh_models_button.click()
    qtbot.waitUntil(lambda: "Loaded 3 models" in dialog.models_status.text(), timeout=5000)
    assert [dialog.model_combo.itemText(i) for i in range(dialog.model_combo.count())] == h.client.models
    assert dialog.model_combo.currentText() == current

    h.client.list_models_error = LLMAuthenticationError("401", user_message="Groq rejected the API key.")
    dialog.refresh_models_button.click()
    qtbot.waitUntil(lambda: "Could not fetch models" in dialog.models_status.text(), timeout=5000)
    assert "Groq rejected the API key." in dialog.models_status.text()


def test_refresh_models_without_key_shows_missing_key(qtbot, make_harness):
    h = make_harness(with_key=False, real_client_factory=True)
    dialog = open_settings(qtbot, h.ctx)
    dialog.refresh_models_button.click()
    qtbot.waitUntil(lambda: "Could not fetch models" in dialog.models_status.text(), timeout=5000)
    assert "No Groq API key" in dialog.models_status.text()
