"""Application entry point (``jarvis`` gui-script and ``python -m jarvis``)."""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import threading
from collections.abc import Callable, Sequence
from types import TracebackType

from jarvis import APP_DISPLAY_NAME, __version__
from jarvis.logging_setup import configure_logging
from jarvis.paths import AppPaths

log = logging.getLogger("jarvis.app")

SMOKE_TEST_DURATION_MS = 1500
SMOKE_TEST_WATCHDOG_MS = 15000
ORGANIZATION_NAME = "JARVIS"
APP_USER_MODEL_ID = "JARVIS.Desktop"


def _parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(prog="jarvis", description=f"{APP_DISPLAY_NAME} desktop AI assistant")
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="start the application, show the main window and quit automatically (for CI)",
    )
    args, rest = parser.parse_known_args(list(argv))
    return args, rest  # unknown arguments are passed through to Qt (e.g. -platform offscreen)


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)  # type: ignore[attr-defined]
    except Exception:  # cosmetic only (taskbar grouping/icon)
        log.debug("Could not set AppUserModelID", exc_info=True)


def _make_error_reporter(interactive: bool):  # -> _ErrorReporter (Qt imported lazily)
    from PySide6.QtCore import QObject, Qt, Signal
    from PySide6.QtWidgets import QMessageBox

    class _ErrorReporter(QObject):
        """Shows a non-fatal error dialog on the GUI thread (signal is queued from any thread)."""

        report = Signal(str)

        def __init__(self) -> None:
            super().__init__()
            self._box: QMessageBox | None = None
            self.report.connect(self._show, Qt.ConnectionType.QueuedConnection)

        def _show(self, summary: str) -> None:
            if not interactive or self._box is not None:
                return  # never stack error dialogs
            box = QMessageBox(QMessageBox.Icon.Warning, APP_DISPLAY_NAME, "An unexpected error occurred.")
            box.setInformativeText(
                f"{summary}\n\nThe application will keep running. Details were written to the log file "
                "(Help › Open Log Folder)."
            )
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.finished.connect(self._on_closed)
            self._box = box
            box.open()

        def _on_closed(self, _code: int) -> None:
            box, self._box = self._box, None
            if box is not None:
                box.deleteLater()

    return _ErrorReporter()


def _install_exception_hooks(report: Callable[[str], None]) -> Callable[[], None]:
    """Log uncaught exceptions (main and worker threads) instead of crashing."""
    previous_sys_hook = sys.excepthook
    previous_thread_hook = threading.excepthook

    def sys_hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            previous_sys_hook(exc_type, exc, tb)
            return
        log.critical("Unhandled exception", exc_info=(exc_type, exc, tb))
        report(f"{exc_type.__name__} in the user interface.")

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "unknown"
        log.critical(
            "Unhandled exception in thread %s", name, exc_info=(args.exc_type, args.exc_value, args.exc_traceback)
        )
        report(f"{args.exc_type.__name__} in background thread '{name}'.")

    sys.excepthook = sys_hook
    threading.excepthook = thread_hook

    def restore() -> None:
        sys.excepthook = previous_sys_hook
        threading.excepthook = previous_thread_hook

    return restore


def main(argv: Sequence[str] | None = None) -> int:
    args, qt_args = _parse_args(sys.argv[1:] if argv is None else argv)
    if args.version:
        print(f"{APP_DISPLAY_NAME} {__version__}")
        return 0
    interactive = not args.smoke_test

    try:
        paths = AppPaths.default().ensure()
    except OSError as exc:
        print(f"{APP_DISPLAY_NAME}: could not create application folders: {exc}", file=sys.stderr or sys.__stdout__)
        return 1
    configure_logging(paths.log_file)
    log.info(
        "Starting %s %s (Python %s, %s)%s",
        APP_DISPLAY_NAME,
        __version__,
        platform.python_version(),
        platform.platform(),
        " [smoke test]" if args.smoke_test else "",
    )

    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication, QMessageBox

    from jarvis.app_context import AppContext
    from jarvis.errors import JarvisError
    from jarvis.ui.main_window import MainWindow
    from jarvis.ui.theme import apply_theme

    _set_windows_app_id()
    if QApplication.instance() is None:
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication.instance() or QApplication([sys.argv[0] if sys.argv else "jarvis", *qt_args])
    assert isinstance(app, QApplication)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setApplicationVersion(__version__)
    app.setStyle("Fusion")

    reporter = _make_error_reporter(interactive)

    def report(summary: str) -> None:
        try:
            reporter.report.emit(summary)
        except RuntimeError:  # reporter already destroyed during shutdown
            pass

    restore_hooks = _install_exception_hooks(report)
    ctx = None
    try:
        try:
            ctx = AppContext.create(paths)
        except JarvisError as exc:
            log.error("Start-up failed: %s", exc)
            if interactive:
                QMessageBox.critical(None, APP_DISPLAY_NAME, exc.user_message)
            return 1
        except Exception:
            log.exception("Start-up failed unexpectedly")
            if interactive:
                QMessageBox.critical(
                    None,
                    APP_DISPLAY_NAME,
                    f"{APP_DISPLAY_NAME} could not start because of an unexpected error.\n\n"
                    f"Details were written to:\n{paths.log_file}",
                )
            return 1

        apply_theme(app, ctx.settings.settings.theme)
        window = MainWindow(ctx)
        window.show()
        if args.smoke_test:
            QTimer.singleShot(SMOKE_TEST_DURATION_MS, window.close)
            QTimer.singleShot(SMOKE_TEST_WATCHDOG_MS, lambda: app.exit(2))  # never hang CI
        code = int(app.exec())
        log.info("%s exited with code %s", APP_DISPLAY_NAME, code)
        return code
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:
                log.exception("Error while closing the application context")
        restore_hooks()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
