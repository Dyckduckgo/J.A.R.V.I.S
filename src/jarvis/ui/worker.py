"""Background execution of chat turns with a thread-safe bridge to the GUI.

Threading model
---------------
* :meth:`ChatService.send_message` is blocking, so every turn runs on a
  dedicated **daemon** worker thread. A daemon thread is used (rather than a
  ``QThread``) so that a network call or tool that ignores cancellation can
  never keep the process alive after the window closed, and so that Qt never
  aborts with "QThread destroyed while thread is still running".
* :class:`_BridgeCallbacks` implements :class:`AgentCallbacks` on the worker
  thread. Each callback emits a signal of a private :class:`_Bridge` object;
  those signals are connected with ``Qt.QueuedConnection`` to slots of
  :class:`TurnRunner`, which lives on the GUI thread. The runner re-emits
  public signals, so *every* public signal is emitted on the GUI thread no
  matter how listeners connect to it.
* ``request_approval`` blocks the worker on a :class:`PendingApproval`
  (``threading.Event``) until the GUI answers. It returns ``False`` when the
  user denies, stops the turn, closes the window, or no answer arrives within
  the approval timeout. It never approves on its own.
* Only one turn runs at a time; :meth:`TurnRunner.start` refuses a second one.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QObject, Qt, Signal, Slot

from jarvis.agent.events import AgentCallbacks, ApprovalRequest, TurnResult
from jarvis.agent.service import ChatService
from jarvis.errors import JarvisError, MissingAPIKeyError, OperationCancelled
from jarvis.llm.base import ChatMessage, ToolCall
from jarvis.tools.base import ToolResult
from jarvis.tools.policy import Decision

log = logging.getLogger(__name__)

GENERIC_ERROR_MESSAGE = (
    "Something went wrong while processing your message. The details were written to the log file."
)
DEFAULT_APPROVAL_TIMEOUT_S = 15 * 60.0
_POLL_INTERVAL_S = 0.05


class PendingApproval:
    """One approval request waiting for the user's answer.

    :meth:`resolve` is idempotent: the first answer wins, later ones are
    ignored, so a late click can never flip a "deny" caused by Stop.
    """

    def __init__(self, request: ApprovalRequest) -> None:
        self.request = request
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._approved = False
        self.resolved_by: str | None = None

    @property
    def resolved(self) -> bool:
        return self._event.is_set()

    @property
    def approved(self) -> bool:
        return self._event.is_set() and self._approved

    def resolve(self, approved: bool, by: str = "user") -> bool:
        """Record the answer. Returns ``True`` if this call decided it."""
        with self._lock:
            if self._event.is_set():
                return False
            self._approved = approved is True
            self.resolved_by = by
            self._event.set()
            return True

    def wait(self, cancel: threading.Event, timeout: float | None) -> bool:
        """Block until answered, cancelled or timed out; returns the decision."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._event.wait(_POLL_INTERVAL_S):
            if cancel.is_set():
                self.resolve(False, by="cancel")
            elif deadline is not None and time.monotonic() >= deadline:
                log.info("Approval for %s timed out; treating as denied", self.request.tool_name)
                self.resolve(False, by="timeout")
        return self.approved and not cancel.is_set()


@dataclass(frozen=True)
class TurnOutcome:
    """Final state of a turn, delivered once on the GUI thread."""

    conversation_id: int
    text: str
    result: TurnResult | None = None
    error: str | None = None
    exception: BaseException | None = None
    cancelled: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def missing_api_key(self) -> bool:
        return isinstance(self.exception, MissingAPIKeyError)


class _Bridge(QObject):
    """Signals emitted on the worker thread (delivered queued to the runner)."""

    message = Signal(object)
    llm_started = Signal(int)
    text_delta = Signal(str)
    tool_started = Signal(object, str, object)
    tool_finished = Signal(object, object)
    approval_needed = Signal(object)
    approval_expired = Signal(object)
    outcome = Signal(object)


class _Turn:
    def __init__(self, conversation_id: int, text: str, model: str | None) -> None:
        self.conversation_id = conversation_id
        self.text = text
        self.model = model
        self.cancel = threading.Event()
        self.thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._pending: PendingApproval | None = None

    def register_pending(self, pending: PendingApproval) -> bool:
        with self._lock:
            if self.cancel.is_set():
                pending.resolve(False, by="cancel")
                return False
            self._pending = pending
            return True

    def clear_pending(self, pending: PendingApproval) -> None:
        with self._lock:
            if self._pending is pending:
                self._pending = None

    def cancel_turn(self) -> PendingApproval | None:
        """Set the cancel flag and deny any pending approval; returns it if it was open."""
        with self._lock:
            self.cancel.set()
            pending = self._pending
        if pending is not None and pending.resolve(False, by="cancel"):
            return pending
        return None


class _BridgeCallbacks(AgentCallbacks):
    """Runs on the worker thread; forwards every event through the bridge."""

    def __init__(self, bridge: _Bridge, turn: _Turn, approval_timeout: float | None) -> None:
        self._bridge = bridge
        self._turn = turn
        self._approval_timeout = approval_timeout

    def _emit(self, signal: Any, *args: Any) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:  # the GUI side is gone (window closed)
            log.debug("Dropped UI event; bridge no longer exists")

    def on_message(self, message: ChatMessage) -> None:
        self._emit(self._bridge.message, message)

    def on_llm_started(self, iteration: int) -> None:
        self._emit(self._bridge.llm_started, iteration)

    def on_text_delta(self, text: str) -> None:
        self._emit(self._bridge.text_delta, text)

    def on_tool_started(self, call: ToolCall, summary: str, decision: Decision) -> None:
        self._emit(self._bridge.tool_started, call, summary, decision)

    def on_tool_finished(self, call: ToolCall, result: ToolResult) -> None:
        self._emit(self._bridge.tool_finished, call, result)

    def request_approval(self, request: ApprovalRequest) -> bool:
        pending = PendingApproval(request)
        if not self._turn.register_pending(pending):
            return False
        try:
            self._emit(self._bridge.approval_needed, pending)
            approved = pending.wait(self._turn.cancel, self._approval_timeout)
            if pending.resolved_by == "timeout":
                self._emit(self._bridge.approval_expired, pending)
            return approved
        finally:
            self._turn.clear_pending(pending)


class TurnRunner(QObject):
    """Owns the (single) running chat turn. All public signals fire on the GUI thread."""

    busy_changed = Signal(bool)
    message_received = Signal(object)  # ChatMessage
    llm_started = Signal(int)
    text_received = Signal(str)
    tool_started = Signal(object, str, object)  # ToolCall, summary, Decision
    tool_finished = Signal(object, object)  # ToolCall, ToolResult
    approval_requested = Signal(object)  # PendingApproval
    approval_closed = Signal(object)  # PendingApproval resolved without the user (stop/timeout)
    turn_finished = Signal(object)  # TurnOutcome

    def __init__(
        self,
        chat: ChatService,
        parent: QObject | None = None,
        *,
        approval_timeout: float | None = DEFAULT_APPROVAL_TIMEOUT_S,
    ) -> None:
        super().__init__(parent)
        self._chat = chat
        self._approval_timeout = approval_timeout
        self._turn: _Turn | None = None
        self._threads: list[threading.Thread] = []

    # -- state -----------------------------------------------------------------

    def is_busy(self) -> bool:
        return self._turn is not None

    @property
    def conversation_id(self) -> int | None:
        return self._turn.conversation_id if self._turn else None

    # -- control ---------------------------------------------------------------

    def start(self, conversation_id: int, text: str, *, model: str | None = None) -> bool:
        """Start a turn on a worker thread. Returns ``False`` if one is already running."""
        if self._turn is not None:
            log.warning("A turn is already running; ignoring new request")
            return False
        turn = _Turn(conversation_id, text, model)
        bridge = _Bridge()  # no parent: kept alive by the worker closure
        queued = Qt.ConnectionType.QueuedConnection
        bridge.message.connect(self._on_message, queued)
        bridge.llm_started.connect(self._on_llm_started, queued)
        bridge.text_delta.connect(self._on_text_delta, queued)
        bridge.tool_started.connect(self._on_tool_started, queued)
        bridge.tool_finished.connect(self._on_tool_finished, queued)
        bridge.approval_needed.connect(self._on_approval_needed, queued)
        bridge.approval_expired.connect(self._on_approval_expired, queued)
        bridge.outcome.connect(self._on_outcome, queued)
        callbacks = _BridgeCallbacks(bridge, turn, self._approval_timeout)

        thread = threading.Thread(
            target=_run_turn,
            args=(self._chat, turn, callbacks, bridge),
            name=f"jarvis-turn-{conversation_id}",
            daemon=True,
        )
        turn.thread = thread
        self._turn = turn
        self._threads = [t for t in self._threads if t.is_alive()] + [thread]
        self.busy_changed.emit(True)
        thread.start()
        return True

    def stop(self) -> None:
        """Cancel the running turn and deny any pending approval."""
        turn = self._turn
        if turn is None:
            return
        log.info("Stop requested for conversation %s", turn.conversation_id)
        pending = turn.cancel_turn()
        if pending is not None:
            self.approval_closed.emit(pending)

    def answer_approval(self, pending: PendingApproval, approved: bool) -> None:
        """Called by the GUI with the user's decision."""
        pending.resolve(approved is True, by="user")

    def wait_for_workers(self, timeout: float) -> bool:
        """Join worker threads (GUI thread blocks up to ``timeout`` seconds)."""
        deadline = time.monotonic() + max(0.0, timeout)
        for thread in list(self._threads):
            remaining = deadline - time.monotonic()
            if remaining > 0:
                thread.join(remaining)
        alive = [t for t in self._threads if t.is_alive()]
        self._threads = alive
        return not alive

    def shutdown(self, timeout: float = 3.0) -> bool:
        """Stop the turn and wait briefly for the worker. Returns ``True`` if it ended."""
        self.stop()
        finished = self.wait_for_workers(timeout)
        if not finished:
            log.warning("Worker thread did not finish within %.1fs; abandoning it (daemon)", timeout)
        return finished

    # -- slots (GUI thread) ----------------------------------------------------

    @Slot(object)
    def _on_message(self, message: ChatMessage) -> None:
        self.message_received.emit(message)

    @Slot(int)
    def _on_llm_started(self, iteration: int) -> None:
        self.llm_started.emit(iteration)

    @Slot(str)
    def _on_text_delta(self, text: str) -> None:
        self.text_received.emit(text)

    @Slot(object, str, object)
    def _on_tool_started(self, call: ToolCall, summary: str, decision: Decision) -> None:
        self.tool_started.emit(call, summary, decision)

    @Slot(object, object)
    def _on_tool_finished(self, call: ToolCall, result: ToolResult) -> None:
        self.tool_finished.emit(call, result)

    @Slot(object)
    def _on_approval_needed(self, pending: PendingApproval) -> None:
        if pending.resolved:  # stopped before the GUI saw it
            return
        self.approval_requested.emit(pending)

    @Slot(object)
    def _on_approval_expired(self, pending: PendingApproval) -> None:
        self.approval_closed.emit(pending)

    @Slot(object)
    def _on_outcome(self, outcome: TurnOutcome) -> None:
        self._turn = None
        self.busy_changed.emit(False)
        self.turn_finished.emit(outcome)


def _run_turn(chat: ChatService, turn: _Turn, callbacks: _BridgeCallbacks, bridge: _Bridge) -> None:
    """Worker-thread body. Always delivers exactly one :class:`TurnOutcome`."""
    outcome = TurnOutcome(turn.conversation_id, turn.text, error=GENERIC_ERROR_MESSAGE)
    try:
        result = chat.send_message(turn.conversation_id, turn.text, callbacks, turn.cancel, model=turn.model)
        outcome = TurnOutcome(turn.conversation_id, turn.text, result=result, cancelled=result.cancelled)
    except OperationCancelled:
        outcome = TurnOutcome(turn.conversation_id, turn.text, cancelled=True)
    except JarvisError as exc:
        log.warning("Chat turn failed: %s: %s", type(exc).__name__, exc)
        outcome = TurnOutcome(turn.conversation_id, turn.text, error=exc.user_message, exception=exc)
    except ValueError as exc:  # input validation in ChatService
        log.info("Message rejected: %s", exc)
        outcome = TurnOutcome(turn.conversation_id, turn.text, error=str(exc), exception=exc)
    except Exception as exc:
        log.exception("Unexpected error during chat turn")
        outcome = TurnOutcome(turn.conversation_id, turn.text, error=GENERIC_ERROR_MESSAGE, exception=exc)
    finally:
        try:
            bridge.outcome.emit(outcome)
        except RuntimeError:
            log.debug("Turn finished after the UI was destroyed")


# -- generic background calls (settings dialog: list models, test key) -----------


class _CallBridge(QObject):
    done = Signal(object, object)  # value, error message (str | None)


class BackgroundCall(QObject):
    """Run ``fn`` on a daemon thread; ``succeeded``/``failed`` fire on the GUI thread.

    ``failed`` carries a user-presentable message: ``JarvisError.user_message``
    or a generic text (the traceback goes to the log).
    """

    succeeded = Signal(object)
    failed = Signal(str)
    finished = Signal()

    _active: set[BackgroundCall] = set()

    def __init__(self, fn: Callable[[], Any], description: str = "background task") -> None:
        super().__init__()
        self._fn = fn
        self._description = description
        self._bridge = _CallBridge()
        self._bridge.done.connect(self._on_done, Qt.ConnectionType.QueuedConnection)
        self._thread: threading.Thread | None = None

    def start(self) -> BackgroundCall:
        BackgroundCall._active.add(self)  # keep alive until finished
        self._thread = threading.Thread(target=self._run, name="jarvis-bg", daemon=True)
        self._thread.start()
        return self

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        value: Any = None
        error: str | None = None
        try:
            value = self._fn()
        except JarvisError as exc:
            log.info("%s failed: %s: %s", self._description, type(exc).__name__, exc)
            error = exc.user_message
        except Exception as exc:
            log.exception("%s failed unexpectedly", self._description)
            error = f"Unexpected error ({type(exc).__name__}). See the log file for details."
        try:
            self._bridge.done.emit(value, error)
        except RuntimeError:
            pass

    @Slot(object, object)
    def _on_done(self, value: Any, error: str | None) -> None:
        BackgroundCall._active.discard(self)
        if error is None:
            self.succeeded.emit(value)
        else:
            self.failed.emit(error)
        self.finished.emit()
