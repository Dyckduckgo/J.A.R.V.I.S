"""Chat service: binds the agent loop to persistent conversations."""

from __future__ import annotations

import logging
import threading

from jarvis.agent.agent import Agent
from jarvis.agent.events import AgentCallbacks, ApprovalRequest, TurnResult
from jarvis.errors import StorageError
from jarvis.llm.base import ChatMessage, Role, ToolCall
from jarvis.storage.database import DEFAULT_TITLE, ConversationStore
from jarvis.tools.base import ToolResult
from jarvis.tools.policy import Decision

log = logging.getLogger(__name__)

MAX_INPUT_CHARS = 32_000
TITLE_LENGTH = 60


def title_from_text(text: str) -> str:
    line = " ".join(text.split())
    if len(line) <= TITLE_LENGTH:
        return line or DEFAULT_TITLE
    return line[: TITLE_LENGTH - 1].rstrip() + "…"


class _PersistingCallbacks(AgentCallbacks):
    """Writes every new message to the store before forwarding the event."""

    def __init__(self, store: ConversationStore, conversation_id: int, inner: AgentCallbacks, retitle: bool) -> None:
        self._store = store
        self._conversation_id = conversation_id
        self._inner = inner
        self._retitle = retitle
        self.persisted: list[tuple[int, Role]] = []
        self.retitled = False

    def on_message(self, message: ChatMessage) -> None:
        message_id = self._store.append_message(self._conversation_id, message)
        self.persisted.append((message_id, message.role))
        if self._retitle and message.role is Role.USER:
            self._retitle = False
            self._store.rename_conversation(self._conversation_id, title_from_text(message.content or ""))
            self.retitled = True
        self._inner.on_message(message)

    def rollback_unanswered_user_message(self) -> bool:
        """If the turn produced nothing but the user's message, remove it so a
        retry does not leave two copies in the history."""
        if len(self.persisted) != 1 or self.persisted[0][1] is not Role.USER:
            return False
        try:
            self._store.delete_message(self.persisted[0][0])
            if self.retitled:
                self._store.rename_conversation(self._conversation_id, DEFAULT_TITLE)
        except StorageError:
            log.exception("Could not roll back the unanswered user message")
            return False
        return True

    def on_llm_started(self, iteration: int) -> None:
        self._inner.on_llm_started(iteration)

    def on_text_delta(self, text: str) -> None:
        self._inner.on_text_delta(text)

    def on_tool_started(self, call: ToolCall, summary: str, decision: Decision) -> None:
        self._inner.on_tool_started(call, summary, decision)

    def on_tool_finished(self, call: ToolCall, result: ToolResult) -> None:
        self._inner.on_tool_finished(call, result)

    def request_approval(self, request: ApprovalRequest) -> bool:
        return self._inner.request_approval(request)


class ChatService:
    def __init__(self, store: ConversationStore, agent: Agent) -> None:
        self._store = store
        self._agent = agent

    @property
    def store(self) -> ConversationStore:
        return self._store

    @property
    def agent(self) -> Agent:
        return self._agent

    def send_message(
        self,
        conversation_id: int,
        text: str,
        callbacks: AgentCallbacks,
        cancel: threading.Event | None = None,
        *,
        model: str | None = None,
    ) -> TurnResult:
        """Run one agent turn for ``conversation_id``; every message produced is
        persisted as soon as it exists. Blocking: call from a worker thread.

        ``model`` is recorded on the conversation for display; the agent itself
        always uses the model from the current settings.

        If the turn fails before anything but the user's message was stored,
        that message is removed again, so the caller can offer the text for a
        retry without duplicating it. Once the assistant or a tool has produced
        output, everything is kept (tools may already have had effects).

        Raises ``ValueError`` for empty/oversized input, :class:`StorageError`
        if the conversation is missing, and :class:`LLMError` from the model.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("Message is empty.")
        if len(text) > MAX_INPUT_CHARS:
            raise ValueError(f"Message is too long ({len(text)} characters; limit {MAX_INPUT_CHARS}).")
        conversation = self._store.get_conversation(conversation_id)
        if conversation is None:
            raise StorageError(
                f"Conversation {conversation_id} not found", user_message="The conversation no longer exists."
            )
        history = self._store.get_messages(conversation_id)
        retitle = conversation.title == DEFAULT_TITLE and not any(m.role is Role.USER for m in history)
        persisting = _PersistingCallbacks(self._store, conversation_id, callbacks, retitle)
        if model:
            self._store.set_conversation_model(conversation_id, model)
        log.info("Turn started in conversation %s (%d stored messages)", conversation_id, len(history))
        try:
            return self._agent.run_turn(history, text, persisting, cancel)
        except Exception:
            if persisting.rollback_unanswered_user_message():
                log.info("Turn failed before any reply; removed the unanswered user message")
            raise
