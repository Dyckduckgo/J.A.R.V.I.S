"""The agent loop: model call -> tool calls -> policy/approval -> results -> repeat.

Invariants maintained for every turn:

* every assistant message that requests tool calls is followed by exactly one
  tool message per call id, even when calls are denied, invalid or cancelled,
  so the stored history is always a valid request for the next turn;
* consequential tool calls never run unless ``callbacks.request_approval``
  returned ``True`` for that specific call;
* every exception raised by a tool is contained and reported to the model as a
  failed result; LLM errors propagate to the caller as :class:`LLMError`.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace

from jarvis.agent.events import AgentCallbacks, ApprovalRequest, TurnResult
from jarvis.agent.prompts import build_system_prompt
from jarvis.config.settings import Settings
from jarvis.errors import JarvisError, LLMToolCallError, OperationCancelled, ToolArgumentError, ToolError
from jarvis.llm.base import ChatClient, ChatMessage, LLMResponse, Role, TokenUsage, ToolCall
from jarvis.tools.base import NotesRepository, ToolContext, ToolResult, truncate
from jarvis.tools.policy import ApprovalPolicy, Decision
from jarvis.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

# Rough request budget. Groq's free tier allows only a few thousand tokens per
# minute on large models, so older history is compacted aggressively.
MAX_CONTEXT_CHARS = 48_000
OLD_TOOL_RESULT_CHARS = 1_500
TOOL_CALL_RETRIES = 1

ERROR_PREFIX = "Error: "
DECLINED_MESSAGE = "The user declined to run this action."
CANCELLED_MESSAGE = "Cancelled by the user before it ran."


class Agent:
    def __init__(
        self,
        client_provider: Callable[[], ChatClient],
        registry: ToolRegistry,
        settings_provider: Callable[[], Settings],
        notes: NotesRepository | None = None,
    ) -> None:
        self._client_provider = client_provider
        self._registry = registry
        self._settings_provider = settings_provider
        self._notes = notes

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    def run_turn(
        self,
        history: Sequence[ChatMessage],
        user_text: str,
        callbacks: AgentCallbacks,
        cancel: threading.Event | None = None,
    ) -> TurnResult:
        """Process one user message. Raises :class:`LLMError` (and subclasses)
        for provider failures and :class:`MissingAPIKeyError` before anything
        is recorded if no key is configured."""
        cancel = cancel or threading.Event()
        settings = self._settings_provider()
        client = self._client_provider()  # fails fast (e.g. missing key) before the user message is recorded

        workspace = settings.workspace_path
        try:
            workspace.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("Workspace %s unavailable: %s", workspace, exc)
        ctx = ToolContext(
            workspace=workspace,
            cancel_event=cancel,
            shell_timeout=settings.shell_timeout_seconds,
            http_timeout=min(30.0, settings.request_timeout_seconds),
            notes=self._notes,
        )
        policy = ApprovalPolicy(settings.disabled_tools)
        tool_schemas = self._registry.schemas(disabled=settings.disabled_tools) or None
        system = ChatMessage.system(build_system_prompt(workspace, settings.custom_instructions))

        user_message = ChatMessage.user(user_text)
        new_messages: list[ChatMessage] = []

        def emit(message: ChatMessage) -> None:
            new_messages.append(message)
            callbacks.on_message(message)

        emit(user_message)
        conversation = prepare_history(history, settings.max_history_messages) + [user_message]
        usage = TokenUsage()

        for iteration in range(1, settings.max_tool_iterations + 1):
            if cancel.is_set():
                return TurnResult("", tuple(new_messages), iteration - 1, cancelled=True, usage=usage)

            callbacks.on_llm_started(iteration)
            streamed: list[str] = []

            def on_text(chunk: str) -> None:
                streamed.append(chunk)
                callbacks.on_text_delta(chunk)

            try:
                response = self._complete(client, [system, *conversation], tool_schemas, settings, on_text, streamed, cancel)
            except OperationCancelled:
                partial = "".join(streamed).strip()
                if partial:
                    emit(ChatMessage.assistant(partial + "\n\n*[stopped]*"))
                return TurnResult(partial, tuple(new_messages), iteration, cancelled=True, usage=usage)

            if response.usage:
                usage = usage + response.usage
            assistant = ChatMessage.assistant(response.content or None, response.tool_calls)

            if not response.tool_calls:
                if not response.content:
                    assistant = replace(assistant, content="")
                emit(assistant)
                log.info("Turn finished after %d model call(s); usage=%s", iteration, usage)
                return TurnResult(response.content, tuple(new_messages), iteration, usage=usage)

            emit(assistant)
            conversation.append(assistant)
            for call in response.tool_calls:
                result = self._run_tool_call(call, ctx, policy, callbacks)
                content = result.content if result.ok else ERROR_PREFIX + result.content
                tool_message = ChatMessage.tool(call, content)
                emit(tool_message)
                conversation.append(tool_message)

            if cancel.is_set():
                return TurnResult("", tuple(new_messages), iteration, cancelled=True, usage=usage)

        notice = (
            f"I stopped after {settings.max_tool_iterations} tool steps without finishing. "
            "Ask me to continue, or raise the step limit in Settings."
        )
        emit(ChatMessage.assistant(notice))
        return TurnResult(
            notice, tuple(new_messages), settings.max_tool_iterations, stopped_at_iteration_limit=True, usage=usage
        )

    # -- helpers -------------------------------------------------------------

    def _complete(
        self,
        client: ChatClient,
        messages: list[ChatMessage],
        tools: list[dict] | None,
        settings: Settings,
        on_text: Callable[[str], None],
        streamed: list[str],
        cancel: threading.Event,
    ) -> LLMResponse:
        attempt = 0
        while True:
            try:
                return client.complete(
                    messages,
                    model=settings.model,
                    tools=tools,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                    stream=settings.stream_responses,
                    on_text=on_text,
                    cancel=cancel,
                )
            except LLMToolCallError:
                # Malformed tool call from the model: retry only if nothing was shown yet.
                if attempt >= TOOL_CALL_RETRIES or streamed:
                    raise
                attempt += 1
                log.warning("Model produced an invalid tool call; retrying (%d/%d)", attempt, TOOL_CALL_RETRIES)

    def _run_tool_call(
        self, call: ToolCall, ctx: ToolContext, policy: ApprovalPolicy, callbacks: AgentCallbacks
    ) -> ToolResult:
        result = self._evaluate_and_run(call, ctx, policy, callbacks)
        result = ToolResult(result.ok, truncate(result.content))
        try:
            callbacks.on_tool_finished(call, result)
        except Exception:
            log.exception("on_tool_finished callback failed")
        return result

    def _evaluate_and_run(
        self, call: ToolCall, ctx: ToolContext, policy: ApprovalPolicy, callbacks: AgentCallbacks
    ) -> ToolResult:
        if ctx.cancelled:
            callbacks.on_tool_started(call, call.name, Decision.DENY)
            return ToolResult.failure(CANCELLED_MESSAGE)

        tool = self._registry.get(call.name)
        if tool is None:
            callbacks.on_tool_started(call, call.name, Decision.DENY)
            available = ", ".join(self._registry.names())
            return ToolResult.failure(f"Unknown tool '{call.name}'. Available tools: {available}.")

        try:
            args = tool.validate(call.parsed_arguments())
        except ToolArgumentError as exc:
            callbacks.on_tool_started(call, f"{call.name}(<invalid arguments>)", Decision.DENY)
            return ToolResult.failure(f"Invalid arguments for {call.name}: {exc.user_message}")

        verdict = policy.evaluate(tool, args, ctx)
        try:
            summary = tool.describe_call(args, ctx)
        except Exception:
            log.exception("describe_call failed for %s", call.name)
            summary = f"{call.name}({call.arguments})"
        callbacks.on_tool_started(call, summary, verdict.decision)

        if verdict.decision is Decision.DENY:
            log.info("Tool call denied by policy: %s (%s)", call.name, verdict.reason)
            return ToolResult.failure(verdict.reason)

        if verdict.decision is Decision.CONFIRM:
            request = ApprovalRequest(call, tool.name, summary, verdict.risk, verdict.reason, args)
            try:
                approved = callbacks.request_approval(request) is True
            except Exception:
                log.exception("Approval callback failed; treating as denied")
                approved = False
            if ctx.cancelled:
                return ToolResult.failure(CANCELLED_MESSAGE)
            log.info("User %s tool call %s: %s", "approved" if approved else "declined", call.name, summary)
            if not approved:
                return ToolResult.failure(DECLINED_MESSAGE)

        try:
            return tool.run(args, ctx)
        except ToolError as exc:
            return ToolResult.failure(exc.user_message)
        except OperationCancelled:
            return ToolResult.failure("Cancelled by the user while running.")
        except JarvisError as exc:
            log.warning("Tool %s failed: %s", call.name, exc)
            return ToolResult.failure(exc.user_message)
        except Exception as exc:  # contain any tool bug
            log.exception("Tool %s crashed", call.name)
            return ToolResult.failure(f"The tool failed unexpectedly ({type(exc).__name__}: {exc}).")


def prepare_history(history: Sequence[ChatMessage], max_messages: int) -> list[ChatMessage]:
    """Select the part of the stored history sent to the model.

    * drops stored system messages (the system prompt is rebuilt each turn);
    * keeps at most ``max_messages`` messages, starting at a user message so a
      tool result is never separated from the assistant call that produced it;
    * compacts tool results from earlier turns and drops whole old turns until
      the request fits :data:`MAX_CONTEXT_CHARS`.
    """
    messages = [m for m in history if m.role is not Role.SYSTEM]
    if len(messages) > max_messages:
        messages = messages[-max_messages:]
    first_user = next((i for i, m in enumerate(messages) if m.role is Role.USER), None)
    messages = messages[first_user:] if first_user is not None else []

    compacted = []
    for m in messages:
        if m.role is Role.TOOL and m.content and len(m.content) > OLD_TOOL_RESULT_CHARS:
            m = replace(m, content=m.content[:OLD_TOOL_RESULT_CHARS] + "\n... [truncated to save context]")
        compacted.append(m)

    def size(items: list[ChatMessage]) -> int:
        return sum(len(m.content or "") + sum(len(c.arguments) for c in m.tool_calls) for m in items)

    while compacted and size(compacted) > MAX_CONTEXT_CHARS:
        next_user = next((i for i, m in enumerate(compacted) if i > 0 and m.role is Role.USER), None)
        if next_user is None:
            compacted = []
            break
        compacted = compacted[next_user:]
    return compacted
