# Architecture

J.A.R.V.I.S. is a layered PySide6 desktop application. Dependencies point
downwards only: the GUI depends on services, services depend on the domain
layer, and nothing below the GUI imports Qt.

```
┌──────────────────────────────────────────────────────────────────────┐
│ ui/            MainWindow, ChatView, ApprovalDialog, SettingsDialog,  │
│                TurnRunner (worker-thread bridge), themes    [Qt only] │
├──────────────────────────────────────────────────────────────────────┤
│ app.py         entry point, logging, crash hooks, --smoke-test        │
│ app_context.py composition root (AppContext, GroqClientFactory)       │
├──────────────────────────────────────────────────────────────────────┤
│ agent/         ChatService ─▶ Agent loop ─▶ ApprovalPolicy            │
│                events (AgentCallbacks, ApprovalRequest, TurnResult)   │
├───────────────┬───────────────────┬──────────────────┬───────────────┤
│ llm/          │ tools/            │ storage/         │ config/       │
│ ChatClient    │ Tool, registry,   │ ConversationStore│ Settings,     │
│ GroqChatClient│ policy, builtin/* │ (SQLite)         │ SecretStore   │
├───────────────┴───────────────────┴──────────────────┴───────────────┤
│ errors.py · logging_setup.py · paths.py                              │
└──────────────────────────────────────────────────────────────────────┘
```

## Modules

| Module | Responsibility |
|---|---|
| `jarvis/paths.py` | Per-user config/data/log directories (platformdirs); `JARVIS_HOME` relocates everything (tests, portable installs). |
| `jarvis/errors.py` | Exception hierarchy. Every `JarvisError` has a `user_message` that is safe to display. |
| `jarvis/logging_setup.py` | Rotating log file (1 MB × 5) and a `RedactingFormatter` that scrubs API keys from every formatted record, including tracebacks. |
| `jarvis/config/settings.py` | Frozen, validated pydantic `Settings`; `SettingsManager` with atomic writes and recovery from corrupt files. |
| `jarvis/config/secrets.py` | `SecretStore`: `GROQ_API_KEY` env var, else the OS credential store via `keyring`. The key never touches settings or logs. |
| `jarvis/storage/database.py` | `ConversationStore`: SQLite (WAL) for conversations, messages (incl. tool calls) and notes; schema versioning; corrupt-file quarantine. |
| `jarvis/llm/base.py` | Provider-neutral `ChatMessage`, `ToolCall`, `LLMResponse`, and the `ChatClient` protocol. |
| `jarvis/llm/groq_client.py` | `GroqChatClient` on the official `groq` SDK: streaming, tool-call reassembly, cancellation, error mapping, model listing. |
| `jarvis/tools/base.py` | `Tool` ABC, `RiskLevel`, `ToolContext`, `ToolResult`, argument validation. |
| `jarvis/tools/policy.py` | `ApprovalPolicy`: AUTO / CONFIRM / DENY per call. |
| `jarvis/tools/builtin/` | The built-in tools (see README). |
| `jarvis/agent/agent.py` | The agent loop and history preparation. |
| `jarvis/agent/service.py` | `ChatService`: runs a turn for a stored conversation and persists every message as it is produced. |
| `jarvis/app_context.py` | Builds and wires all services into one `AppContext`. |

## A turn, end to end

1. The GUI's worker thread calls `ChatService.send_message(conversation_id, text, callbacks, cancel)`.
2. `ChatService` loads stored history and wraps the callbacks so each new message is written to SQLite before the GUI hears about it.
3. `Agent.run_turn`
   1. obtains a client from the factory (fails fast with `MissingAPIKeyError` before recording anything);
   2. builds the system prompt (workspace, OS, time, custom instructions);
   3. trims history (`prepare_history`): starts at a user message, compacts old tool output, enforces a character budget;
   4. calls the model with the enabled tools' schemas, streaming text through `on_text_delta`;
   5. if the model requested tools, for each call: parse → validate → `ApprovalPolicy.evaluate` → (CONFIRM: block on `request_approval`) → run → record a tool message;
   6. repeats until the model answers without tool calls, the user cancels, or `max_tool_iterations` is reached.
4. Every assistant tool call is always answered by exactly one tool message (denied, invalid and cancelled calls included), so the stored history is always a valid request for the next turn.

## Approval policy

| Situation | Decision |
|---|---|
| Unknown tool, or tool disabled in Settings | DENY (the model is told why) |
| Tool's intrinsic risk is `CONSEQUENTIAL` | CONFIRM – always |
| `SAFE` tool whose `assess_risk()` escalates for this call (e.g. a path outside the workspace) | CONFIRM |
| `assess_risk()` raises | CONFIRM (fail closed) |
| Otherwise | AUTO |

Risk can only be escalated, never downgraded, and there is no setting that
auto-approves consequential tools. If the host does not implement
`request_approval`, the default implementation denies.

## Threading model

* The GUI thread never blocks on the network or tools.
* A turn runs on a daemon `threading.Thread` owned by `ui/worker.py:TurnRunner`
  (a daemon thread cannot keep the process alive if a call ignores
  cancellation after the window closed). Agent callbacks run on that thread and
  are emitted through a private `QObject` bridge with explicit
  `QueuedConnection`s, so every public `TurnRunner` signal fires on the GUI thread.
* Approval requests block the worker on a `PendingApproval` (`threading.Event`)
  until the dialog answers. The first answer wins; Stop, window close, and a
  15-minute timeout all resolve the request as *deny*. Nothing auto-approves.
* Only one turn runs at a time; while it runs, switching or deleting the busy
  conversation is disabled.
* Settings' "Refresh models" and "Test connection" use `BackgroundCall`
  (daemon thread + queued signal) so the dialog stays responsive.
* Cancellation is cooperative through a `threading.Event` checked between
  stream chunks, before each tool call, and inside long-running tools (shell
  commands poll it and kill their process tree).
* `ConversationStore` serialises access to a single SQLite connection with a lock.

## Rendering

Chat bubbles are `QTextBrowser`s fed through `QTextDocument.setMarkdown` with
`MarkdownNoHTML`: raw HTML in model or tool output is shown as text,
`loadResource` is overridden so no image or remote resource is ever fetched,
and links open only on click and only for `http`, `https` and `mailto`.
Streaming text is re-rendered on a throttle (80 ms, growing adaptively to keep
render work under ~25% of the GUI thread on long answers); all format changes
happen inside one edit block, so a 20,000-character answer re-renders in about
80 ms instead of about 600 ms (measured on the development machine).

## Error handling and recovery

| Failure | Behaviour |
|---|---|
| Invalid/corrupt `settings.json` | Moved to `settings.json.invalid-<timestamp>`, defaults used, user notified. |
| Corrupt database | Moved to `jarvis.db.corrupt-<timestamp>` (never deleted), new database created, user notified. |
| Database from a newer version | Refused with a clear message (no data is touched). |
| Missing API key | Turn refused before anything is stored; GUI points to Settings. |
| Turn fails before any reply (e.g. rate limit) | `ChatService` removes the just-stored user message, the GUI re-renders from the store and puts the text back in the input box, so a retry does not duplicate it. If the assistant or a tool already produced output, everything is kept (tools may have had effects). |
| 401/403, 404 model, 413, 429, 5xx, timeouts, connection errors, stream errors | Mapped to `LLMError` subclasses with friendly messages. 429/5xx/connection errors are retried by the SDK with back-off. |
| Model emits malformed tool call (`tool_use_failed`) | Retried once if nothing was streamed yet, then reported. |
| Tool raises | Contained; reported to the model as an error result. |
| Unhandled exception anywhere | Logged with traceback by `sys.excepthook` / `threading.excepthook`; GUI shows a non-fatal error dialog. |
