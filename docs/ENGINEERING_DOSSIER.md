# Engineering dossier – J.A.R.V.I.S. 1.0.0

Record of how the application was designed, built, verified and reviewed,
with the results actually observed and the limitations that remain.
Date: 2026-10-23. Platform: Windows 10 Pro 19045, Python 3.14.3.

## 1. Outcome

| Item | Status |
|---|---|
| Desktop app launches (source) | Verified on the real Windows desktop: `python -m jarvis --smoke-test` → exit 0 |
| Desktop app launches (frozen) | `dist\JARVIS\JARVIS.exe --smoke-test` → exit 0 (see §6.3) |
| Automated tests | 631 tests: **625 passed, 6 skipped, 0 failed** (two consecutive full runs) |
| Live Groq API | **Not verified** – no API key was available in this environment (see §8, blocker B1) |
| Independent security review | Done; 1 Medium + 2 Low findings, all fixed with regression tests |
| Independent GUI code review | The review agent was stopped by an API rate limit; replaced by the lead engineer's own review (§7.2) |

## 2. Starting point (Phase 1)

* The project folder was **empty** and **not a Git repository**.
  No files existed to preserve; nothing outside this folder was modified.
* A sibling folder on the development machine holds an earlier, unrelated
  attempt. It was **not read beyond its folder listing and `pyproject.toml`,
  and not modified**; its secrets file was never opened.
* Toolchain found: Python 3.14.3 (global), Tk 8.6, Node, Git. `GROQ_API_KEY` was not set.
* A project virtual environment was created at `.venv`; all work used it.

## 3. Requirements traceability

| # | Requirement | Where it is implemented | Evidence |
|---|---|---|---|
| 1 | Desktop GUI from the beginning | `src/jarvis/ui/` (PySide6) | 54 GUI tests; real-desktop launch; screenshots reviewed (dark & light) |
| 2 | Groq API, configurable models | `llm/groq_client.py`; Settings → model combo + live **Refresh** from `/models` | `tests/unit/test_groq_client.py` (real SDK, mocked transport); live tests ready (skipped) |
| 3 | Modular architecture | Layers in `docs/ARCHITECTURE.md`; no Qt below `ui/` | Core tested without Qt |
| 4 | Agentic tool execution | `agent/agent.py` loop + 12 tools in `tools/builtin/` | `tests/unit/test_agent.py`, `tests/integration/`, 450 tool tests |
| 5 | Auto-run safe tasks, confirm consequential ones | `tools/policy.py`, `ui/approval_dialog.py`, `ui/worker.py` | Policy tests incl. downgrade attempts; GUI allow/deny/stop tests; security review found no bypass |
| 6 | Persistent history & settings | `storage/database.py` (SQLite), `config/settings.py` (JSON) | Round-trip, reopen, corruption-recovery tests |
| 7 | Exception handling, logging, recovery | `errors.py`, `logging_setup.py`, crash hooks in `app.py`, recovery paths | Error-mapping tests, corrupt file tests, redaction tests |
| 8 | Comprehensive automated tests | `tests/` (unit, integration, GUI, live) | §6 |
| 9 | Installation, dependencies, packaging | `pyproject.toml`, `requirements.lock`, `packaging/jarvis.spec`, `scripts/build_windows.ps1` | Frozen build + smoke test (§6.3) |
| 10 | Professional documentation | `README.md`, `docs/ARCHITECTURE.md`, `docs/SECURITY.md`, this file | Cross-checked against code (§9) |

Acceptance criteria:

| Criterion | Result |
|---|---|
| The application launches successfully | Yes – source and frozen builds, real desktop and offscreen |
| Core functionality is real, not mocked | Yes – only the two external boundaries are replaced in tests (remote LLM, OS credential store). Tools really touch files, run PowerShell, serve/fetch HTTP on 127.0.0.1, and write SQLite. |
| API credentials handled securely | OS credential store / env var; never in files or logs; redaction tested; key stripped from shell children |
| Errors handled gracefully | Friendly `user_message` for every expected failure; crash hooks for the rest |
| Tests executed, actual results reported | §6 (copied from the runs) |
| Documentation matches code | §9 |
| No destructive action without authorization | Model has no delete tool; consequential tools always ask; app-level deletes confirm; corrupt files are moved aside, never deleted |

## 4. Design decisions

| Decision | Rationale |
|---|---|
| **PySide6 (Qt 6)** over Tkinter/Flet | Professional widgets, Markdown rendering, robust cross-thread signals, offscreen platform for headless GUI tests (pytest-qt), PyInstaller support. Dependency is `PySide6-Essentials` (no WebEngine). |
| **Official `groq` SDK** over raw HTTP | Maintained wire format, typed errors, built-in back-off for 429/5xx/connection errors; still fully testable through `httpx.MockTransport`. |
| **Approval policy is not configurable** beyond disabling tools | Requirement 5 must hold regardless of settings; risk can only be escalated, never downgraded; unknown/failed assessments fail closed. |
| **Workspace sandbox** for automatic reads and for all writes | Reads outside it need approval; writes outside it are refused even with approval. |
| **`keyring`** for the API key | Uses Windows Credential Manager; env var override for CI. |
| **SQLite (WAL) + immediate persistence** | Every message is stored as soon as it exists, so a crash loses at most the text being streamed. |
| **Daemon worker thread** (not `QThread`) per turn | A stuck network call or tool can never keep the process alive or trigger "QThread destroyed while running". Queued signals keep all UI work on the GUI thread. |
| **History budget** (`prepare_history`) | Groq's free-tier tokens-per-minute limits are low; older tool output is compacted and old turns dropped at user-message boundaries so tool calls never lose their results. |
| Default model `llama-3.3-70b-versatile` | Listed by Groq as a production model with tool use and parallel tool calls (checked against console.groq.com/docs on 2026-10-23). |

## 5. Implementation record (Phases 2–3)

Work was split into independent streams with explicit contracts written first
(`llm/base.py`, `tools/base.py`, `tools/policy.py`, `agent/events.py`,
`agent/service.py`, `app_context.py`):

| Stream | Owner | Output |
|---|---|---|
| Core: paths, errors, logging, settings, secrets, storage, Groq client, agent loop, service, wiring, packaging | Lead engineer | `src/jarvis/*` except `ui/` and `tools/builtin/`; 117 core unit + 7 integration tests |
| Built-in tools | Tools sub-agent (parallel) | `tools/builtin/` (12 tools, path sandbox, display hardening); 450 tests |
| Desktop GUI and entry point | GUI sub-agent (parallel) | `ui/`, `app.py`; 54 GUI tests |
| Security review | Independent reviewer sub-agent (read-only, PoC scripts) | §7.1 |
| GUI/integration review | Reviewer sub-agent → interrupted by rate limit → lead engineer | §7.2 |

Size: 41 source modules, ~7,900 lines of application code; ~5,000 lines of tests.

### Defects found during integration and fixed

| ID | Defect | Root cause | Found by | Fix |
|---|---|---|---|---|
| D1 | Corrupt database could not be quarantined on Windows (`WinError 32`) | A connection that failed at `PRAGMA journal_mode` was never closed, keeping the file locked | `test_corrupt_database_is_quarantined_not_deleted` | `_connect` closes the connection on any setup error |
| D2 | Approval text could show paths relative to the workspace | `Tool.describe_call` received no `ToolContext` | Tools sub-agent (contract review) | `describe_call(args, ctx=None)`; agent passes the turn context; integration test asserts the absolute path |
| D3 | `NaN`/`Infinity` passed numeric range checks | Comparisons with NaN are always false | Tools sub-agent | Non-finite numbers rejected |
| D4 | HTTP connection pool never closed at exit | Factory only dropped its reference | GUI sub-agent | `GroqClientFactory.close()` called from `AppContext.close()` |
| D5 | "Remove key" enabled when only `GROQ_API_KEY` was set | No API to ask whether a key is stored | GUI sub-agent | `SecretStore.has_stored_key()` |
| D6 | Retrying after a failed turn duplicated the user message | The message was persisted before the model failed | GUI sub-agent | `ChatService` rolls back an unanswered user message; GUI re-renders from the store |
| D7 | Approval dialog showed only ~3 lines of the action description | Wrapped line count unknown before layout | Visual check of real-desktop screenshot | Size hint estimates wrapped lines (up to 10 visible) |
| D8 | Calculator summary lost a space after the signature change | Editing slip by the lead engineer | Existing tools test | Fixed |
| D9 | Long streamed answers froze the GUI (576 ms per re-render at 20k chars; quadratic total) | Each Markdown format merge re-laid out the live document | Lead's GUI review, measured | Single edit block (→ 83 ms) + adaptive render throttle |
| D10 | Reloaded tool cards ignored per-call risk escalation | Decision derived from the tool's intrinsic risk only | GUI sub-agent | Re-assessed with `assess_risk` against the current workspace |

## 6. Verification results (Phase 4)

### 6.1 Test suite

Command: `.venv\Scripts\python -m pytest tests -q`

```
625 passed, 6 skipped in 53.84s      (first run)
625 passed, 6 skipped in 56.09s      (second run, unchanged code)
```

| Area | Tests | What they exercise |
|---|---|---|
| `tests/unit` (core) | 117 | settings (validation, atomic save, corrupt-file recovery), secrets (keyring/env, masking), log redaction, storage (round-trip, cascade, search escaping, corruption quarantine, schema guard, 4-thread concurrency, notes cap), Groq client (real SDK; SSE streaming, tool-call reassembly, cancellation, 10 error paths, retries, model filtering), agent loop (auto/confirm/deny, default-deny host, escalation, bad calls, contained exceptions, iteration limit, cancel during approval and streaming, retry of malformed tool calls, history trimming invariants), service/context (persistence, titles, rollback, recovery notices, client cache) |
| `tests/unit/tools` | 450 | every built-in tool against the real file system, PowerShell, a real local HTTP server, SQLite; path-escape tricks incl. real symlinks and NTFS junctions; process-tree kill on timeout; SSRF redirect blocking |
| `tests/integration` | 7 | `ChatService` + `Agent` + policy + built-in tools + SQLite with a scripted model: automatic reads, approval-gated reads/writes/shell, write refusal outside the workspace, notes, disabled tools |
| `tests/ui` | 54 | real widgets offscreen: launch, streaming, auto tool cards, approval allow/deny/stop, error and missing-key paths, conversation management with confirmation, settings validation/persistence/key handling, dialog defaults, `--smoke-test` subprocess |
| `tests/live` | 3 | real Groq API – **skipped: no key** |

Skipped tests: 3 live (no `GROQ_API_KEY`), 3 POSIX-only (permission bits / executable bit; not applicable on Windows).

### 6.2 Manual / exploratory checks performed

* Real-desktop launch from source with a temporary `JARVIS_HOME` → exit 0, clean log.
* Screenshots of the main window (dark and light), approval dialog and settings dialog rendered on the real desktop and inspected: raw HTML in model output shows literally; tool cards show absolute paths and decision/status badges; the no-key banner and status bar render correctly.
* Performance (offscreen, development machine): re-rendering a 20,000-character Markdown answer: 83 ms worst case during streaming (was 576 ms); loading 240 stored messages with 60 tool cards: 1.35 s.

### 6.3 Packaging

* `pyinstaller --noconfirm --clean packaging/jarvis.spec` → exit 0; one-folder build `dist\JARVIS\` (≈134 MB). Built twice: once after integration and again from the final code after all review fixes; both builds passed the smoke test.
* `dist\JARVIS\JARVIS.exe --smoke-test` on the real desktop → exit 0, with both the null keyring backend and the real Windows Credential Manager backend (read-only lookup of the app's own entry; no warnings).
* Only build warning: optional hidden import `tzdata` not found (unused).

## 7. Reviews (Phase 5)

### 7.1 Security review (independent)

Method: adversarial-model assumption; proof-of-concept scripts against the real
code in a scratch directory; read-only on the repository.

| # | Severity | Finding | Resolution |
|---|---|---|---|
| S1 | Medium | `fetch_url` followed redirects (and DNS answers) into loopback/private/link-local/metadata addresses without approval (SSRF) | **Fixed**: every hop is resolved and refused if non-public, unless the approved URL was visibly local (then flagged in the approval text). 26 regression tests reproduce the PoC. |
| S2 | Low | `save_note` runs automatically with no row cap | **Fixed**: 10 notes per message, 10,000 total |
| S3 | Low | Non-Groq secret shapes logged before registration were not redacted | **Fixed**: `sk-…`, `token=`, `secret=`, `password=` patterns |
| S4 | Low | Path resolved separately in `assess_risk` and `run` (TOCTOU) | Accepted – not reachable without an already-approved command; documented |
| S5 | Info | `open_target` extension deny-list is not exhaustive | Accepted – always approval-gated with exact path; documented |
| S6 | Info | Device names / ADS treated as inside the workspace | No action – reads reject non-regular files; no escape |

Verified as working (do not change): non-downgradable, fail-closed policy;
default-deny approval; path sandbox against `..`, symlinks, junctions, 8.3
names, UNC/device prefixes, case; approval text shows exact arguments and
reveals hidden characters; shell hardening (absolute PowerShell path, key
stripped from child env, tree kill); decompressed-size cap on downloads
(gzip bomb capped at 2 MB); parameterised SQL with escaped `LIKE`; AST-only
calculator with resource caps; key never in settings/DB.

### 7.2 GUI and integration review

The reviewer sub-agent stopped early because of an API rate limit. The lead
engineer reviewed the threading bridge, approval handshake, window state
machine, rendering and shutdown instead:

* Approval handshake: first answer wins; Stop, window close and the 15-minute
  timeout deny; a dialog that fails to construct denies; no path auto-approves. **OK.**
* Cross-thread delivery: worker signals reach the GUI only through queued
  connections; a test asserts every public signal fires on the GUI thread. **OK.**
* Busy-state rules: switching/deleting the busy conversation, New chat and
  a second turn are refused during a turn. **OK.**
* Rendering: `MarkdownNoHTML`, `loadResource` disabled, safe link schemes only. **OK.**
* Defects found and fixed: D9 (render performance), D10 (reloaded decisions), D7 (dialog sizing).

## 8. Unresolved limitations and blockers

| ID | Item | Impact / workaround |
|---|---|---|
| B1 | **No Groq API key was available**, so no request reached the real Groq API in this session. The client is verified against the real SDK with recorded-format responses, and `tests/live/` (3 tests: model list, streaming, end-to-end agent tool call) is ready. | Run `$env:GROQ_API_KEY="gsk_..."; .venv\Scripts\python -m pytest -m live` once a key is available. |
| L1 | Tested on Windows 10 only. macOS/Linux code paths exist (POSIX shell, `open`/`xdg-open`, keyring backends) but were not executed. | — |
| L2 | GUI verified by automated offscreen tests, real-desktop smoke launches and screenshots; no extended interactive session by a human tester. | Recommended before wider distribution. |
| L3 | SSRF guard is subject to DNS rebinding between the guard's lookup and the connection. | Approval is still required for every fetch. |
| L4 | Per-conversation model selection is not supported; the current Settings model is used and recorded on the conversation for display. | — |
| L5 | Policy decisions are not stored; reloaded cards re-assess against the *current* workspace. | Cosmetic. |
| L6 | A worker stuck in a call that ignores cancellation is abandoned (daemon thread) 3 s after the window closes. | Process still exits. |
| L7 | Chat view creates one widget per message (no virtualisation); 240 messages load in ~1.4 s. | Start new conversations for very long sessions. |
| L8 | The frozen build is unsigned and ~134 MB; no installer (MSI) is produced. | Code-sign and wrap with an installer for distribution. |
| L9 | No CI pipeline is configured; tests and the build were run locally. | Add a Windows CI job running `pytest` and `scripts\build_windows.ps1`. |

## 9. Documentation cross-check

`README.md` (install, usage, shortcuts, tool table, data locations,
troubleshooting), `docs/ARCHITECTURE.md` (layers, turn lifecycle, policy,
threading, recovery, rendering) and `docs/SECURITY.md` were checked against
the code after the final fixes: shortcuts and menu names against
`ui/main_window.py`; tool names, risks and limits against `tools/builtin/`;
data paths against `AppPaths.default()` output on this machine; error
behaviour against `llm/groq_client.py` and `agent/service.py`; threading
against `ui/worker.py`.

## 10. Deployment

From source (developer machine):

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
.venv\Scripts\jarvis
```

Standalone Windows build:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

Distribute the whole `dist\JARVIS\` folder; users start `JARVIS.exe`, then
enter their Groq key under **Settings → API key**.
