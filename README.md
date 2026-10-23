# J.A.R.V.I.S. – Desktop AI Assistant

A desktop AI assistant for Windows (also runs on macOS/Linux) powered by the
[Groq API](https://console.groq.com). It chats, streams answers, and can act on
your computer through tools. Read-only tasks run automatically; anything with
consequences waits for you to approve it.

- **Desktop GUI** (PySide6/Qt) with conversation sidebar, search, Markdown rendering, streaming, light/dark themes.
- **Groq integration** through the official SDK, with a configurable model (live model list from the API).
- **Agentic tool use** with 12 built-in tools and a strict approval policy.
- **Persistent history** in a local SQLite database; settings in a JSON file.
- **Secure key handling** through the OS credential store (Windows Credential Manager).
- **Robust**: friendly error messages, retries, crash hooks, log rotation, and recovery from corrupt files.

## Requirements

- Python 3.11 or newer (developed and tested on Python 3.14.3, Windows 10)
- A Groq API key – create one at <https://console.groq.com/keys>

## Install and run (from source)

```powershell
git clone https://github.com/Dyckduckgo/J.A.R.V.I.S.git
cd J.A.R.V.I.S
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\jarvis
```

`python -m jarvis` works too. On first start, open **Settings → API key**,
paste your Groq key and press **Save**. The key goes into the Windows
Credential Manager (service `JARVIS-Desktop`), never into a file.

Alternatively set the environment variable `GROQ_API_KEY`; it takes precedence
over the stored key.

To reproduce the exact tested dependency set instead, use the pinned versions
in `requirements.lock`:

```powershell
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install --no-deps -e .
```

## Using it

| Action | How |
|---|---|
| Send | <kbd>Enter</kbd> (<kbd>Shift</kbd>+<kbd>Enter</kbd> inserts a newline) |
| Stop the current answer / deny pending approval | **Stop** button or <kbd>Esc</kbd> |
| New conversation | <kbd>Ctrl</kbd>+<kbd>N</kbd> |
| Settings | <kbd>Ctrl</kbd>+<kbd>,</kbd> |
| Rename / export / delete a conversation | Right-click it in the sidebar (delete asks first) |

### Tools

| Tool | What it does | Runs |
|---|---|---|
| `get_datetime` | Current local date and time | automatically |
| `calculate` | Safe arithmetic and math functions (no `eval`) | automatically |
| `system_info` | OS, CPU, memory, disk (no username/hostname) | automatically |
| `list_directory` | List a folder | automatically inside the workspace, otherwise asks |
| `read_file` | Read a text file | automatically inside the workspace, otherwise asks |
| `search_files` | Find files by glob and optional text | automatically inside the workspace, otherwise asks |
| `save_note` / `search_notes` | Personal notes stored in the app's database (max 10 saved per message) | automatically |
| `write_file` | Create, overwrite or append a file **inside the workspace only** | **asks every time** |
| `run_shell_command` | Run a PowerShell (Windows) or `sh` command with a timeout | **asks every time** |
| `fetch_url` | Download a web page as text (redirects into your local network are blocked unless you approved a local address) | **asks every time** |
| `open_target` | Open a URL or existing file/folder with its default app (never executables) | **asks every time** |

The **workspace** (default `Documents\JARVIS Workspace`) is the assistant's
sandbox; change it in Settings. Individual tools can be disabled in
**Settings → Tools**. The approval dialog shows the exact path, command or URL;
*Deny* is the default.

> **Privacy:** everything you type and every tool result is sent to Groq to
> produce replies. Conversations are stored only on this computer.

## Where data lives

| What | Default location (Windows) |
|---|---|
| Settings | `%APPDATA%\JARVIS\settings.json` |
| Conversations & notes | `%LOCALAPPDATA%\JARVIS\jarvis.db` |
| Logs | `%LOCALAPPDATA%\JARVIS\Logs\jarvis.log` (Help → Open Log Folder) |
| API key | Windows Credential Manager, service `JARVIS-Desktop` |

Set `JARVIS_HOME` to keep everything (except the credential) under one folder.

## Tests

```powershell
.venv\Scripts\python -m pytest            # whole suite; GUI tests run offscreen
.venv\Scripts\python -m pytest -m "not gui"
$env:GROQ_API_KEY="gsk_..."; .venv\Scripts\python -m pytest -m live   # real Groq API (opt-in)
```

## Build a standalone Windows app

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

Produces `dist\JARVIS\JARVIS.exe` (one-folder build, no Python needed) and
runs a headless smoke test of the frozen app.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) – layers, turn lifecycle, threading, error handling
- [docs/SECURITY.md](docs/SECURITY.md) – threat model and mitigations
- [docs/ENGINEERING_DOSSIER.md](docs/ENGINEERING_DOSSIER.md) – implementation record, test results, limitations

## Troubleshooting

| Symptom | Fix |
|---|---|
| "No Groq API key is configured" | Settings → API key, or set `GROQ_API_KEY`. |
| "Groq rejected the API key" | The key is wrong or revoked; create a new one. |
| "rate limit reached" / "too large for … your plan" | Free-tier limits are low on large models: wait, start a new conversation, lower *History messages*, or pick a smaller model such as `llama-3.1-8b-instant`. |
| "model … is not available" | Settings → General → **Refresh** the model list and choose another. |
| Settings or history reset after start-up | The file was corrupt; the old copy was kept next to it (`*.invalid-*` / `*.corrupt-*`). |

## License

MIT – see [LICENSE](LICENSE).
