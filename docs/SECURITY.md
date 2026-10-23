# Security model

J.A.R.V.I.S. gives a cloud-hosted language model the ability to act on the
local machine. The design assumes the model can be wrong or manipulated
(prompt injection through files, web pages or command output) and keeps the
user in control of every action with real consequences.

## Threats and mitigations

| Threat | Mitigation |
|---|---|
| API key disclosure | Key stored in the OS credential store (`keyring`) or supplied via `GROQ_API_KEY`; never written to `settings.json`, the database or logs, and removed from the environment of commands run by `run_shell_command`. All log output passes through `RedactingFormatter` (Groq `gsk_` keys, `sk-…` keys, bearer tokens, `api_key=`/`token=`/`secret=`/`password=` pairs and the exact registered key value). The settings UI shows only a masked key and clears the input after saving. |
| Model performs a harmful action | Every tool has an intrinsic risk level. Consequential tools (write files, run shell commands, fetch URLs, open files/URLs) always require an explicit *Allow once* in a modal dialog that shows the exact path/command/URL and full arguments. *Deny* is the default button; Escape, closing the dialog, pressing Stop or closing the window all deny. |
| Prompt injection via tool output | The system prompt tells the model to treat tool output as data; more importantly, injected instructions still cannot execute anything consequential without the user's approval, and the approval dialog shows the real arguments, not the model's description of them. |
| Data exfiltration to the network | `fetch_url` and `open_target` are consequential, so every outbound request initiated by the model is shown (full URL) and needs approval. Anything the model wants to send must appear in that approved URL. |
| SSRF into the local network | Every `fetch_url` hop (including redirects) is resolved and refused if it points to a loopback, private, link-local, CGNAT or otherwise non-public address, unless the approved URL itself was visibly local (`localhost`, `127.0.0.1`, `192.168.x.x`, …), which the approval text then flags as "an address on this computer or local network". A public URL therefore cannot redirect or DNS-resolve into a router page, local service or cloud metadata endpoint. Redirects between public hosts are followed (max 5) and the final URL is reported. |
| Silent writes by automatic tools | The only automatic tool that writes is `save_note` (the app's own notes table): at most 10 notes per message and 10,000 in total. Every automatic tool call is shown as a card in the chat. |
| Reading sensitive files | File-reading tools run automatically only inside the configured workspace. Paths are resolved (symlinks, junctions, 8.3 short names and `..` collapsed) before the check, so escaping the workspace escalates the call to *requires approval*. UNC/device paths (`\\host\share`, `\\?\…`) are rejected outright – resolving them would already open an SMB connection. |
| Misleading approval text | The dialog shows the tool's description of the call *and* the exact validated JSON arguments that will be passed to the tool. Control characters are rejected in paths; invisible/bidi characters (e.g. U+202E) are shown as `<U+XXXX>`. |
| Writing outside the sandbox | `write_file` refuses any path outside the workspace, even with approval; inside, approval is still required. Overwrites are atomic. |
| Launching programs | `open_target` refuses executables and script types; arbitrary commands are only possible through `run_shell_command`, which always shows the full command and requires approval, runs with a timeout, and is killed (whole process tree) on timeout or Stop. |
| Resource exhaustion by tools | Output truncation, file size limits, directory/search caps, calculator limits on exponent/factorial/input size, HTTP body cap, shell timeout. |
| Destructive app operations | The only destructive operations in the app (deleting a conversation, removing the stored API key) are user-initiated from the GUI and ask for confirmation. The model has no delete tool. Corrupt settings/database files are moved aside, never deleted. |
| Unsafe rendering of model output | Model and tool text is rendered as escaped text / Markdown without raw HTML; links open only on explicit click. |
| Privacy | `system_info` omits username and hostname. Conversations are stored locally only. Everything typed in chat and every tool result is sent to Groq to generate replies – the README states this. |

## Independent review

An independent security review (adversarial-model assumption, proof-of-concept
scripts against the real code) found no approval bypass. It confirmed one
Medium issue – `fetch_url` following redirects into private addresses – which
is fixed and covered by `tests/unit/tools/test_builtin_web_ssrf.py`, plus two
Low issues (unbounded automatic notes; a redaction gap for non-Groq key
shapes), both fixed. Details: `docs/ENGINEERING_DOSSIER.md`.

## Known limitations

* Approval is per call; once a shell command is approved it runs with the
  user's full privileges. Review commands before allowing them.
* The workspace check protects automatic *reads*; the user can still approve a
  read outside it.
* A malicious page fetched with approval can still influence the model's next
  suggestions (prompt injection); it cannot bypass approval.
* SSRF protection resolves each host before httpx connects; a DNS-rebinding
  attacker who changes the answer between those two lookups is not stopped.
* `open_target` blocks executables and scripts by extension; a deny-list cannot
  cover every risky type (disk images, macro-enabled documents). It is a second
  line of defence – every `open_target` call still needs approval and shows
  the exact path.
* Paths are resolved separately when the call is assessed and when it runs; a
  file-system change in between could redirect it. The model cannot cause such
  a change without an already-approved command.
* `GROQ_API_KEY` in the environment is visible to other processes run by the
  same user – prefer the credential store for day-to-day use.
