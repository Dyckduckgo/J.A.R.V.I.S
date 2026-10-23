"""File-system tools: list, read, search and write.

Reading is SAFE inside the workspace and CONSEQUENTIAL (confirmation) outside;
writing is always CONSEQUENTIAL and refused outside the workspace.
"""

from __future__ import annotations

import codecs
import fnmatch
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from jarvis.errors import ToolArgumentError, ToolError
from jarvis.tools.base import RiskLevel, ToolContext, ToolResult, truncate
from jarvis.tools.builtin._display import reveal_hidden
from jarvis.tools.builtin._paths import (
    PathReadTool,
    WorkspaceAwareTool,
    check_path_argument,
    display_path,
    format_size,
    is_inside_workspace,
    is_within,
    resolve_path,
    resolve_workspace,
    try_resolve,
)

MAX_LIST_ENTRIES = 500
MAX_READ_BYTES = 10 * 1024 * 1024
MAX_READ_CHARS = 100_000
BINARY_SNIFF_BYTES = 8192
MAX_SEARCH_FILE_BYTES = 2 * 1024 * 1024
MAX_SCANNED_FILES = 20_000
MAX_SEARCH_SECONDS = 20.0
MAX_SEARCH_RESULTS = 500
MAX_WRITE_CHARS = 2_000_000
MAX_WRITE_BYTES = 2 * 1024 * 1024
SKIPPED_DIRECTORIES = frozenset({".git", ".hg", ".svn", "__pycache__", "node_modules"})


# --- text helpers ------------------------------------------------------------


def looks_binary(head: bytes) -> bool:
    """NUL bytes in the first block mean binary, unless it is BOM-marked UTF-16/32 text."""
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return False
    return b"\x00" in head[:BINARY_SNIFF_BYTES]


def decode_text(data: bytes) -> str:
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return data.decode("utf-32", errors="replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def _is_link_or_junction(entry: os.DirEntry[str]) -> bool:
    try:
        if entry.is_symlink():
            return True
        is_junction = getattr(entry, "is_junction", None)
        return bool(is_junction and is_junction())
    except OSError:
        return True


def _is_hidden(entry: os.DirEntry[str]) -> bool:
    if entry.name.startswith("."):
        return True
    if os.name == "nt":
        try:
            attributes = entry.stat(follow_symlinks=False).st_file_attributes
        except (OSError, AttributeError):
            return False
        return bool(attributes & (stat.FILE_ATTRIBUTE_HIDDEN | stat.FILE_ATTRIBUTE_SYSTEM))
    return False


def _os_error_text(exc: OSError) -> str:
    return exc.strerror or str(exc)


# --- list_directory ------------------------------------------------------------


class ListDirectoryTool(PathReadTool):
    name = "list_directory"
    description = (
        "List the contents of a folder: name, type (dir/file) and size, folders first. Relative paths are "
        "inside the workspace folder. Listing outside the workspace requires the user's approval."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Folder to list; relative to the workspace, absolute, or starting with ~. Default '.'.",
                "default": ".",
                "maxLength": 1024,
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Include hidden entries (dot-files and files marked hidden). Default false.",
                "default": False,
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        target = display_path(args.get("path", "."), self._display_workspace(ctx))
        suffix = " (including hidden entries)" if args.get("show_hidden") else ""
        return f"List folder {target}{suffix}"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = resolve_path(args.get("path", "."), ctx.workspace)
        show_hidden = bool(args.get("show_hidden", False))
        if target.is_file():
            raise ToolError(f"Not a folder: {target} (it is a file; use read_file to read it).")
        try:
            with os.scandir(target) as iterator:
                entries = list(iterator)
        except FileNotFoundError as exc:
            raise ToolError(f"Folder not found: {target}") from exc
        except NotADirectoryError as exc:
            raise ToolError(f"Not a folder: {target}") from exc
        except PermissionError as exc:
            raise ToolError(f"Permission denied: cannot list {target}") from exc
        except OSError as exc:
            raise ToolError(f"Cannot list {target}: {_os_error_text(exc)}") from exc

        hidden_count = 0
        visible: list[tuple[bool, os.DirEntry[str]]] = []
        for entry in entries:
            if not show_hidden and _is_hidden(entry):
                hidden_count += 1
                continue
            try:
                is_dir = entry.is_dir()
            except OSError:
                is_dir = False
            visible.append((is_dir, entry))
        visible.sort(key=lambda item: (not item[0], item[1].name.casefold()))

        dirs = sum(1 for is_dir, _ in visible if is_dir)
        files = len(visible) - dirs
        lines = [f"Folder: {target} ({dirs} folder{'s' if dirs != 1 else ''}, {files} file{'s' if files != 1 else ''})"]
        for is_dir, entry in visible[:MAX_LIST_ENTRIES]:
            if is_dir:
                lines.append(f"[dir]  {entry.name}/")
                continue
            try:
                size = format_size(entry.stat().st_size)
            except OSError:
                size = "size unknown"
            lines.append(f"[file] {entry.name}  ({size})")
        if not visible:
            lines.append("(empty)" if not hidden_count else "(no visible entries)")
        if len(visible) > MAX_LIST_ENTRIES:
            lines.append(f"... showing the first {MAX_LIST_ENTRIES} of {len(visible)} entries.")
        if hidden_count:
            lines.append(f"({hidden_count} hidden entr{'ies' if hidden_count != 1 else 'y'} not shown; use show_hidden=true)")
        return ToolResult.success(truncate("\n".join(lines)))


# --- read_file -----------------------------------------------------------------


class ReadFileTool(PathReadTool):
    name = "read_file"
    description = (
        "Read a text file (UTF-8; UTF-16 with BOM also works). Returns up to max_chars characters starting at "
        "'offset'; for long files call again with the offset given in the output. Binary files and files larger "
        "than 10 MB are refused. Reading outside the workspace requires the user's approval."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File to read; relative to the workspace, absolute, or starting with ~.",
                "minLength": 1,
                "maxLength": 1024,
            },
            "max_chars": {
                "type": "integer",
                "description": f"Maximum number of characters to return (default 20000, max {MAX_READ_CHARS}).",
                "default": 20_000,
                "minimum": 1,
                "maximum": MAX_READ_CHARS,
            },
            "offset": {
                "type": "integer",
                "description": "Character offset to start reading from (default 0).",
                "default": 0,
                "minimum": 0,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        target = display_path(args.get("path", ""), self._display_workspace(ctx))
        offset = args.get("offset") or 0
        start = f" from character {offset:,}" if offset else ""
        return f"Read file {target} (up to {args.get('max_chars', 20_000):,} characters{start})"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        target = resolve_path(args["path"], ctx.workspace)
        max_chars = int(args.get("max_chars", 20_000))
        offset = int(args.get("offset", 0))
        try:
            info = target.stat()
        except FileNotFoundError as exc:
            raise ToolError(f"File not found: {target}") from exc
        except PermissionError as exc:
            raise ToolError(f"Permission denied: {target}") from exc
        except OSError as exc:
            raise ToolError(f"Cannot access {target}: {_os_error_text(exc)}") from exc
        if stat.S_ISDIR(info.st_mode):
            raise ToolError(f"{target} is a folder; use list_directory instead.")
        if not stat.S_ISREG(info.st_mode):
            raise ToolError(f"{target} is not a regular file.")
        if info.st_size > MAX_READ_BYTES:
            raise ToolError(
                f"{target} is too large to read ({format_size(info.st_size)}; the limit is {format_size(MAX_READ_BYTES)})."
            )
        try:
            with open(target, "rb") as fh:
                data = fh.read(MAX_READ_BYTES + 1)
        except PermissionError as exc:
            raise ToolError(f"Permission denied: {target}") from exc
        except OSError as exc:
            raise ToolError(f"Cannot read {target}: {_os_error_text(exc)}") from exc
        if len(data) > MAX_READ_BYTES:
            raise ToolError(f"{target} is too large to read (the limit is {format_size(MAX_READ_BYTES)}).")
        if looks_binary(data[:BINARY_SNIFF_BYTES]):
            raise ToolError(f"{target} appears to be a binary file; only text files can be read.")

        text = decode_text(data)
        total = len(text)
        if total and offset >= total:
            raise ToolError(f"offset {offset:,} is beyond the end of the file ({total:,} characters).")
        chunk = text[offset : offset + max_chars]
        end = offset + len(chunk)
        header = [f"File: {target}", f"Size: {len(data):,} bytes, {total:,} characters"]
        if offset or end < total:
            note = f"Showing characters {offset:,}-{end:,} of {total:,}"
            if end < total:
                note += f" (truncated; call read_file with offset={end} to continue)"
            header.append(note)
        body = chunk if chunk else "(empty file)"
        output = "\n".join(header) + "\n\n" + body
        return ToolResult.success(truncate(output, max_chars + len(output) - len(body) + 1))


# --- search_files --------------------------------------------------------------


def _normalise_pattern(pattern: str) -> tuple[str, bool]:
    """Return (casefolded pattern, match_against_relative_path)."""
    text = pattern.strip().replace("\\", "/")
    while text.startswith("**/"):
        text = text[3:]
    while text.startswith("./"):
        text = text[2:]
    if not text:
        raise ToolArgumentError("The search pattern is empty.")
    return text.casefold(), "/" in text


class _ScanLimits:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self.scanned = 0
        self.deadline = time.monotonic() + MAX_SEARCH_SECONDS
        self.stopped_reason: str | None = None

    def should_stop(self) -> bool:
        if self.ctx.cancel_event.is_set():
            self.stopped_reason = "cancelled"
        elif self.scanned >= MAX_SCANNED_FILES:
            self.stopped_reason = f"stopped after scanning {MAX_SCANNED_FILES:,} files"
        elif time.monotonic() > self.deadline:
            self.stopped_reason = f"stopped after {MAX_SEARCH_SECONDS:g} seconds"
        return self.stopped_reason is not None


def _walk_files(root: Path, limits: _ScanLimits) -> Iterator[os.DirEntry[str]]:
    """Depth-first walk that never follows directory symlinks/junctions."""
    stack = [str(root)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda e: e.name.casefold())
        except OSError:
            continue
        subdirs: list[str] = []
        for entry in entries:
            if limits.should_stop():
                return
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if is_dir:
                if entry.name not in SKIPPED_DIRECTORIES and not _is_link_or_junction(entry):
                    subdirs.append(entry.path)
                continue
            limits.scanned += 1
            yield entry
        stack.extend(reversed(subdirs))


def _find_in_file(path: str, needle: str) -> tuple[int, str] | None:
    """Return (line number, line) of the first case-insensitive match, or None."""
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_SEARCH_FILE_BYTES + 1)
    except OSError:
        return None
    if len(data) > MAX_SEARCH_FILE_BYTES or looks_binary(data[:BINARY_SNIFF_BYTES]):
        return None
    text = decode_text(data)
    folded = text.casefold()
    index = folded.find(needle)
    if index < 0:
        return None
    line_number = folded.count("\n", 0, index) + 1
    lines = text.split("\n")
    line = lines[line_number - 1].strip() if line_number <= len(lines) else ""
    if len(line) > 200:
        line = line[:197] + "..."
    return line_number, line


class SearchFilesTool(PathReadTool):
    name = "search_files"
    description = (
        "Find files recursively by name pattern (glob such as '*.py' or 'report*.docx', case-insensitive), "
        "optionally only those whose text contains a given string (case-insensitive; text files up to 2 MB). "
        "Skips .git, node_modules and __pycache__ folders and does not follow folder links. "
        "Searching outside the workspace requires the user's approval."
    )
    path_argument = "directory"
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob for file names, e.g. '*.txt', 'notes_*.md', '*' for all files.",
                "minLength": 1,
                "maxLength": 260,
            },
            "directory": {
                "type": "string",
                "description": "Folder to search in (recursively). Default '.' (the workspace).",
                "default": ".",
                "maxLength": 1024,
            },
            "contains": {
                "type": "string",
                "description": "Optional text that matching files must contain (case-insensitive).",
                "minLength": 1,
                "maxLength": 500,
            },
            "max_results": {
                "type": "integer",
                "description": f"Maximum number of files to return (default 50, max {MAX_SEARCH_RESULTS}).",
                "default": 50,
                "minimum": 1,
                "maximum": MAX_SEARCH_RESULTS,
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        result = super().validate(args)
        _normalise_pattern(result["pattern"])
        return result

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        target = display_path(args.get("directory", "."), self._display_workspace(ctx))
        text = f"Search {target} for files matching {args.get('pattern', '')!r}"
        if args.get("contains"):
            text += f" containing {args['contains']!r}"
        return reveal_hidden(text)

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = resolve_path(args.get("directory", "."), ctx.workspace, "directory")
        pattern, match_path = _normalise_pattern(args["pattern"])
        contains = args.get("contains")
        needle = contains.casefold() if contains else None
        max_results = int(args.get("max_results", 50))
        if not root.exists():
            raise ToolError(f"Folder not found: {root}")
        if not root.is_dir():
            raise ToolError(f"Not a folder: {root}")
        # A search started inside the workspace must not read through links that lead outside it.
        confine_to = resolve_workspace(ctx.workspace) if is_inside_workspace(root, ctx.workspace) else None

        limits = _ScanLimits(ctx)
        results: list[str] = []
        hit_result_limit = False
        for entry in _walk_files(root, limits):
            relative = os.path.relpath(entry.path, root)
            subject = relative.replace(os.sep, "/") if match_path else entry.name
            if not fnmatch.fnmatchcase(subject.casefold(), pattern):
                continue
            try:
                if not entry.is_file():
                    continue
                if entry.is_symlink() and confine_to is not None:
                    if not is_within(Path(entry.path).resolve(), confine_to):
                        continue
                size = entry.stat().st_size
            except (OSError, RuntimeError):
                continue
            line = f"{relative}  ({format_size(size)})"
            if needle is not None:
                if size > MAX_SEARCH_FILE_BYTES:
                    continue
                found = _find_in_file(entry.path, needle)
                if found is None:
                    continue
                line += f"  line {found[0]}: {found[1]}"
            results.append(line)
            if len(results) >= max_results:
                hit_result_limit = True
                break

        if limits.stopped_reason == "cancelled":
            partial = "\n".join(results)
            return ToolResult.failure(
                truncate(f"Search cancelled by the user after scanning {limits.scanned:,} files." + (f"\nPartial results:\n{partial}" if partial else ""))
            )
        criteria = f"{args['pattern']!r}" + (f" containing {contains!r}" if contains else "")
        header = f"Search in {root} for {criteria}: {len(results)} file{'s' if len(results) != 1 else ''} found (scanned {limits.scanned:,} files)."
        lines = [header, *results]
        if hit_result_limit:
            lines.append(f"... stopped at max_results={max_results}; there may be more matches.")
        if limits.stopped_reason:
            lines.append(f"... search incomplete: {limits.stopped_reason}.")
        if not results:
            lines.append("No matching files.")
        return ToolResult.success(truncate("\n".join(lines)))


# --- write_file ----------------------------------------------------------------


class WriteFileTool(WorkspaceAwareTool):
    name = "write_file"
    description = (
        "Write a UTF-8 text file inside the workspace (always requires the user's approval). "
        "mode='create' (default) makes a new file and fails if it exists; 'overwrite' replaces the file "
        "atomically; 'append' adds to the end. Missing parent folders are created. Paths outside the "
        "workspace are refused."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File to write, relative to the workspace (or an absolute path inside it).",
                "minLength": 1,
                "maxLength": 1024,
            },
            "content": {
                "type": "string",
                "description": "Full text to write (or to append).",
                "maxLength": MAX_WRITE_CHARS,
            },
            "mode": {
                "type": "string",
                "enum": ["create", "overwrite", "append"],
                "description": "'create' (default, fails if the file exists), 'overwrite' or 'append'.",
                "default": "create",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }
    risk = RiskLevel.CONSEQUENTIAL

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        result = super().validate(args)
        result["path"] = check_path_argument(result["path"])
        return result

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        workspace = self._display_workspace(ctx)
        raw = args.get("path", "")
        target = try_resolve(raw, workspace)
        shown = display_path(raw, workspace)
        content = args.get("content") or ""
        if not isinstance(content, str):
            content = str(content)
        size = f"{len(content):,} characters ({len(content.encode('utf-8', errors='replace')):,} bytes)"
        mode = args.get("mode", "create")
        exists = target is not None and target.exists()
        if mode == "overwrite":
            text = (
                f"OVERWRITE existing file {shown} with {size}"
                if exists
                else f"Create file {shown} with {size} (mode 'overwrite'; no file exists there yet)"
            )
        elif mode == "append":
            text = (
                f"Append {size} to existing file {shown}"
                if exists
                else f"Create file {shown} with {size} (mode 'append'; no file exists there yet)"
            )
        else:
            text = f"Create new file {shown} with {size}"
            if exists:
                text += " (a file already exists there, so this will fail)"
        if target is not None and workspace is not None and not is_inside_workspace(target, workspace):
            text += " [outside the workspace: will be refused]"
        return text

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        workspace = resolve_workspace(ctx.workspace)
        target = resolve_path(args["path"], ctx.workspace)
        if not is_within(target, workspace) or os.path.normcase(str(target)) == os.path.normcase(str(workspace)):
            raise ToolError(f"Refusing to write outside the workspace ({workspace}): {target}")
        content: str = args["content"]
        mode: str = args.get("mode", "create")
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ToolError(f"Content is too large ({format_size(len(data))}; the limit is {format_size(MAX_WRITE_BYTES)}).")
        if target.is_dir():
            raise ToolError(f"{target} is a folder, not a file.")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except FileExistsError as exc:
            raise ToolError(f"Cannot create folder {target.parent}: a file with that name exists.") from exc
        except OSError as exc:
            raise ToolError(f"Cannot create folder {target.parent}: {_os_error_text(exc)}") from exc

        try:
            if mode == "create":
                verb = "Created"
                self._create(target, data)
            elif mode == "overwrite":
                verb = "Overwrote" if target.exists() else "Created"
                self._atomic_replace(target, data)
            elif mode == "append":
                verb = "Appended to" if target.exists() else "Created"
                with open(target, "ab") as fh:
                    fh.write(data)
            else:  # validate() prevents this
                raise ToolArgumentError(f"Unknown mode {mode!r}.")
        except FileExistsError as exc:
            raise ToolError(
                f"File already exists: {target}. Use mode='overwrite' to replace it or mode='append' to add to it."
            ) from exc
        except PermissionError as exc:
            raise ToolError(f"Permission denied writing {target}.") from exc
        except OSError as exc:
            raise ToolError(f"Cannot write {target}: {_os_error_text(exc)}") from exc
        return ToolResult.success(f"{verb} {target}: {len(content):,} characters ({len(data):,} bytes).")

    @staticmethod
    def _create(target: Path, data: bytes) -> None:
        with open(target, "xb") as fh:
            try:
                fh.write(data)
            except BaseException:
                fh.close()
                try:
                    target.unlink()
                except OSError:
                    pass
                raise

    @staticmethod
    def _atomic_replace(target: Path, data: bytes) -> None:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            if os.name != "nt" and target.exists():
                try:
                    shutil.copymode(target, tmp_name)
                except OSError:
                    pass
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
