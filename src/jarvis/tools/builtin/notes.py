"""Notes stored in the assistant's own database (SAFE: no external effects)."""

from __future__ import annotations

import threading
import time
import weakref
from typing import Any

from jarvis.errors import JarvisError, ToolArgumentError, ToolError
from jarvis.tools.base import NotesRepository, RiskLevel, Tool, ToolContext, ToolResult, truncate
from jarvis.tools.builtin._display import reveal_hidden

MAX_NOTE_CHARS = 4000
MAX_TAGS_CHARS = 200
MAX_NOTE_RESULTS = 50
MAX_NOTES_PER_TURN = 10


def normalise_tags(raw: str) -> str:
    """'Work, ideas,,work ' -> 'work, ideas' (trimmed, lower-cased, de-duplicated, order kept)."""
    seen: list[str] = []
    for part in raw.split(","):
        tag = " ".join(part.split()).lower()
        if tag and tag not in seen:
            seen.append(tag)
    return ", ".join(seen)


def _require_notes(ctx: ToolContext) -> NotesRepository:
    if ctx.notes is None:
        raise ToolError("Notes are not available right now (no notes storage is configured).")
    return ctx.notes


class SaveNoteTool(Tool):
    name = "save_note"
    description = (
        "Save a short note in the assistant's private notes database so it can be found later with "
        "search_notes (for example something the user asks you to remember)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": f"The note text (at most {MAX_NOTE_CHARS} characters).",
                "minLength": 1,
                "maxLength": MAX_NOTE_CHARS,
            },
            "tags": {
                "type": "string",
                "description": "Optional comma-separated tags, e.g. 'shopping, weekend'.",
                "maxLength": MAX_TAGS_CHARS,
            },
        },
        "required": ["content"],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    def validate(self, args: dict[str, Any]) -> dict[str, Any]:
        result = super().validate(args)
        if not result["content"].strip():
            raise ToolArgumentError("Argument 'content' must not be blank.")
        return result

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        content = str(args.get("content", ""))
        preview = " ".join(content.split())
        if len(preview) > 80:
            preview = preview[:77] + "..."
        tags = normalise_tags(str(args.get("tags") or ""))
        return reveal_hidden(f"Save note ({len(content):,} characters{', tags: ' + tags if tags else ''}): {preview}")

    def __init__(self) -> None:
        # Runs without approval, so bound how much one turn can write (e.g. under
        # prompt injection). Keyed by the turn's cancel event, one per turn.
        self._saved_per_turn: weakref.WeakKeyDictionary[threading.Event, int] = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        notes = _require_notes(ctx)
        content = args["content"].strip()
        if not content:
            raise ToolArgumentError("Argument 'content' must not be blank.")
        tags = normalise_tags(args.get("tags") or "")
        with self._lock:
            saved = self._saved_per_turn.get(ctx.cancel_event, 0)
            if saved >= MAX_NOTES_PER_TURN:
                raise ToolError(
                    f"Not saved: at most {MAX_NOTES_PER_TURN} notes can be saved per message. "
                    "Ask the user before saving more."
                )
            self._saved_per_turn[ctx.cancel_event] = saved + 1
        try:
            note = notes.add_note(content, tags)
        except JarvisError as exc:
            raise ToolError(f"Could not save the note: {exc.user_message}") from exc
        suffix = f" with tags: {note.tags}" if note.tags else ""
        return ToolResult.success(f"Saved note #{note.id}{suffix}.")


class SearchNotesTool(Tool):
    name = "search_notes"
    description = (
        "Search the assistant's saved notes (case-insensitive match on text and tags); newest first. "
        "Omit the query to list the most recent notes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to look for in note contents or tags. Empty or omitted lists the latest notes.",
                "maxLength": 200,
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of notes to return (default 10, max {MAX_NOTE_RESULTS}).",
                "default": 10,
                "minimum": 1,
                "maximum": MAX_NOTE_RESULTS,
            },
        },
        "required": [],
        "additionalProperties": False,
    }
    risk = RiskLevel.SAFE

    def describe_call(self, args: dict[str, Any], ctx: ToolContext | None = None) -> str:
        query = str(args.get("query") or "").strip()
        return f"Search notes for {query!r}" if query else "List recent notes"

    def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        notes = _require_notes(ctx)
        query = (args.get("query") or "").strip()
        limit = int(args.get("limit", 10))
        try:
            found = notes.search_notes(query, limit)
        except JarvisError as exc:
            raise ToolError(f"Could not search notes: {exc.user_message}") from exc
        if not found:
            return ToolResult.success(f"No notes match {query!r}." if query else "There are no saved notes yet.")
        heading = f"{len(found)} note{'s' if len(found) != 1 else ''}" + (f" matching {query!r}" if query else "")
        blocks = [heading + ":"]
        for note in found:
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(note.created_at))
            tags = f" [tags: {note.tags}]" if note.tags else ""
            blocks.append(f"#{note.id} ({stamp}){tags}\n{note.content}")
        return ToolResult.success(truncate("\n\n".join(blocks)))
