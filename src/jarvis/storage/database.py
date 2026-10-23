"""SQLite persistence for conversations, messages and notes.

One connection is shared across threads and serialised with a lock; SQLite in
WAL mode keeps reads cheap. If the database file is corrupt it is moved aside
(never deleted) and a fresh database is created; :attr:`recovery_notice`
explains what happened so the GUI can tell the user.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from jarvis.errors import StorageError
from jarvis.llm.base import ChatMessage, Role, ToolCall
from jarvis.tools.base import Note

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_TITLE = "New conversation"
MAX_NOTES = 10_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    model       TEXT    NOT NULL DEFAULT '',
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role          TEXT    NOT NULL,
    content       TEXT,
    tool_calls    TEXT,
    tool_call_id  TEXT,
    name          TEXT,
    created_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);
CREATE TABLE IF NOT EXISTS notes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT    NOT NULL,
    tags        TEXT    NOT NULL DEFAULT '',
    created_at  REAL    NOT NULL
);
"""


@dataclass(frozen=True)
class Conversation:
    id: int
    title: str
    model: str
    created_at: float
    updated_at: float
    message_count: int = 0


class ConversationStore:
    """Thread-safe repository. Also satisfies :class:`jarvis.tools.base.NotesRepository`."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path) if str(path) != ":memory:" else None
        self._lock = threading.RLock()
        self.recovery_notice: str | None = None
        self._conn = self._open()

    # -- lifecycle -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        target = str(self._path) if self._path else ":memory:"
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target, check_same_thread=False, timeout=10)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            if self._path:
                conn.execute("PRAGMA journal_mode = WAL")
        except sqlite3.Error:
            conn.close()  # release the file handle so a damaged file can be moved aside (Windows)
            raise
        return conn

    def _initialise(self, conn: sqlite3.Connection) -> None:
        result = conn.execute("PRAGMA quick_check").fetchone()
        if not result or result[0] != "ok":
            raise sqlite3.DatabaseError(f"integrity check failed: {result[0] if result else 'no result'}")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise StorageError(
                f"Database schema version {version} is newer than supported ({SCHEMA_VERSION}).",
                user_message="The conversation database was created by a newer version of J.A.R.V.I.S.",
            )
        with conn:
            conn.executescript(_SCHEMA)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _open(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = None
        try:
            conn = self._connect()
            self._initialise(conn)
            return conn
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                conn.close()
            if self._path is None:
                raise StorageError(f"Could not initialise in-memory database: {exc}") from exc
            log.error("Database %s is unusable (%s); moving it aside", self._path, exc)
            backup = self._quarantine()
            self.recovery_notice = (
                "The conversation database was damaged and has been replaced with a new one."
                + (f" The damaged file was kept as {backup.name}." if backup else "")
            )
            try:
                conn = self._connect()
                self._initialise(conn)
                return conn
            except sqlite3.DatabaseError as exc2:
                raise StorageError(
                    f"Could not create database at {self._path}: {exc2}",
                    user_message="The conversation database could not be created.",
                ) from exc2

    def _quarantine(self) -> Path | None:
        assert self._path is not None
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = self._path.with_name(f"{self._path.name}.corrupt-{stamp}")
        try:
            os.replace(self._path, backup)
        except OSError:
            log.exception("Could not move damaged database aside")
            return None
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self._path) + suffix)
            if sidecar.exists():
                try:
                    os.replace(sidecar, Path(str(backup) + suffix))
                except OSError:
                    log.warning("Could not move %s aside", sidecar)
        return backup

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                log.exception("Error closing database")

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                with self._conn:
                    yield self._conn
            except sqlite3.Error as exc:
                log.exception("Database operation failed")
                raise StorageError(f"Database error: {exc}", user_message="A database error occurred.") from exc

    # -- conversations -------------------------------------------------------

    def create_conversation(self, title: str = DEFAULT_TITLE, model: str = "") -> Conversation:
        now = time.time()
        with self._tx() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (title, model, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (title.strip() or DEFAULT_TITLE, model, now, now),
            )
            conversation_id = int(cur.lastrowid)
        return Conversation(conversation_id, title.strip() or DEFAULT_TITLE, model, now, now, 0)

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        with self._tx() as conn:
            row = conn.execute(
                "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
                "FROM conversations c WHERE c.id = ?",
                (conversation_id,),
            ).fetchone()
        return _row_to_conversation(row) if row else None

    def list_conversations(self, search: str = "", limit: int = 500) -> list[Conversation]:
        query = (
            "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
            "FROM conversations c"
        )
        params: list[object] = []
        if search.strip():
            query += (
                " WHERE c.title LIKE ? ESCAPE '\\' OR EXISTS (SELECT 1 FROM messages m "
                "WHERE m.conversation_id = c.id AND m.role IN ('user', 'assistant') AND m.content LIKE ? ESCAPE '\\')"
            )
            pattern = f"%{_escape_like(search.strip())}%"
            params += [pattern, pattern]
        query += " ORDER BY c.updated_at DESC, c.id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._tx() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_conversation(r) for r in rows]

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        title = title.strip()
        if not title:
            raise StorageError("empty title", user_message="The title must not be empty.")
        with self._tx() as conn:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title[:200], time.time(), conversation_id),
            )

    def set_conversation_model(self, conversation_id: int, model: str) -> None:
        with self._tx() as conn:
            conn.execute("UPDATE conversations SET model = ? WHERE id = ?", (model, conversation_id))

    def delete_conversation(self, conversation_id: int) -> None:
        """Permanently delete a conversation. Callers must obtain user confirmation first."""
        with self._tx() as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        log.info("Deleted conversation %s", conversation_id)

    # -- messages ------------------------------------------------------------

    def append_message(self, conversation_id: int, message: ChatMessage) -> int:
        tool_calls = json.dumps([c.to_dict() for c in message.tool_calls]) if message.tool_calls else None
        with self._tx() as conn:
            exists = conn.execute("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if not exists:
                raise StorageError(
                    f"Conversation {conversation_id} does not exist",
                    user_message="The conversation no longer exists.",
                )
            cur = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, tool_calls, tool_call_id, name, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    message.role.value,
                    message.content,
                    tool_calls,
                    message.tool_call_id,
                    message.name,
                    message.created_at,
                ),
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (time.time(), conversation_id))
            return int(cur.lastrowid)

    def delete_message(self, message_id: int) -> bool:
        """Remove one message (used to roll back a turn that failed before any reply)."""
        with self._tx() as conn:
            return conn.execute("DELETE FROM messages WHERE id = ?", (message_id,)).rowcount > 0

    def get_messages(self, conversation_id: int) -> list[ChatMessage]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id", (conversation_id,)
            ).fetchall()
        messages = []
        for row in rows:
            try:
                messages.append(_row_to_message(row))
            except (ValueError, KeyError, TypeError) as exc:
                log.warning("Skipping unreadable message %s: %s", row["id"], exc)
        return messages

    def export_markdown(self, conversation_id: int) -> str:
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise StorageError(f"Conversation {conversation_id} not found", user_message="Conversation not found.")
        lines = [f"# {conversation.title}", ""]
        for msg in self.get_messages(conversation_id):
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(msg.created_at))
            if msg.role is Role.USER:
                lines += [f"## You ({stamp})", "", msg.content or "", ""]
            elif msg.role is Role.ASSISTANT:
                if msg.content:
                    lines += [f"## J.A.R.V.I.S. ({stamp})", "", msg.content, ""]
                for call in msg.tool_calls:
                    lines += [f"> Tool call `{call.name}`: `{call.arguments}`", ""]
            elif msg.role is Role.TOOL:
                body = (msg.content or "").strip()
                if len(body) > 2000:
                    body = body[:2000] + "\n... [truncated]"
                lines += [f"> Tool result `{msg.name}`:", "", "```", body, "```", ""]
        return "\n".join(lines).rstrip() + "\n"

    # -- notes ---------------------------------------------------------------

    def add_note(self, content: str, tags: str = "") -> Note:
        content = content.strip()
        if not content:
            raise StorageError("empty note", user_message="A note must not be empty.")
        now = time.time()
        with self._tx() as conn:
            if conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0] >= MAX_NOTES:
                raise StorageError(
                    "note limit reached",
                    user_message=f"The notes database is full ({MAX_NOTES:,} notes). Delete old notes first.",
                )
            cur = conn.execute(
                "INSERT INTO notes (content, tags, created_at) VALUES (?, ?, ?)", (content, tags.strip(), now)
            )
            return Note(int(cur.lastrowid), content, tags.strip(), now)

    def search_notes(self, query: str = "", limit: int = 20) -> list[Note]:
        sql = "SELECT * FROM notes"
        params: list[object] = []
        if query.strip():
            pattern = f"%{_escape_like(query.strip())}%"
            sql += " WHERE content LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\'"
            params += [pattern, pattern]
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, int(limit)))
        with self._tx() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [Note(int(r["id"]), r["content"], r["tags"], float(r["created_at"])) for r in rows]

    def delete_note(self, note_id: int) -> bool:
        with self._tx() as conn:
            return conn.execute("DELETE FROM notes WHERE id = ?", (note_id,)).rowcount > 0


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _row_to_conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=int(row["id"]),
        title=row["title"],
        model=row["model"] or "",
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        message_count=int(row["message_count"]) if "message_count" in row.keys() else 0,
    )


def _row_to_message(row: sqlite3.Row) -> ChatMessage:
    calls: tuple[ToolCall, ...] = ()
    if row["tool_calls"]:
        calls = tuple(ToolCall.from_dict(item) for item in json.loads(row["tool_calls"]))
    return ChatMessage(
        role=Role(row["role"]),
        content=row["content"],
        tool_calls=calls,
        tool_call_id=row["tool_call_id"],
        name=row["name"],
        created_at=float(row["created_at"]),
    )
