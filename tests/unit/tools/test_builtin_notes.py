"""save_note / search_notes round-trip through a real ConversationStore."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from builtin_helpers import call

from jarvis.errors import ToolArgumentError, ToolError
from jarvis.storage.database import ConversationStore
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.notes import SaveNoteTool, SearchNotesTool, normalise_tags


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ConversationStore]:
    db = ConversationStore(tmp_path / "t.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def notes_ctx(ctx: ToolContext, store: ConversationStore) -> ToolContext:
    return replace(ctx, notes=store)


def test_save_and_search_round_trip(notes_ctx: ToolContext, store: ConversationStore) -> None:
    saved = call(SaveNoteTool(), {"content": "  Buy oat milk and coffee  ", "tags": "Shopping, weekend,,shopping "}, notes_ctx)
    assert saved.ok
    assert saved.content == "Saved note #1 with tags: shopping, weekend."
    stored = store.search_notes()
    assert [(n.content, n.tags) for n in stored] == [("Buy oat milk and coffee", "shopping, weekend")]

    found = call(SearchNotesTool(), {"query": "OAT MILK"}, notes_ctx)
    assert found.ok
    assert found.content.startswith("1 note matching 'OAT MILK':")
    assert re.search(r"^#1 \(\d{4}-\d{2}-\d{2} \d{2}:\d{2}\) \[tags: shopping, weekend\]$", found.content, re.MULTILINE)
    assert "Buy oat milk and coffee" in found.content

    by_tag = call(SearchNotesTool(), {"query": "weekend"}, notes_ctx)
    assert "Buy oat milk" in by_tag.content


def test_search_lists_newest_first_and_respects_limit(notes_ctx: ToolContext, store: ConversationStore) -> None:
    for i in range(5):
        store.add_note(f"note number {i}")
        time.sleep(0.002)
    result = call(SearchNotesTool(), {"limit": 3}, notes_ctx)
    assert result.content.startswith("3 notes:")
    order = re.findall(r"note number (\d)", result.content)
    assert order == ["4", "3", "2"]


def test_search_without_matches(notes_ctx: ToolContext) -> None:
    assert call(SearchNotesTool(), {}, notes_ctx).content == "There are no saved notes yet."
    call(SaveNoteTool(), {"content": "alpha"}, notes_ctx)
    assert call(SearchNotesTool(), {"query": "beta"}, notes_ctx).content == "No notes match 'beta'."


def test_note_without_tags(notes_ctx: ToolContext) -> None:
    assert call(SaveNoteTool(), {"content": "plain"}, notes_ctx).content == "Saved note #1."
    assert "[tags:" not in call(SearchNotesTool(), {}, notes_ctx).content


def test_notes_unavailable_without_repository(ctx: ToolContext) -> None:
    assert ctx.notes is None
    with pytest.raises(ToolError, match="not available"):
        call(SaveNoteTool(), {"content": "x"}, ctx)
    with pytest.raises(ToolError, match="not available"):
        call(SearchNotesTool(), {}, ctx)


def test_storage_errors_become_tool_errors(notes_ctx: ToolContext, store: ConversationStore) -> None:
    store.close()  # every further database call fails
    with pytest.raises(ToolError, match="Could not save the note"):
        call(SaveNoteTool(), {"content": "x"}, notes_ctx)
    with pytest.raises(ToolError, match="Could not search notes"):
        call(SearchNotesTool(), {}, notes_ctx)


@pytest.mark.parametrize(
    "args",
    [{}, {"content": ""}, {"content": "   "}, {"content": "x" * 4001}, {"content": "x", "tags": 5},
     {"content": "x", "tags": "t" * 201}],
)  # fmt: skip
def test_save_note_validation(args: dict) -> None:
    with pytest.raises(ToolArgumentError):
        SaveNoteTool().validate(args)


def test_save_note_accepts_maximum_length(notes_ctx: ToolContext, store: ConversationStore) -> None:
    call(SaveNoteTool(), {"content": "y" * 4000}, notes_ctx)
    assert len(store.search_notes()[0].content) == 4000


@pytest.mark.parametrize("args", [{"limit": 0}, {"limit": 51}, {"limit": "5"}, {"query": "q" * 201}])
def test_search_notes_validation(args: dict) -> None:
    with pytest.raises(ToolArgumentError):
        SearchNotesTool().validate(args)


def test_normalise_tags() -> None:
    assert normalise_tags(" Work ,ideas,, work,  big   plans ") == "work, ideas, big plans"
    assert normalise_tags("") == ""


def test_describe_calls() -> None:
    assert SaveNoteTool().describe_call({"content": "remember\nthis", "tags": "a,b"}) == (
        "Save note (13 characters, tags: a, b): remember this"
    )
    assert SearchNotesTool().describe_call({"query": "milk"}) == "Search notes for 'milk'"
    assert SearchNotesTool().describe_call({}) == "List recent notes"


def test_save_note_is_capped_per_turn(tmp_path) -> None:
    import threading

    from jarvis.storage.database import ConversationStore
    from jarvis.tools.base import ToolContext
    from jarvis.tools.builtin.notes import MAX_NOTES_PER_TURN, SaveNoteTool

    store = ConversationStore(tmp_path / "cap.db")
    tool = SaveNoteTool()
    turn = ToolContext(workspace=tmp_path, cancel_event=threading.Event(), notes=store)
    for i in range(MAX_NOTES_PER_TURN):
        assert tool.run({"content": f"note {i}"}, turn).ok
    with pytest.raises(ToolError, match="per message"):
        tool.run({"content": "one too many"}, turn)
    next_turn = ToolContext(workspace=tmp_path, cancel_event=threading.Event(), notes=store)
    assert tool.run({"content": "new turn"}, next_turn).ok
    assert len(store.search_notes(limit=100)) == MAX_NOTES_PER_TURN + 1
    store.close()
