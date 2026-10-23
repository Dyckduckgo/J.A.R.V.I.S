from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from jarvis.errors import StorageError
from jarvis.llm.base import ChatMessage, Role, ToolCall
from jarvis.storage.database import DEFAULT_TITLE, SCHEMA_VERSION, ConversationStore


@pytest.fixture
def store(tmp_path: Path) -> ConversationStore:
    s = ConversationStore(tmp_path / "data" / "jarvis.db")
    yield s
    s.close()


def test_conversation_lifecycle(store: ConversationStore) -> None:
    conv = store.create_conversation(model="m1")
    assert conv.title == DEFAULT_TITLE
    store.rename_conversation(conv.id, "  Trip planning  ")
    fetched = store.get_conversation(conv.id)
    assert fetched is not None and fetched.title == "Trip planning" and fetched.model == "m1"
    with pytest.raises(StorageError):
        store.rename_conversation(conv.id, "   ")
    store.delete_conversation(conv.id)
    assert store.get_conversation(conv.id) is None


def test_messages_roundtrip_including_tool_calls(store: ConversationStore) -> None:
    conv = store.create_conversation()
    call = ToolCall("call_1", "read_file", '{"path": "a.txt"}')
    sent = [
        ChatMessage.user("read a.txt"),
        ChatMessage.assistant(None, [call]),
        ChatMessage.tool(call, "hello"),
        ChatMessage.assistant("The file says hello."),
    ]
    for m in sent:
        store.append_message(conv.id, m)
    loaded = store.get_messages(conv.id)
    assert [m.role for m in loaded] == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    assert loaded[1].tool_calls == (call,)
    assert loaded[2].tool_call_id == "call_1" and loaded[2].name == "read_file"
    assert loaded == sent  # dataclass equality incl. timestamps
    assert store.get_conversation(conv.id).message_count == 4


def test_append_to_missing_conversation_fails(store: ConversationStore) -> None:
    with pytest.raises(StorageError):
        store.append_message(9999, ChatMessage.user("hi"))


def test_delete_cascades_messages(store: ConversationStore, tmp_path: Path) -> None:
    conv = store.create_conversation()
    store.append_message(conv.id, ChatMessage.user("hi"))
    store.delete_conversation(conv.id)
    raw = sqlite3.connect(tmp_path / "data" / "jarvis.db")
    assert raw.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    raw.close()


def test_list_and_search_orders_by_recent_activity(store: ConversationStore) -> None:
    a = store.create_conversation("Alpha")
    b = store.create_conversation("Beta")
    store.append_message(a.id, ChatMessage.user("discussing 100% of the budget_plan"))
    titles = [c.title for c in store.list_conversations()]
    assert titles == ["Alpha", "Beta"]  # Alpha was updated most recently
    assert [c.title for c in store.list_conversations("beta")] == ["Beta"]
    assert [c.title for c in store.list_conversations("100%")] == ["Alpha"]  # LIKE wildcards escaped
    assert store.list_conversations("budget_plan")[0].id == a.id
    assert store.list_conversations("nothing-matches") == []
    assert b.id != a.id


def test_export_markdown(store: ConversationStore) -> None:
    conv = store.create_conversation("Export me")
    call = ToolCall("c1", "calculate", '{"expression": "2+2"}')
    for m in (ChatMessage.user("2+2?"), ChatMessage.assistant(None, [call]), ChatMessage.tool(call, "4"),
              ChatMessage.assistant("It is **4**.")):
        store.append_message(conv.id, m)
    text = store.export_markdown(conv.id)
    assert text.startswith("# Export me")
    assert "2+2?" in text and "It is **4**." in text and "`calculate`" in text
    with pytest.raises(StorageError):
        store.export_markdown(12345)


def test_notes(store: ConversationStore) -> None:
    n1 = store.add_note("Buy milk", tags="shopping")
    store.add_note("Call Alex about the project", tags="work")
    assert [n.content for n in store.search_notes("milk")] == ["Buy milk"]
    assert [n.content for n in store.search_notes("work")] == ["Call Alex about the project"]
    assert len(store.search_notes()) == 2
    assert store.delete_note(n1.id) is True
    assert store.delete_note(n1.id) is False
    with pytest.raises(StorageError):
        store.add_note("   ")


def test_persistence_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "jarvis.db"
    s1 = ConversationStore(path)
    conv = s1.create_conversation("Persisted")
    s1.append_message(conv.id, ChatMessage.user("remember me"))
    s1.close()
    s2 = ConversationStore(path)
    assert s2.get_messages(conv.id)[0].content == "remember me"
    assert s2.recovery_notice is None
    s2.close()


def test_corrupt_database_is_quarantined_not_deleted(tmp_path: Path) -> None:
    path = tmp_path / "jarvis.db"
    garbage = b"this is definitely not a sqlite database" * 200
    path.write_bytes(garbage)
    store = ConversationStore(path)
    assert store.recovery_notice and "damaged" in store.recovery_notice
    backups = list(tmp_path.glob("jarvis.db.corrupt-*"))
    assert len(backups) == 1 and backups[0].read_bytes() == garbage
    conv = store.create_conversation("fresh")  # new database is usable
    assert store.get_conversation(conv.id) is not None
    store.close()


def test_newer_schema_version_refused(tmp_path: Path) -> None:
    path = tmp_path / "jarvis.db"
    raw = sqlite3.connect(path)
    raw.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    raw.commit()
    raw.close()
    with pytest.raises(StorageError) as info:
        ConversationStore(path)
    assert "newer version" in info.value.user_message


def test_concurrent_writes_are_serialised(store: ConversationStore) -> None:
    conv = store.create_conversation()
    errors: list[BaseException] = []

    def writer(n: int) -> None:
        try:
            for i in range(25):
                store.append_message(conv.id, ChatMessage.user(f"t{n}-{i}"))
        except BaseException as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(store.get_messages(conv.id)) == 100


def test_total_notes_are_capped(store: ConversationStore, monkeypatch: pytest.MonkeyPatch) -> None:
    import jarvis.storage.database as database

    monkeypatch.setattr(database, "MAX_NOTES", 3)
    for i in range(3):
        store.add_note(f"n{i}")
    with pytest.raises(StorageError) as info:
        store.add_note("overflow")
    assert "full" in info.value.user_message
