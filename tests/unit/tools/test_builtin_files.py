"""list_directory, read_file, search_files and write_file behaviour."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest
from builtin_helpers import call, make_junction, make_symlink

import jarvis.tools.builtin.files as files_module
from jarvis.errors import ToolArgumentError, ToolError
from jarvis.tools.base import RiskLevel, ToolContext
from jarvis.tools.builtin.files import (
    MAX_LIST_ENTRIES,
    MAX_READ_BYTES,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)


def _hide_on_windows(path: Path) -> None:
    import ctypes

    if not ctypes.windll.kernel32.SetFileAttributesW(str(path), stat.FILE_ATTRIBUTE_HIDDEN):
        pytest.skip("could not set the hidden attribute")


# --- list_directory -------------------------------------------------------------


def test_list_directory_lists_folders_first_with_sizes(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "zeta").mkdir()
    (workspace / "Alpha").mkdir()
    (workspace / "b.txt").write_bytes(b"x" * 1500)
    (workspace / "a.txt").write_bytes(b"hello")
    (workspace / ".env").write_text("SECRET=1", encoding="utf-8")

    result = call(ListDirectoryTool(), {}, ctx)
    assert result.ok
    lines = result.content.splitlines()
    assert lines[0] == f"Folder: {workspace.resolve()} (2 folders, 2 files)"
    assert lines[1:5] == ["[dir]  Alpha/", "[dir]  zeta/", "[file] a.txt  (5 bytes)", "[file] b.txt  (1.5 KB)"]
    assert ".env" not in result.content
    assert "1 hidden entry not shown" in result.content


def test_list_directory_can_show_hidden_entries(workspace: Path, ctx: ToolContext) -> None:
    (workspace / ".env").write_text("x", encoding="utf-8")
    result = call(ListDirectoryTool(), {"show_hidden": True}, ctx)
    assert "[file] .env  (1 byte)" in result.content
    assert "hidden" not in result.content


@pytest.mark.skipif(os.name != "nt", reason="hidden attribute is Windows-only")
def test_list_directory_respects_windows_hidden_attribute(workspace: Path, ctx: ToolContext) -> None:
    secret = workspace / "desktop.ini"
    secret.write_text("x", encoding="utf-8")
    _hide_on_windows(secret)
    assert "desktop.ini" not in call(ListDirectoryTool(), {}, ctx).content
    assert "desktop.ini" in call(ListDirectoryTool(), {"show_hidden": True}, ctx).content


def test_list_directory_subfolder_and_empty(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "empty").mkdir()
    result = call(ListDirectoryTool(), {"path": "empty"}, ctx)
    assert result.ok
    assert "(0 folders, 0 files)" in result.content and "(empty)" in result.content


def test_list_directory_caps_entries(workspace: Path, ctx: ToolContext) -> None:
    for i in range(MAX_LIST_ENTRIES + 12):
        (workspace / f"f{i:04d}.txt").write_bytes(b"")
    result = call(ListDirectoryTool(), {}, ctx)
    assert result.ok
    assert sum(1 for line in result.content.splitlines() if line.startswith("[file]")) == MAX_LIST_ENTRIES
    assert f"showing the first {MAX_LIST_ENTRIES} of {MAX_LIST_ENTRIES + 12} entries" in result.content


def test_list_directory_errors(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "file.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ToolError, match="Folder not found"):
        call(ListDirectoryTool(), {"path": "missing"}, ctx)
    with pytest.raises(ToolError, match="Not a folder"):
        call(ListDirectoryTool(), {"path": "file.txt"}, ctx)
    with pytest.raises(ToolArgumentError):
        ListDirectoryTool().validate({"show_hidden": "yes"})
    with pytest.raises(ToolArgumentError):
        ListDirectoryTool().validate({"path": "   "})


@pytest.mark.skipif(os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs POSIX non-root")
def test_list_directory_permission_denied_posix(workspace: Path, ctx: ToolContext) -> None:
    locked = workspace / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        with pytest.raises(ToolError, match="Permission denied"):
            call(ListDirectoryTool(), {"path": "locked"}, ctx)
    finally:
        locked.chmod(0o755)


@pytest.mark.skipif(os.name != "nt", reason="Windows system folder")
def test_list_directory_permission_denied_windows(ctx: ToolContext) -> None:
    protected = Path(os.environ.get("SystemDrive", "C:") + "\\") / "System Volume Information"
    try:
        os.listdir(protected)
    except PermissionError:
        pass
    except OSError:
        pytest.skip("protected folder not available")
    else:
        pytest.skip("running with rights to read the protected folder")
    with pytest.raises(ToolError, match="Permission denied"):
        ListDirectoryTool().run({"path": str(protected), "show_hidden": False}, ctx)


def test_list_directory_describe_call(workspace: Path, ctx: ToolContext) -> None:
    tool = ListDirectoryTool()
    args = tool.validate({"path": "sub", "show_hidden": True})
    assert tool.describe_call(args, ctx) == f"List folder {(workspace / 'sub').resolve()} (including hidden entries)"


# --- read_file -------------------------------------------------------------------


def test_read_file_returns_text_and_size(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "notes.txt").write_bytes("héllo wörld\nline 2\n".encode())
    result = call(ReadFileTool(), {"path": "notes.txt"}, ctx)
    assert result.ok
    assert result.content.startswith(f"File: {(workspace / 'notes.txt').resolve()}\n")
    assert "Size: 21 bytes, 19 characters" in result.content
    assert result.content.endswith("\n\nhéllo wörld\nline 2\n")
    assert "truncated" not in result.content


def test_read_file_truncates_and_continues_with_offset(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "long.txt").write_text("".join(str(i % 10) for i in range(250)), encoding="utf-8")
    first = call(ReadFileTool(), {"path": "long.txt", "max_chars": 100}, ctx)
    assert "Showing characters 0-100 of 250" in first.content
    assert "offset=100" in first.content
    assert first.content.endswith("\n\n" + "0123456789" * 10)

    last = call(ReadFileTool(), {"path": "long.txt", "max_chars": 100, "offset": 200}, ctx)
    assert "Showing characters 200-250 of 250" in last.content
    assert "truncated" not in last.content
    assert last.content.endswith("0123456789" * 5)

    with pytest.raises(ToolError, match="beyond the end"):
        call(ReadFileTool(), {"path": "long.txt", "offset": 250}, ctx)


def test_read_file_refuses_binary_files(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + bytes(range(256)))
    with pytest.raises(ToolError, match="binary"):
        call(ReadFileTool(), {"path": "image.png"}, ctx)


def test_read_file_refuses_files_over_10_mb(workspace: Path, ctx: ToolContext) -> None:
    big = workspace / "big.log"
    with open(big, "wb") as fh:
        fh.truncate(MAX_READ_BYTES + 1)
    with pytest.raises(ToolError, match="too large"):
        call(ReadFileTool(), {"path": "big.log"}, ctx)


def test_read_file_decodes_utf16_with_bom(workspace: Path, ctx: ToolContext) -> None:
    # What Windows PowerShell 5.1 writes for `"text" > file.txt`.
    (workspace / "ps.txt").write_bytes("über PowerShell\r\n".encode("utf-16"))
    result = call(ReadFileTool(), {"path": "ps.txt"}, ctx)
    assert result.ok and "über PowerShell" in result.content


def test_read_file_replaces_invalid_utf8(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "latin1.txt").write_bytes("café".encode("latin-1"))
    result = call(ReadFileTool(), {"path": "latin1.txt"}, ctx)
    assert result.ok and result.content.endswith("caf\ufffd")


def test_read_file_empty_file(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "empty.txt").write_bytes(b"")
    result = call(ReadFileTool(), {"path": "empty.txt"}, ctx)
    assert result.ok and result.content.endswith("(empty file)")


def test_read_file_errors(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "folder").mkdir()
    with pytest.raises(ToolError, match="File not found"):
        call(ReadFileTool(), {"path": "nope.txt"}, ctx)
    with pytest.raises(ToolError, match="is a folder"):
        call(ReadFileTool(), {"path": "folder"}, ctx)


@pytest.mark.parametrize(
    "args",
    [{}, {"path": ""}, {"path": "a.txt", "max_chars": 0}, {"path": "a.txt", "max_chars": 100_001},
     {"path": "a.txt", "max_chars": "10"}, {"path": "a.txt", "offset": -1}],
)  # fmt: skip
def test_read_file_argument_validation(args: dict) -> None:
    with pytest.raises(ToolArgumentError):
        ReadFileTool().validate(args)


def test_read_file_defaults_are_applied() -> None:
    assert ReadFileTool().validate({"path": "a.txt"}) == {"path": "a.txt", "max_chars": 20_000, "offset": 0}


def test_read_file_large_max_chars_is_not_cut_by_default_truncation(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "big.txt").write_text("a" * 60_000, encoding="utf-8")
    result = call(ReadFileTool(), {"path": "big.txt", "max_chars": 60_000}, ctx)
    assert result.content.endswith("a" * 60_000)


def test_read_file_describe_call_uses_workspace_from_risk_assessment(workspace: Path, ctx: ToolContext) -> None:
    tool = ReadFileTool()
    args = tool.validate({"path": "docs/readme.md"})
    assert "relative to the workspace folder" in tool.describe_call(args)  # nothing known yet
    assert tool.assess_risk(args, ctx) is RiskLevel.SAFE
    assert str((workspace / "docs" / "readme.md").resolve()) in tool.describe_call(args)


# --- search_files ----------------------------------------------------------------


@pytest.fixture
def tree(workspace: Path) -> Path:
    files = {
        "main.py": "print('hello')\n",
        "README.md": "# Project\nThe Magic Word is here.\n",
        "src/app.py": "import os\n\ndef run():\n    return 'MAGIC word'\n",
        "src/util/helpers.py": "x = 1\n",
        "src/util/data.json": '{"magic word": true}',
        "docs/guide.txt": "nothing to see",
        ".git/hooks/pre-commit.py": "magic word",
        "node_modules/pkg/index.py": "magic word",
        "__pycache__/main.cpython-314.py": "magic word",
    }
    for rel, content in files.items():
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (workspace / "blob.py").write_bytes(b"magic word\x00\x01\x02")
    return workspace


def _paths(content: str) -> list[str]:
    return [line.split("  (")[0].replace("\\", "/") for line in content.splitlines()[1:] if "  (" in line]


def test_search_files_matches_names_recursively(tree: Path, ctx: ToolContext) -> None:
    result = call(SearchFilesTool(), {"pattern": "*.py"}, ctx)
    assert result.ok
    assert sorted(_paths(result.content)) == ["blob.py", "main.py", "src/app.py", "src/util/helpers.py"]
    assert "4 files found" in result.content


def test_search_files_is_case_insensitive_and_supports_path_patterns(tree: Path, ctx: ToolContext) -> None:
    assert _paths(call(SearchFilesTool(), {"pattern": "readme.MD"}, ctx).content) == ["README.md"]
    assert sorted(_paths(call(SearchFilesTool(), {"pattern": "src/util/*"}, ctx).content)) == [
        "src/util/data.json",
        "src/util/helpers.py",
    ]
    assert _paths(call(SearchFilesTool(), {"pattern": "**/app.py"}, ctx).content) == ["src/app.py"]


def test_search_files_contains_filter(tree: Path, ctx: ToolContext) -> None:
    result = call(SearchFilesTool(), {"pattern": "*", "contains": "magic WORD"}, ctx)
    assert result.ok
    assert sorted(_paths(result.content)) == ["README.md", "src/app.py", "src/util/data.json"]
    assert "line 2: The Magic Word is here." in result.content
    assert "line 4: return 'MAGIC word'" in result.content
    assert "blob.py" not in result.content  # binary files are skipped


def test_search_files_skips_large_files_for_content(tree: Path, ctx: ToolContext) -> None:
    (tree / "huge.txt").write_text("magic word\n" + "x" * (2 * 1024 * 1024), encoding="utf-8")
    result = call(SearchFilesTool(), {"pattern": "*.txt", "contains": "magic"}, ctx)
    assert "huge.txt" not in result.content


def test_search_files_in_subdirectory_and_no_matches(tree: Path, ctx: ToolContext) -> None:
    result = call(SearchFilesTool(), {"pattern": "*.py", "directory": "src/util"}, ctx)
    assert _paths(result.content) == ["helpers.py"]
    empty = call(SearchFilesTool(), {"pattern": "*.exe"}, ctx)
    assert empty.ok and "No matching files." in empty.content


def test_search_files_respects_max_results(tree: Path, ctx: ToolContext) -> None:
    result = call(SearchFilesTool(), {"pattern": "*", "max_results": 2}, ctx)
    assert len(_paths(result.content)) == 2
    assert "stopped at max_results=2" in result.content


def test_search_files_caps_scanned_files(tree: Path, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(files_module, "MAX_SCANNED_FILES", 3)
    result = call(SearchFilesTool(), {"pattern": "*"}, ctx)
    assert result.ok
    assert "scanned 3 files" in result.content
    assert "search incomplete: stopped after scanning 3 files" in result.content


def test_search_files_honours_cancellation(tree: Path, ctx: ToolContext) -> None:
    ctx.cancel_event.set()
    result = call(SearchFilesTool(), {"pattern": "*"}, ctx)
    assert not result.ok
    assert "cancelled" in result.content


def test_search_files_does_not_follow_links_out_of_the_workspace(
    tree: Path, outside: Path, ctx: ToolContext
) -> None:
    (outside / "leak.txt").write_text("magic word from outside", encoding="utf-8")
    make_junction(tree / "linked", outside) if os.name == "nt" else make_symlink(tree / "linked", outside)
    result = call(SearchFilesTool(), {"pattern": "*.txt", "contains": "magic"}, ctx)
    assert "leak.txt" not in result.content and "secret.txt" not in result.content


def test_search_files_skips_symlinked_files_pointing_outside(tree: Path, outside: Path, ctx: ToolContext) -> None:
    make_symlink(tree / "alias.txt", outside / "secret.txt")
    result = call(SearchFilesTool(), {"pattern": "*.txt", "contains": "secret"}, ctx)
    assert "alias.txt" not in result.content


def test_search_files_errors(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "f.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ToolError, match="Folder not found"):
        call(SearchFilesTool(), {"pattern": "*", "directory": "missing"}, ctx)
    with pytest.raises(ToolError, match="Not a folder"):
        call(SearchFilesTool(), {"pattern": "*", "directory": "f.txt"}, ctx)


@pytest.mark.parametrize(
    "args",
    [{}, {"pattern": ""}, {"pattern": "**/"}, {"pattern": "*", "max_results": 0}, {"pattern": "*", "max_results": 501},
     {"pattern": "*", "contains": ""}, {"pattern": "*", "directory": "a\nb"}],
)  # fmt: skip
def test_search_files_argument_validation(args: dict) -> None:
    with pytest.raises(ToolArgumentError):
        SearchFilesTool().validate(args)


def test_search_files_describe_call(workspace: Path, ctx: ToolContext) -> None:
    tool = SearchFilesTool()
    args = tool.validate({"pattern": "*.md", "contains": "todo"})
    assert tool.describe_call(args, ctx) == f"Search {workspace.resolve()} for files matching '*.md' containing 'todo'"


# --- write_file ------------------------------------------------------------------


def test_write_file_creates_file_and_parent_folders(workspace: Path, ctx: ToolContext) -> None:
    result = call(WriteFileTool(), {"path": "out/deep/new.txt", "content": "line 1\nline 2\n"}, ctx)
    target = (workspace / "out" / "deep" / "new.txt").resolve()
    assert result.ok
    assert result.content == f"Created {target}: 14 characters (14 bytes)."
    assert target.read_bytes() == b"line 1\nline 2\n"  # exact bytes: no newline translation


def test_write_file_create_refuses_existing_file(workspace: Path, ctx: ToolContext) -> None:
    existing = workspace / "keep.txt"
    existing.write_text("original", encoding="utf-8")
    with pytest.raises(ToolError, match="already exists.*overwrite"):
        call(WriteFileTool(), {"path": "keep.txt", "content": "new"}, ctx)
    assert existing.read_text(encoding="utf-8") == "original"


def test_write_file_overwrite_replaces_content_atomically(workspace: Path, ctx: ToolContext) -> None:
    target = workspace / "data.txt"
    target.write_text("old content that is longer", encoding="utf-8")
    result = call(WriteFileTool(), {"path": "data.txt", "content": "new ✓", "mode": "overwrite"}, ctx)
    assert result.ok and result.content.startswith("Overwrote ")
    assert target.read_text(encoding="utf-8") == "new ✓"
    assert sorted(p.name for p in workspace.iterdir()) == ["data.txt"]  # no temp files left behind


def test_write_file_overwrite_failure_keeps_original(
    workspace: Path, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = workspace / "data.txt"
    target.write_text("precious", encoding="utf-8")

    def failing_replace(src: str, dst: str) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(files_module.os, "replace", failing_replace)
    with pytest.raises(ToolError, match="No space left"):
        call(WriteFileTool(), {"path": "data.txt", "content": "half-written", "mode": "overwrite"}, ctx)
    monkeypatch.undo()
    assert target.read_text(encoding="utf-8") == "precious"
    assert sorted(p.name for p in workspace.iterdir()) == ["data.txt"]


def test_write_file_overwrite_creates_missing_file(workspace: Path, ctx: ToolContext) -> None:
    result = call(WriteFileTool(), {"path": "fresh.txt", "content": "a", "mode": "overwrite"}, ctx)
    assert result.content.startswith("Created ")
    assert (workspace / "fresh.txt").read_text(encoding="utf-8") == "a"


def test_write_file_append(workspace: Path, ctx: ToolContext) -> None:
    call(WriteFileTool(), {"path": "log.txt", "content": "one\n", "mode": "append"}, ctx)
    result = call(WriteFileTool(), {"path": "log.txt", "content": "two\n", "mode": "append"}, ctx)
    assert result.content.startswith("Appended to ")
    assert (workspace / "log.txt").read_text(encoding="utf-8") == "one\ntwo\n"


def test_write_file_refuses_folders_and_oversized_content(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "folder").mkdir()
    with pytest.raises(ToolError, match="is a folder"):
        call(WriteFileTool(), {"path": "folder", "content": "x", "mode": "overwrite"}, ctx)
    with pytest.raises(ToolError, match="too large"):
        call(WriteFileTool(), {"path": "big.txt", "content": "日" * 800_000}, ctx)  # 2.4 MB of UTF-8
    assert not (workspace / "big.txt").exists()


def test_write_file_parent_is_a_file(workspace: Path, ctx: ToolContext) -> None:
    (workspace / "blocker").write_text("x", encoding="utf-8")
    with pytest.raises(ToolError):
        call(WriteFileTool(), {"path": "blocker/child.txt", "content": "x"}, ctx)


@pytest.mark.parametrize(
    "args",
    [{"path": "a.txt"}, {"content": "x"}, {"path": "a.txt", "content": "x", "mode": "replace"},
     {"path": "a.txt", "content": 5}, {"path": "a.txt", "content": "x" * 2_000_001}, {"path": " ", "content": "x"}],
)  # fmt: skip
def test_write_file_argument_validation(args: dict) -> None:
    with pytest.raises(ToolArgumentError):
        WriteFileTool().validate(args)


def test_write_file_defaults_to_create_mode() -> None:
    assert WriteFileTool().validate({"path": "a.txt", "content": ""})["mode"] == "create"


def test_write_file_describe_call_states_path_mode_and_size(workspace: Path, ctx: ToolContext) -> None:
    tool = WriteFileTool()
    target = (workspace / "report.txt").resolve()
    create = tool.validate({"path": "report.txt", "content": "héllo"})
    assert tool.describe_call(create, ctx) == f"Create new file {target} with 5 characters (6 bytes)"

    target.write_text("old", encoding="utf-8")
    overwrite = tool.validate({"path": "report.txt", "content": "abc", "mode": "overwrite"})
    assert tool.describe_call(overwrite, ctx) == f"OVERWRITE existing file {target} with 3 characters (3 bytes)"
    append = tool.validate({"path": "report.txt", "content": "abc", "mode": "append"})
    assert tool.describe_call(append, ctx) == f"Append 3 characters (3 bytes) to existing file {target}"
    assert "already exists" in tool.describe_call(create, ctx)


def test_write_file_describe_call_uses_workspace_seen_by_policy(workspace: Path, ctx: ToolContext) -> None:
    # The agent calls assess_risk (via the policy) before describe_call(args).
    tool = WriteFileTool()
    args = tool.validate({"path": "sub/x.txt", "content": "x"})
    assert tool.assess_risk(args, ctx) is RiskLevel.CONSEQUENTIAL
    assert str((workspace / "sub" / "x.txt").resolve()) in tool.describe_call(args)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_write_file_overwrite_keeps_permissions(workspace: Path, ctx: ToolContext) -> None:
    target = workspace / "script.sh"
    target.write_text("old", encoding="utf-8")
    target.chmod(0o750)
    call(WriteFileTool(), {"path": "script.sh", "content": "new", "mode": "overwrite"}, ctx)
    assert stat.S_IMODE(target.stat().st_mode) == 0o750
