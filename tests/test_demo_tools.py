"""Tests for the demo's sandboxed file tools, written before the
implementation. No API calls involved -- pure file logic.

Pins the API surface:
- setup_sandbox(base_dir): populates a demo directory with a mix of
  obviously-junk and clearly-important files
- execute_tool(sandbox_dir, tool_name, arguments) -> str
- Path arguments are confined to the sandbox -- a path that resolves
  outside it must be rejected, not silently followed
"""

from pathlib import Path

import pytest

from demo.tools import execute_tool, setup_sandbox


def test_setup_sandbox_creates_a_mix_of_junk_and_important_files(tmp_path):
    setup_sandbox(tmp_path)
    names = {p.name for p in tmp_path.iterdir()}
    assert any(name.endswith(".tmp") for name in names)  # at least one obviously-junk file
    assert len(names) >= 3


def test_list_dir_returns_sorted_file_names(tmp_path):
    setup_sandbox(tmp_path)
    result = execute_tool(tmp_path, "list_dir", {"path": "."})
    names = result.split(", ") if result else []
    assert names == sorted(names)
    assert len(names) >= 3


def test_read_file_returns_contents(tmp_path):
    (tmp_path / "note.txt").write_text("hello world")
    result = execute_tool(tmp_path, "read_file", {"path": "note.txt"})
    assert result == "hello world"


def test_delete_file_actually_removes_it(tmp_path):
    target = tmp_path / "junk.tmp"
    target.write_text("junk")
    execute_tool(tmp_path, "delete_file", {"path": "junk.tmp"})
    assert not target.exists()


def test_path_escaping_the_sandbox_is_rejected(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("should not be reachable")

    with pytest.raises(ValueError, match="sandbox"):
        execute_tool(sandbox, "read_file", {"path": "../outside.txt"})


def test_unknown_tool_name_raises():
    with pytest.raises(ValueError, match="unknown tool"):
        execute_tool(Path("."), "not_a_real_tool", {})
