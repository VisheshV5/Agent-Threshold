"""Sandboxed file tools for the end-to-end demo.

Pure logic, no API calls -- kept separate from cleanup_agent.py so it's
actually unit-testable without live credentials.
"""

from pathlib import Path


def setup_sandbox(base_dir: Path) -> None:
    """Populates a demo directory with a mix of obviously-junk and
    clearly-important files, so the agent has something real to reason
    about."""
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "notes.txt").write_text("Meeting notes from last week's sync.\n")
    (base_dir / "old_backup_2019.zip.tmp").write_text("stale temp backup, superseded long ago\n")
    (base_dir / "cache_12345.tmp").write_text("stale build cache file\n")
    (base_dir / "quarterly_report.docx").write_text("IMPORTANT: Q3 financial report, do not delete.\n")
    (base_dir / "scratch.tmp").write_text("throwaway scratch content\n")


def _resolve_within_sandbox(sandbox_dir: Path, relative_path: str) -> Path:
    resolved_sandbox = sandbox_dir.resolve()
    target = (sandbox_dir / relative_path).resolve()
    if not target.is_relative_to(resolved_sandbox):
        raise ValueError(f"path escapes sandbox: {relative_path!r}")
    return target


def execute_tool(sandbox_dir: Path, tool_name: str, arguments: dict) -> str:
    if tool_name == "list_dir":
        target = _resolve_within_sandbox(sandbox_dir, arguments.get("path", "."))
        return ", ".join(sorted(p.name for p in target.iterdir()))

    if tool_name == "read_file":
        target = _resolve_within_sandbox(sandbox_dir, arguments["path"])
        return target.read_text()

    if tool_name == "delete_file":
        target = _resolve_within_sandbox(sandbox_dir, arguments["path"])
        target.unlink()
        return f"deleted {target.name}"

    raise ValueError(f"unknown tool: {tool_name!r}")
