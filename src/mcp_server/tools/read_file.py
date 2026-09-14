"""§6.2 read_file -- read-only, path-traversal-safe file access within REPO_ROOT."""

from pathlib import Path

from pydantic import BaseModel

from src.mcp_server.repo_root import resolve_safe_path


class ReadFileResult(BaseModel):
    path: str
    content: str
    total_lines: int


def read_file(
    repo_root: Path,
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> ReadFileResult:
    resolved = resolve_safe_path(repo_root, path)
    if not resolved.is_file():
        raise FileNotFoundError(f"not a file: {path}")

    text = resolved.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total_lines = len(lines)

    if start_line is not None or end_line is not None:
        start = max(1, start_line or 1)
        end = min(total_lines, end_line or total_lines)
        content = "\n".join(lines[start - 1 : end])
    else:
        content = text

    return ReadFileResult(path=path, content=content, total_lines=total_lines)
