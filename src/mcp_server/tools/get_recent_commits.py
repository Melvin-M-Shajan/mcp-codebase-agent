"""§6.4 get_recent_commits -- `git log` against the local clone via GitPython."""

from pathlib import Path

from git import Repo
from pydantic import BaseModel

from src.mcp_server.repo_root import resolve_safe_path


class Commit(BaseModel):
    hash: str
    author: str
    date: str
    message: str
    files_changed: list[str]


def get_recent_commits(repo_root: Path, path: str | None = None, limit: int = 10) -> list[Commit]:
    limit = min(max(limit, 1), 50)
    repo = Repo(repo_root)

    kwargs: dict = {"max_count": limit}
    if path:
        safe_path = resolve_safe_path(repo_root, path)
        kwargs["paths"] = str(safe_path.relative_to(repo_root))

    commits = list(repo.iter_commits(**kwargs))
    results: list[Commit] = []
    for commit in commits:
        files_changed = list(commit.stats.files.keys())
        results.append(
            Commit(
                hash=commit.hexsha,
                author=commit.author.name or "",
                date=commit.committed_datetime.isoformat(),
                message=commit.message.strip(),
                files_changed=files_changed,
            )
        )
    return results
