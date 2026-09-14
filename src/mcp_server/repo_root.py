"""Resolves and validates REPO_ROOT: the one directory every read-only tool is allowed
to touch. Centralized here so path-traversal protection lives in exactly one place."""

import os
from pathlib import Path


class InvalidRepoRootError(RuntimeError):
    pass


def get_repo_root() -> Path:
    raw = os.environ.get("REPO_ROOT")
    if not raw:
        raise InvalidRepoRootError("REPO_ROOT environment variable is not set")
    root = Path(raw).resolve()
    if not root.is_dir():
        raise InvalidRepoRootError(f"REPO_ROOT does not exist or is not a directory: {root}")
    if not (root / ".git").exists():
        raise InvalidRepoRootError(f"REPO_ROOT is not a git repository: {root}")
    return root


class PathTraversalError(ValueError):
    pass


def resolve_safe_path(repo_root: Path, relative_path: str) -> Path:
    """Resolves `relative_path` against `repo_root` and guarantees the result stays
    inside `repo_root`. Rejects absolute paths, `..` segments, and symlink escapes."""
    if not relative_path or relative_path.strip() == "":
        raise PathTraversalError("path must not be empty")

    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise PathTraversalError(f"absolute paths are not allowed: {relative_path}")
    if ".." in candidate.parts:
        raise PathTraversalError(f"'..' path segments are not allowed: {relative_path}")

    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError:
        raise PathTraversalError(f"path escapes repo root: {relative_path}") from None
    return resolved
