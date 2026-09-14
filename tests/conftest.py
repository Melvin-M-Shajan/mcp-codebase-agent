"""Builds a tiny, real git repo per test (§14: "a small fixture repo ... not the full
target repo") so read_file / run_tests / get_recent_commits exercise real filesystem and
git behavior instead of mocks."""

import subprocess
from pathlib import Path

import pytest

_MATH_UTILS = '''"""Tiny arithmetic helpers, fixture for MCP tool tests."""


def add(a, b):
    return a + b


def subtract(a, b):
    return a - b
'''

_TEST_MATH_UTILS = """from sample_pkg.math_utils import add, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 2) == 3
"""

_README = "# sample-repo\n\nFixture repository used by tests/test_tools.py.\n"


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


@pytest.fixture
def fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "sample_repo"
    (repo / "sample_pkg").mkdir(parents=True)
    (repo / "tests").mkdir()

    (repo / "sample_pkg" / "__init__.py").write_text("", encoding="utf-8")
    (repo / "sample_pkg" / "math_utils.py").write_text(_MATH_UTILS, encoding="utf-8")
    (repo / "tests" / "test_math_utils.py").write_text(_TEST_MATH_UTILS, encoding="utf-8")
    (repo / "README.md").write_text(_README, encoding="utf-8")
    (repo / "pytest.ini").write_text("[pytest]\npythonpath = .\n", encoding="utf-8")

    _run_git(repo, "init", "-q")
    _run_git(repo, "config", "user.email", "fixture@example.com")
    _run_git(repo, "config", "user.name", "Fixture Bot")
    _run_git(repo, "add", "-A")
    _run_git(repo, "commit", "-q", "-m", "Initial commit: math_utils + tests")

    (repo / "README.md").write_text(_README + "\nSecond line.\n", encoding="utf-8")
    _run_git(repo, "add", "-A")
    _run_git(repo, "commit", "-q", "-m", "Expand README")

    return repo
