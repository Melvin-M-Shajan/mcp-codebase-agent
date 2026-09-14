"""§14 unit tests for each MCP tool, against the fixture repo -- not the full target
repo. Covers schema compliance, path-traversal rejection in read_file, and truncation
behavior in run_tests."""

import subprocess

import pytest

from src.mcp_server.repo_root import PathTraversalError
from src.mcp_server.tools import search_code as search_code_module
from src.mcp_server.tools.get_recent_commits import get_recent_commits
from src.mcp_server.tools.read_file import read_file
from src.mcp_server.tools.run_tests import OUTPUT_TRUNCATE_CHARS, run_tests


class TestReadFile:
    def test_full_content(self, fixture_repo):
        result = read_file(fixture_repo, "sample_pkg/math_utils.py")
        assert result.path == "sample_pkg/math_utils.py"
        assert "def add" in result.content
        assert result.total_lines == len(result.content.splitlines())

    def test_line_range(self, fixture_repo):
        result = read_file(fixture_repo, "sample_pkg/math_utils.py", start_line=1, end_line=1)
        assert result.content.strip() == '"""Tiny arithmetic helpers, fixture for MCP tool tests."""'
        assert result.total_lines > 1  # total_lines reflects the whole file, not the slice

    def test_rejects_dotdot_traversal(self, fixture_repo):
        with pytest.raises(PathTraversalError):
            read_file(fixture_repo, "../outside.py")

    def test_rejects_absolute_path(self, fixture_repo):
        with pytest.raises(PathTraversalError):
            read_file(fixture_repo, "/etc/passwd")

    def test_rejects_symlink_escape(self, fixture_repo, tmp_path):
        outside = tmp_path / "outside_secret.txt"
        outside.write_text("secret", encoding="utf-8")
        link = fixture_repo / "escape_link"
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks not supported in this environment")
        with pytest.raises(PathTraversalError):
            read_file(fixture_repo, "escape_link")

    def test_missing_file_raises(self, fixture_repo):
        with pytest.raises(FileNotFoundError):
            read_file(fixture_repo, "does_not_exist.py")


class TestRunTests:
    def test_full_suite_passes(self, fixture_repo):
        result = run_tests(fixture_repo)
        assert result.passed == 2
        assert result.failed == 0

    def test_target_subset(self, fixture_repo):
        result = run_tests(fixture_repo, target="tests/test_math_utils.py")
        assert result.passed == 2
        assert result.failed == 0

    def test_rejects_unsafe_target(self, fixture_repo):
        result = run_tests(fixture_repo, target="../../etc")
        assert "rejected" in result.output

    def test_truncates_output(self, fixture_repo, monkeypatch):
        class FakeCompleted:
            stdout = "x" * 10_000
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeCompleted())
        result = run_tests(fixture_repo)
        assert len(result.output) == OUTPUT_TRUNCATE_CHARS


class TestGetRecentCommits:
    def test_returns_commits(self, fixture_repo):
        commits = get_recent_commits(fixture_repo, limit=5)
        assert len(commits) == 2
        assert commits[0].message.startswith("Expand README")
        assert all(c.hash and c.author and c.date for c in commits)

    def test_respects_limit(self, fixture_repo):
        commits = get_recent_commits(fixture_repo, limit=1)
        assert len(commits) == 1

    def test_filters_by_path(self, fixture_repo):
        commits = get_recent_commits(fixture_repo, path="sample_pkg/math_utils.py")
        assert len(commits) == 1
        assert "sample_pkg/math_utils.py" in commits[0].files_changed


class TestSearchCode:
    def test_returns_matches_shaped_per_schema(self, monkeypatch):
        class FakeEmbedder:
            def embed_query(self, text):
                return [0.0] * 1536

        class FakeMatch:
            metadata = {"file_path": "sample_pkg/math_utils.py", "start_line": 3, "end_line": 4, "text": "def add(a, b):\n    return a + b"}
            score = 0.87

        class FakeResponse:
            matches = [FakeMatch()]

        class FakeIndex:
            def query(self, **kwargs):
                assert kwargs["top_k"] == 3
                assert kwargs["namespace"] == "test-ns"
                return FakeResponse()

        monkeypatch.setattr(search_code_module, "get_embeddings_client", lambda task_type: FakeEmbedder())
        monkeypatch.setattr(search_code_module, "get_pinecone_client", lambda: object())
        monkeypatch.setattr(search_code_module, "ensure_index", lambda pc, name: FakeIndex())

        results = search_code_module.search_code(namespace="test-ns", query="add two numbers", top_k=3)

        assert len(results) == 1
        assert results[0].file_path == "sample_pkg/math_utils.py"
        assert results[0].start_line == 3
        assert results[0].score == 0.87

    def test_clamps_top_k(self, monkeypatch):
        class FakeEmbedder:
            def embed_query(self, text):
                return [0.0] * 1536

        class FakeResponse:
            matches = []

        class FakeIndex:
            def query(self, **kwargs):
                assert kwargs["top_k"] == 20
                return FakeResponse()

        monkeypatch.setattr(search_code_module, "get_embeddings_client", lambda task_type: FakeEmbedder())
        monkeypatch.setattr(search_code_module, "get_pinecone_client", lambda: object())
        monkeypatch.setattr(search_code_module, "ensure_index", lambda pc, name: FakeIndex())

        search_code_module.search_code(namespace="test-ns", query="q", top_k=999)
