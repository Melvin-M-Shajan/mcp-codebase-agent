"""§6.5 MCP server entry point -- registers all 4 tools, reads REPO_ROOT from the
environment at startup, refuses to start if REPO_ROOT is not a valid git repo.

Read-only by construction: no write/delete tool is registered here, ever (§4 scope).
"""

import os
from typing import Annotated

from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from pydantic import Field

from src.mcp_server.repo_root import get_repo_root
from src.mcp_server.tools.get_recent_commits import Commit
from src.mcp_server.tools.get_recent_commits import get_recent_commits as _get_recent_commits
from src.mcp_server.tools.read_file import ReadFileResult
from src.mcp_server.tools.read_file import read_file as _read_file
from src.mcp_server.tools.run_tests import RunTestsResult
from src.mcp_server.tools.run_tests import run_tests as _run_tests
from src.mcp_server.tools.search_code import CodeMatch
from src.mcp_server.tools.search_code import search_code as _search_code

load_dotenv()

REPO_ROOT = get_repo_root()
NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "default")

mcp = MCPServer("mcp-codebase-agent")


@mcp.tool(
    description=(
        "Semantic + keyword search across the indexed repository's source files. "
        "Returns the most relevant code chunks for a natural-language or keyword query."
    ),
)
def search_code(
    query: str,
    top_k: Annotated[int, Field(default=8, ge=1, le=20)] = 8,
) -> list[CodeMatch]:
    return _search_code(namespace=NAMESPACE, query=query, top_k=top_k)


@mcp.tool(
    description="Read the contents of a specific file in the repository, optionally restricted to a line range.",
)
def read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> ReadFileResult:
    return _read_file(REPO_ROOT, path, start_line, end_line)


@mcp.tool(
    description="Run the repository's test suite, or a subset matching a target path/pattern, and return pass/fail results.",
)
def run_tests(target: str | None = None) -> RunTestsResult:
    return _run_tests(REPO_ROOT, target)


@mcp.tool(
    description="Get recent commit history for the whole repo or a specific file/path.",
)
def get_recent_commits(
    path: str | None = None,
    limit: Annotated[int, Field(default=10, le=50)] = 10,
) -> list[Commit]:
    return _get_recent_commits(REPO_ROOT, path, limit)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
