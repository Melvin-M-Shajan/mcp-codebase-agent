"""Thin async wrapper around the real MCP stdio client -- the agent's tool_node talks
to the MCP server (src/mcp_server/server.py) over the actual MCP protocol, spawned as a
subprocess, not via in-process function calls."""

import os
import sys
from contextlib import asynccontextmanager

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@asynccontextmanager
async def mcp_session():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.server"],
        env=dict(os.environ),
        cwd=os.getcwd(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool(session: ClientSession, name: str, args: dict) -> dict | list:
    result = await session.call_tool(name, args)
    if result.is_error:
        raise RuntimeError(f"MCP tool {name!r} failed: {result.content}")
    structured = result.structured_content
    if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
        return structured["result"]
    return structured
