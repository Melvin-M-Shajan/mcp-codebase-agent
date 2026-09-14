"""§7.1 state schema -- verbatim from the PRD.

`plan` is typed `str | None` per the PRD; plan_node writes it as a JSON-encoded string
(`{"tool": ..., "args": ...}` or `{"done": true}`), which the router in graph.py parses.
That satisfies both the literal type here and §7.2's "structured {'tool': ..., 'args':
...} the router reads" description of plan_node's output.
"""

from typing import TypedDict


class ToolCall(TypedDict):
    tool: str
    args: dict
    result: dict | list


class AgentState(TypedDict):
    question: str
    plan: str | None
    tool_calls: list[ToolCall]
    retrieved_context: list[dict]  # RAG hits from §8, if used
    answer: str | None
    step_count: int
    max_steps: int  # default 6 -- hard cap to prevent infinite loops
