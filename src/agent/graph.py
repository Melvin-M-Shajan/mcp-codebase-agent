"""§7.2/§7.3 graph nodes and routing.

Nodes are built by `build_graph(session, planner_llms, answer_llms)` as closures over a
live MCP `ClientSession` and lists of chat model clients (see src/agent/llm.py for why
two roles and why lists), but each still matches the PRD's `(state: AgentState) ->
AgentState` node signature -- the closure is just how session/llm get in without
changing that signature or resorting to module-level globals.
"""

import asyncio
import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.agent.mcp_client import call_tool
from src.agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    TOOL_CALL_PROTOCOL_NOTE,
    TOOL_DESCRIPTIONS,
)
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

# plan_node hands these to the LLM as real, callable tools (via bind_tools) rather than
# asking it to fill out a separate custom "decision" schema. In practice, models with
# strong agentic post-training (this project uses openai/gpt-oss-120b via Groq) try to
# emit a genuine tool call the moment a prompt describes tools as available regardless
# of which structured-output method is requested -- binding the real tools plus one
# pseudo-tool for "I'm done" works with that instinct instead of fighting it, and was
# verified against the live model before adopting it (see README "Deviations from PRD").
FINISH_TOOL_NAME = "finish_investigation"

_PLAN_TOOL_SPECS = [
    {
        "name": "search_code",
        "description": "Semantic + keyword search across the indexed repository's source files.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language or keyword search query."},
                "top_k": {"type": ["integer", "null"], "default": 8, "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a specific file in the repository, optionally restricted to a line range.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Repo-relative file path."},
                "start_line": {"type": ["integer", "null"]},
                "end_line": {"type": ["integer", "null"]},
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_tests",
        "description": "Run the repository's test suite, or a subset matching a target path/pattern.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": ["string", "null"],
                    "description": "Optional test file/path/pattern to restrict the run.",
                },
            },
        },
    },
    {
        "name": "get_recent_commits",
        "description": "Get recent commit history for the whole repo or a specific file/path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": ["string", "null"]},
                "limit": {"type": "integer", "default": 10, "maximum": 50},
            },
        },
    },
    {
        "name": FINISH_TOOL_NAME,
        "description": (
            "Call this once you have enough information to answer the question well -- "
            "do not call any other tool in the same turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {"reasoning": {"type": "string", "description": "Why the gathered context is enough."}},
            "required": ["reasoning"],
        },
    },
]


_RETRY_DELAY_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s")


def _parse_groq_retry_seconds(text: str, default: float = 90.0) -> float:
    match = _RETRY_DELAY_RE.search(text)
    if not match:
        return default
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))
    return minutes * 60 + seconds + 5  # small safety margin


async def _invoke_with_retry(
    runnables: list, messages, max_retries: int = 4, base_delay: float = 8.0, max_daily_cap_waits: int = 6
):
    """`runnables` is one client per available API key (see src/agent/llm.py), all bound
    identically. Groq's free tier applies per-model tokens-per-minute limits; a
    multi-question run (agent loop + RAGAS judge calls) legitimately bursts past those
    sometimes -- retry with backoff on the same client instead of failing the whole run
    on a transient 429. A daily (TPD) cap is a different story -- retrying the same key
    immediately just burns more of a budget that won't refill for a long time, so that
    falls through to the next key first; once every key is on a daily cap, sleep for
    Groq's own reported reset time (it's a rolling window, so this is usually minutes,
    not a full day) and cycle through the keys again, up to `max_daily_cap_waits` times."""
    last_exc: Exception | None = None
    for _wait_round in range(max_daily_cap_waits):
        all_daily_capped = True
        for runnable in runnables:
            delay = base_delay
            for attempt in range(max_retries):
                try:
                    return await runnable.ainvoke(messages)
                except Exception as exc:
                    last_exc = exc
                    text = str(exc).lower()
                    if "tpd" in text or "tokens per day" in text:
                        logger.warning("daily token cap hit on this key, falling through to next key: %s", exc)
                        break
                    all_daily_capped = False
                    is_rate_limit = "rate_limit" in text or "429" in text
                    if not is_rate_limit or attempt == max_retries - 1:
                        raise
                    logger.warning(
                        "rate limited, retrying in %.0fs (%d/%d): %s", delay, attempt + 1, max_retries, exc
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
        if not all_daily_capped:
            raise last_exc
        wait_s = _parse_groq_retry_seconds(str(last_exc))
        logger.warning("every key is daily-capped, sleeping %.0fs before retrying all of them again", wait_s)
        await asyncio.sleep(wait_s)
    raise last_exc


SNIPPET_PREVIEW_CHARS = 400
RESULT_PREVIEW_CHARS = 500


def _summarize_context(state: AgentState) -> str:
    # Deliberately avoids a `tool(args) -> result` shape: openai/gpt-oss-120b (this
    # project's LLM, see README "Deviations from PRD") reflexively echoes a fake
    # tool-call JSON payload as its own output when the prompt contains text shaped like
    # a function call, even with no tools bound and no such call requested.
    #
    # Snippets/results are truncated for the prompt (not for RAGAS's context, which uses
    # the full untruncated data -- see run_eval.py): this context gets resent on every
    # plan_node/answer_node call in the loop (up to max_steps+1 times), and Groq's free
    # tier has a real daily token budget per model, so trimming it matters.
    parts: list[str] = []
    if state["retrieved_context"]:
        parts.append("Retrieved code (RAG search over the indexed repo):")
        for hit in state["retrieved_context"]:
            snippet = hit["snippet"][:SNIPPET_PREVIEW_CHARS]
            parts.append(f"- {hit['file_path']}:{hit['start_line']}-{hit['end_line']} (score={hit['score']:.3f})\n{snippet}")
    if state["tool_calls"]:
        parts.append("\nEvidence gathered from tools used so far:")
        for i, call in enumerate(state["tool_calls"], start=1):
            result_preview = json.dumps(call["result"], indent=2, default=str)[:RESULT_PREVIEW_CHARS]
            parts.append(
                f"- Evidence {i}, from the {call['tool']} tool, called with {call['args']}:\n{result_preview}"
            )
    return "\n".join(parts) if parts else "(nothing gathered yet)"


def build_graph(session, planner_llms, answer_llms):
    """`planner_llms`/`answer_llms` are lists (one client per available API key) -- see
    src/agent/llm.py's get_planner_llms()/get_answer_llms()."""
    planners = [llm.bind_tools(_PLAN_TOOL_SPECS, tool_choice="auto") for llm in planner_llms]

    async def retrieve_node(state: AgentState) -> AgentState:
        hits = await call_tool(session, "search_code", {"query": state["question"], "top_k": 5})
        return {**state, "retrieved_context": hits}

    async def plan_node(state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content=PLAN_SYSTEM_PROMPT + "\n\n" + TOOL_DESCRIPTIONS + "\n\n" + TOOL_CALL_PROTOCOL_NOTE),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n\n"
                    f"Context gathered so far:\n{_summarize_context(state)}\n\n"
                    f"Steps taken so far: {state['step_count']} / {state['max_steps']}\n"
                    "Decide the next action."
                )
            ),
        ]
        response = await _invoke_with_retry(planners, messages)
        if not response.tool_calls:
            # Model answered in prose instead of calling a tool despite instructions --
            # treat that as "done" rather than crash; answer_node still produces the
            # real final answer from the gathered context.
            plan_json = json.dumps({"done": True, "tool": None, "args": {}, "reasoning": "model answered directly"})
        else:
            call = response.tool_calls[0]
            if call["name"] == FINISH_TOOL_NAME:
                plan_json = json.dumps(
                    {"done": True, "tool": None, "args": {}, "reasoning": call["args"].get("reasoning", "")}
                )
            else:
                plan_json = json.dumps({"done": False, "tool": call["name"], "args": call["args"], "reasoning": ""})
        return {**state, "plan": plan_json}

    async def tool_node(state: AgentState) -> AgentState:
        decision = json.loads(state["plan"])
        tool_name = decision["tool"]
        # The model sometimes passes explicit nulls for omitted optional args (allowed by
        # the tool schemas above so it doesn't get rejected for doing so) -- strip them so
        # the underlying MCP tool's own defaults apply instead of an explicit None.
        args = {k: v for k, v in (decision.get("args") or {}).items() if v is not None}
        try:
            result = await call_tool(session, tool_name, args)
        except Exception as exc:
            result = {"error": str(exc)}
        tool_calls = state["tool_calls"] + [{"tool": tool_name, "args": args, "result": result}]
        return {**state, "tool_calls": tool_calls, "step_count": state["step_count"] + 1}

    async def answer_node(state: AgentState) -> AgentState:
        exhausted = state["step_count"] >= state["max_steps"]
        if exhausted:
            logger.warning(
                "step budget exhausted for question=%r after %d steps", state["question"], state["step_count"]
            )
        note = (
            "\nNote: the step budget was exhausted before investigation felt complete -- "
            "answer with what was gathered so far.\n"
            if exhausted
            else ""
        )
        messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {state['question']}\n\n{_summarize_context(state)}\n{note}"),
        ]
        response = await _invoke_with_retry(answer_llms, messages)
        return {**state, "answer": response.content}

    def route_after_plan(state: AgentState) -> str:
        if state["step_count"] >= state["max_steps"]:
            logger.warning(
                "step budget exhausted, forcing answer_node for question=%r", state["question"]
            )
            return "answer_node"
        decision = json.loads(state["plan"])
        if decision.get("done"):
            return "answer_node"
        return "tool_node"

    graph = StateGraph(AgentState)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("plan_node", plan_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("answer_node", answer_node)

    graph.set_entry_point("retrieve_node")
    graph.add_edge("retrieve_node", "plan_node")
    graph.add_conditional_edges(
        "plan_node", route_after_plan, {"answer_node": "answer_node", "tool_node": "tool_node"}
    )
    graph.add_edge("tool_node", "plan_node")
    graph.add_edge("answer_node", END)

    return graph.compile()


def new_state(question: str, max_steps: int = 6) -> AgentState:
    return AgentState(
        question=question,
        plan=None,
        tool_calls=[],
        retrieved_context=[],
        answer=None,
        step_count=0,
        max_steps=max_steps,
    )
