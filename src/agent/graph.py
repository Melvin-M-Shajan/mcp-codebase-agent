"""§7.2/§7.3 graph nodes and routing.

Nodes are built by `build_graph(session, llm)` as closures over a live MCP
`ClientSession` and a chat model, but each still matches the PRD's `(state: AgentState)
-> AgentState` node signature -- the closure is just how session/llm get in without
changing that signature or resorting to module-level globals.
"""

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.agent.mcp_client import call_tool
from src.agent.prompts import ANSWER_SYSTEM_PROMPT, PLAN_SYSTEM_PROMPT, TOOL_DESCRIPTIONS
from src.agent.state import AgentState

logger = logging.getLogger(__name__)


class PlanDecision(BaseModel):
    done: bool = Field(
        description="True if enough information has been gathered to answer; False to call another tool."
    )
    tool: str | None = Field(
        default=None,
        description="One of search_code, read_file, run_tests, get_recent_commits. Required if done is False.",
    )
    args: dict = Field(default_factory=dict, description="Arguments for the chosen tool.")
    reasoning: str = Field(description="Brief reasoning for this decision.")


def _summarize_context(state: AgentState) -> str:
    parts: list[str] = []
    if state["retrieved_context"]:
        parts.append("Retrieved code (RAG search over the indexed repo):")
        for hit in state["retrieved_context"]:
            parts.append(
                f"- {hit['file_path']}:{hit['start_line']}-{hit['end_line']} (score={hit['score']:.3f})\n{hit['snippet']}"
            )
    if state["tool_calls"]:
        parts.append("\nTool calls made so far:")
        for call in state["tool_calls"]:
            result_preview = json.dumps(call["result"], indent=2, default=str)[:2000]
            parts.append(f"- {call['tool']}({call['args']}) ->\n{result_preview}")
    return "\n".join(parts) if parts else "(nothing gathered yet)"


def build_graph(session, llm):
    structured_llm = llm.with_structured_output(PlanDecision)

    async def retrieve_node(state: AgentState) -> AgentState:
        hits = await call_tool(session, "search_code", {"query": state["question"], "top_k": 8})
        return {**state, "retrieved_context": hits}

    async def plan_node(state: AgentState) -> AgentState:
        messages = [
            SystemMessage(content=PLAN_SYSTEM_PROMPT + "\n\n" + TOOL_DESCRIPTIONS),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n\n"
                    f"Context gathered so far:\n{_summarize_context(state)}\n\n"
                    f"Steps taken so far: {state['step_count']} / {state['max_steps']}\n"
                    "Decide the next action."
                )
            ),
        ]
        decision: PlanDecision = await structured_llm.ainvoke(messages)
        plan_json = json.dumps(
            {"done": decision.done, "tool": decision.tool, "args": decision.args, "reasoning": decision.reasoning}
        )
        return {**state, "plan": plan_json}

    async def tool_node(state: AgentState) -> AgentState:
        decision = json.loads(state["plan"])
        tool_name = decision["tool"]
        args = decision.get("args") or {}
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
        response = await llm.ainvoke(messages)
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
