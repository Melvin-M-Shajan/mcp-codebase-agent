"""§10 demo -- simple REPL: reads a question from stdin, runs the agent graph, prints
the final answer plus a compact log of which tools were called in what order."""

import asyncio

from dotenv import load_dotenv

from src.agent.graph import build_graph, new_state
from src.agent.llm import get_llm
from src.agent.mcp_client import mcp_session


async def ask(session, llm, question: str, max_steps: int = 6) -> dict:
    graph = build_graph(session, llm)
    state = new_state(question, max_steps=max_steps)
    return await graph.ainvoke(state)


def format_tool_log(state: dict) -> str:
    if not state["tool_calls"]:
        return "  (no tool calls -- answered from RAG context alone)"
    return "\n".join(f"  {i + 1}. {c['tool']}({c['args']})" for i, c in enumerate(state["tool_calls"]))


async def main_async() -> None:
    load_dotenv()
    llm = get_llm()
    async with mcp_session() as session:
        print("MCP Codebase Agent -- ask a question about the indexed repo ('exit' to quit)\n")
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                break
            if not question or question.lower() in ("exit", "quit"):
                break

            final_state = await ask(session, llm, question)

            print(f"\n{final_state['answer']}\n")
            print("Tools called (in order):")
            print(format_tool_log(final_state))
            print()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
