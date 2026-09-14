"""§14 integration test: run the full agent graph against one fixture question with a
mocked MCP client and a mocked LLM, asserting it terminates within max_steps and
produces a non-empty answer -- no real API keys or network calls involved."""

import itertools

import pytest

import src.agent.graph as graph_module
from src.agent.graph import PlanDecision, build_graph, new_state


class _FakeStructuredLLM:
    def __init__(self, decisions):
        self._decisions = iter(decisions)

    async def ainvoke(self, messages):
        return next(self._decisions)


class _FakeAnswer:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, decisions, answer_text="Fake grounded answer citing foo.py:1-2."):
        self._decisions = decisions
        self._answer_text = answer_text

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self._decisions)

    async def ainvoke(self, messages):
        return _FakeAnswer(self._answer_text)


async def _fake_call_tool(session, name, args):
    if name == "search_code":
        return [{"file_path": "foo.py", "start_line": 1, "end_line": 2, "snippet": "def foo(): ...", "score": 0.9}]
    return {"fake": "result", "tool": name}


@pytest.fixture(autouse=True)
def patch_call_tool(monkeypatch):
    monkeypatch.setattr(graph_module, "call_tool", _fake_call_tool)


class TestAgentGraph:
    @pytest.mark.asyncio
    async def test_terminates_with_answer_when_plan_says_done_immediately(self):
        decisions = [PlanDecision(done=True, tool=None, args={}, reasoning="enough context")]
        llm = FakeLLM(decisions)
        graph = build_graph(session=object(), llm=llm)

        final_state = await graph.ainvoke(new_state("Where is foo implemented?", max_steps=6))

        assert final_state["answer"]
        assert final_state["step_count"] == 0
        assert final_state["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_calls_tool_then_answers_when_plan_says_not_done_once(self):
        decisions = [
            PlanDecision(done=False, tool="read_file", args={"path": "foo.py"}, reasoning="need the file"),
            PlanDecision(done=True, tool=None, args={}, reasoning="enough now"),
        ]
        llm = FakeLLM(decisions)
        graph = build_graph(session=object(), llm=llm)

        final_state = await graph.ainvoke(new_state("Why does foo fail?", max_steps=6))

        assert final_state["answer"]
        assert final_state["step_count"] == 1
        assert len(final_state["tool_calls"]) == 1
        assert final_state["tool_calls"][0]["tool"] == "read_file"

    @pytest.mark.asyncio
    async def test_step_budget_forces_answer_when_plan_never_says_done(self):
        never_done = itertools.cycle(
            [PlanDecision(done=False, tool="search_code", args={"query": "x"}, reasoning="keep looking")]
        )
        llm = FakeLLM(never_done)
        graph = build_graph(session=object(), llm=llm)

        final_state = await graph.ainvoke(new_state("An unanswerable question", max_steps=3))

        assert final_state["answer"]
        assert final_state["step_count"] == 3
        assert len(final_state["tool_calls"]) == 3
