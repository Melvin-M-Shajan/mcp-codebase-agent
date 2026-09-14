"""§14 integration test: run the full agent graph against one fixture question with a
mocked MCP client and a mocked LLM, asserting it terminates within max_steps and
produces a non-empty answer -- no real API keys or network calls involved."""

import itertools

import pytest

import src.agent.graph as graph_module
from src.agent.graph import FINISH_TOOL_NAME, build_graph, new_state


class _FakeToolCallResponse:
    def __init__(self, name, args):
        self.tool_calls = [{"name": name, "args": args, "id": "fake-id", "type": "tool_call"}]


def _finish(reasoning="enough context"):
    return _FakeToolCallResponse(FINISH_TOOL_NAME, {"reasoning": reasoning})


def _call(tool, args):
    return _FakeToolCallResponse(tool, args)


class _FakeBoundLLM:
    def __init__(self, responses):
        self._responses = iter(responses)

    async def ainvoke(self, messages):
        return next(self._responses)


class _FakeAnswer:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, responses, answer_text="Fake grounded answer citing foo.py:1-2."):
        self._responses = responses
        self._answer_text = answer_text

    def bind_tools(self, tools, **kwargs):
        return _FakeBoundLLM(self._responses)

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
        llm = FakeLLM([_finish()])
        graph = build_graph(session=object(), planner_llm=llm, answer_llm=llm)

        final_state = await graph.ainvoke(new_state("Where is foo implemented?", max_steps=6))

        assert final_state["answer"]
        assert final_state["step_count"] == 0
        assert final_state["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_calls_tool_then_answers_when_plan_says_not_done_once(self):
        responses = [_call("read_file", {"path": "foo.py"}), _finish("enough now")]
        llm = FakeLLM(responses)
        graph = build_graph(session=object(), planner_llm=llm, answer_llm=llm)

        final_state = await graph.ainvoke(new_state("Why does foo fail?", max_steps=6))

        assert final_state["answer"]
        assert final_state["step_count"] == 1
        assert len(final_state["tool_calls"]) == 1
        assert final_state["tool_calls"][0]["tool"] == "read_file"

    @pytest.mark.asyncio
    async def test_step_budget_forces_answer_when_plan_never_says_done(self):
        never_done = itertools.cycle([_call("search_code", {"query": "x"})])
        llm = FakeLLM(never_done)
        graph = build_graph(session=object(), planner_llm=llm, answer_llm=llm)

        final_state = await graph.ainvoke(new_state("An unanswerable question", max_steps=3))

        assert final_state["answer"]
        assert final_state["step_count"] == 3
        assert len(final_state["tool_calls"]) == 3
