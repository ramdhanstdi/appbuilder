"""Peer-to-peer discussion guards: unknown targets, self-talk, cycles, depth, and budget."""

import pytest
from langchain_core.messages import AIMessage

from app import agent as agent_module
from app.agent import (
    MAX_ASSIGN_TASKS_PER_REQUEST,
    MAX_DISCUSSION_DEPTH,
    _run_discussion,
    _run_specialist,
    assign_task,
    reset_task_budget,
)
from app.protocol import is_failure


class _FakeLLM:
    """Stands in for a bound ChatOpenAI: records that it was used, answers immediately."""

    def __init__(self, label):
        self.label = label
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content=f"STATUS: DONE\nanswered by {self.label}")


async def test_unknown_target_agent_is_rejected():
    result = await _run_discussion("ba", {"agent": "devops", "message": "hi"}, "t1", ())
    assert is_failure(result)
    assert "devops" in result


async def test_agent_cannot_discuss_with_itself():
    result = await _run_discussion("qa", {"agent": "qa", "message": "hi"}, "t1", ())
    assert is_failure(result)


async def test_target_already_in_the_call_chain_is_rejected():
    # frontend -> backend -> frontend would loop forever; the cycle guard stops it.
    result = await _run_discussion(
        "backend", {"agent": "frontend", "message": "hi"}, "t1", ("frontend",)
    )
    assert is_failure(result)
    assert "chain" in result.lower()


async def test_exceeding_max_discussion_depth_is_rejected():
    deep_chain = tuple("ba" for _ in range(MAX_DISCUSSION_DEPTH))
    result = await _run_discussion("qa", {"agent": "frontend", "message": "hi"}, "t1", deep_chain)
    assert is_failure(result)
    assert "depth" in result.lower()


async def test_at_max_depth_the_specialist_is_bound_without_discuss_with(
    monkeypatch, no_stream_writer
):
    """The structural guarantee: the tool is removed from the binding, not forbidden by prompt."""
    with_discuss = _FakeLLM("with_discuss")
    without_discuss = _FakeLLM("without_discuss")
    monkeypatch.setitem(agent_module._SPECIALIST_LLMS, "backend", with_discuss)
    monkeypatch.setitem(agent_module._SPECIALIST_LLMS_NO_DISCUSS, "backend", without_discuss)

    deep_chain = tuple("ba" for _ in range(MAX_DISCUSSION_DEPTH))
    await _run_specialist("backend", "check this", "t1", source="qa", chain=deep_chain)

    assert without_discuss.calls == 1
    assert with_discuss.calls == 0


async def test_below_max_depth_the_specialist_keeps_discuss_with(monkeypatch, no_stream_writer):
    with_discuss = _FakeLLM("with_discuss")
    without_discuss = _FakeLLM("without_discuss")
    monkeypatch.setitem(agent_module._SPECIALIST_LLMS, "backend", with_discuss)
    monkeypatch.setitem(agent_module._SPECIALIST_LLMS_NO_DISCUSS, "backend", without_discuss)

    await _run_specialist("backend", "build it", "t1", source="pm", chain=())

    assert with_discuss.calls == 1
    assert without_discuss.calls == 0


async def test_assign_task_budget_is_enforced_and_resettable(monkeypatch):
    async def fake_specialist(agent_key, content, thread_id, **kwargs):
        return "STATUS: DONE\nok"

    monkeypatch.setattr(agent_module, "_run_specialist", fake_specialist)
    config = {"configurable": {"thread_id": "budget-test"}}
    reset_task_budget("budget-test")

    for _ in range(MAX_ASSIGN_TASKS_PER_REQUEST):
        result = await assign_task.ainvoke({"agent": "ba", "task": "spec it"}, config=config)
        assert not is_failure(result)

    exhausted = await assign_task.ainvoke({"agent": "ba", "task": "one more"}, config=config)
    assert is_failure(exhausted)
    assert str(MAX_ASSIGN_TASKS_PER_REQUEST) in exhausted

    reset_task_budget("budget-test")
    after_reset = await assign_task.ainvoke({"agent": "ba", "task": "new request"}, config=config)
    assert not is_failure(after_reset)


async def test_assign_task_rejects_an_unknown_agent():
    config = {"configurable": {"thread_id": "unknown-agent"}}
    result = await assign_task.ainvoke({"agent": "designer", "task": "x"}, config=config)
    assert is_failure(result)


@pytest.mark.parametrize("agent_key", ["ba", "frontend", "backend", "qa"])
def test_only_specialists_with_discuss_with_get_the_reduced_binding(agent_key):
    assert agent_key in agent_module._SPECIALIST_LLMS
    assert agent_key in agent_module._SPECIALIST_LLMS_NO_DISCUSS
