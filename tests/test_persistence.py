"""Persistence wiring: specialist-history serialization and cumulative metrics."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import agent as agent_module
from app.agent import export_specialist_history, import_specialist_history
from app.server import _add_totals


def test_specialist_history_round_trips_through_json(clean_session_state):
    tid = "round-trip"
    agent_module._SPECIALIST_HISTORY[tid] = {
        "ba": [
            HumanMessage(content="write the spec"),
            AIMessage(
                content="",
                tool_calls=[{"name": "write_code_file",
                             "args": {"file_path": "my-app/docs/SPEC.md"}, "id": "c1"}],
            ),
            ToolMessage(content="OK: file saved", tool_call_id="c1", name="write_code_file"),
            AIMessage(content="STATUS: DONE"),
        ],
    }
    dumped = export_specialist_history(tid)

    # Simulate a restart: drop in-memory state, then rehydrate from the dump.
    agent_module._SPECIALIST_HISTORY.clear()
    import_specialist_history(tid, dumped)

    restored = agent_module._SPECIALIST_HISTORY[tid]["ba"]
    assert [type(m).__name__ for m in restored] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    # The tool-call plumbing must survive so the provider accepts the resumed history.
    assert restored[1].tool_calls[0]["id"] == "c1"
    assert restored[2].tool_call_id == "c1"
    assert restored[3].content == "STATUS: DONE"


def test_import_specialist_history_tolerates_empty_and_unknown(clean_session_state):
    import_specialist_history("x", None)          # must not raise
    import_specialist_history("x", {})            # must not raise
    import_specialist_history("x", {"not_an_agent": [{"type": "human", "data": {"content": "h"}}]})
    assert agent_module._SPECIALIST_HISTORY.get("x", {}) == {}


def test_export_of_unknown_session_is_empty(clean_session_state):
    assert export_specialist_history("never") == {}


def test_add_totals_accumulates_cost_across_restarts():
    base = {"total_tokens": 1000, "cost_usd": 0.5, "llm_calls": 4, "retries": 1}
    delta = {"total_tokens": 250, "cost_usd": 0.12, "llm_calls": 2, "retries": 0}
    out = _add_totals(base, delta)
    assert out["total_tokens"] == 1250
    assert out["cost_usd"] == 0.62
    assert out["llm_calls"] == 6
    assert out["retries"] == 1


def test_add_totals_handles_empty_base():
    out = _add_totals({}, {"total_tokens": 42, "cost_usd": 0.01})
    assert out["total_tokens"] == 42
    assert out["cost_usd"] == 0.01
    assert out["llm_calls"] == 0
