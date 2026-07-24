"""Per-session state: released on disconnect, and bounded while the session runs."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app import agent as agent_module
from app.agent import MAX_HISTORY_MESSAGES, _trim_history, cleanup_session, reset_task_budget
from app.language import _PINNED, pin_language


def test_cleanup_session_releases_history_and_budget():
    thread_id = "session-to-drop"
    agent_module._SPECIALIST_HISTORY[thread_id] = {"ba": [HumanMessage(content="hi")]}
    reset_task_budget(thread_id)
    pin_language(thread_id, "en")
    assert thread_id in agent_module._SPECIALIST_HISTORY
    assert thread_id in agent_module._ASSIGN_BUDGET
    assert thread_id in _PINNED

    cleanup_session(thread_id)

    assert thread_id not in agent_module._SPECIALIST_HISTORY
    assert thread_id not in agent_module._ASSIGN_BUDGET
    assert thread_id not in _PINNED


def test_cleanup_session_tolerates_an_unknown_thread():
    cleanup_session("never-seen")  # must not raise


def test_cleanup_session_leaves_other_sessions_untouched():
    agent_module._SPECIALIST_HISTORY["keep"] = {"qa": []}
    agent_module._SPECIALIST_HISTORY["drop"] = {"qa": []}
    cleanup_session("drop")
    assert "keep" in agent_module._SPECIALIST_HISTORY


def _pair(index: int) -> list:
    """An AIMessage with a tool call plus its matching ToolMessage reply."""
    call_id = f"call-{index}"
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": "read_code_file", "args": {"file_path": "a"}, "id": call_id}],
        ),
        ToolMessage(content="content", tool_call_id=call_id, name="read_code_file"),
    ]


def test_short_history_is_returned_unchanged():
    history = [HumanMessage(content=str(i)) for i in range(5)]
    assert _trim_history(history) is history


def test_long_history_is_trimmed_to_the_cap():
    history = [HumanMessage(content=str(i)) for i in range(200)]
    trimmed = _trim_history(history)
    assert len(trimmed) <= MAX_HISTORY_MESSAGES
    # The window keeps the most recent messages, not the oldest.
    assert trimmed[-1].content == "199"


def test_trimming_never_starts_with_an_orphaned_tool_message():
    history = []
    for i in range(120):
        history.extend(_pair(i))
    trimmed = _trim_history(history)
    assert not isinstance(trimmed[0], ToolMessage)


def test_every_surviving_tool_call_keeps_its_reply():
    history = [HumanMessage(content="task")]
    for i in range(120):
        history.extend(_pair(i))
    trimmed = _trim_history(history)

    replied_to = {m.tool_call_id for m in trimmed if isinstance(m, ToolMessage)}
    for message in trimmed:
        for call in getattr(message, "tool_calls", None) or []:
            assert call["id"] in replied_to


def test_trimmed_tool_messages_all_have_a_preceding_call():
    history = []
    for i in range(120):
        history.extend(_pair(i))
    trimmed = _trim_history(history)

    announced = set()
    for message in trimmed:
        for call in getattr(message, "tool_calls", None) or []:
            announced.add(call["id"])
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in announced
