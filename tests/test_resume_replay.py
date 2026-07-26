"""Reconnect behavior: a saved session is replayed verbatim; a new one is separate.

The event log and session metadata are file-based, so this exercises the resume/replay
path deterministically without depending on the SQLite checkpoint. (Durable PM memory
across a real restart is covered by the checkpointer itself and verified manually against
a live server; TestClient's portal thread does not persist aiosqlite writes.)
"""

import importlib

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "sessions" / "cp.sqlite"))
    monkeypatch.setenv("PM_WORKSPACE", str(tmp_path / "workspace"))
    monkeypatch.setenv("RUNS_DIR", str(tmp_path / "runs"))

    from app import session_store
    importlib.reload(session_store)
    from app import server
    importlib.reload(server)

    from fastapi.testclient import TestClient

    from app import agent

    class FakePM:
        """Answers with plain text and no tool calls, so a turn completes at once."""
        async def ainvoke(self, messages):
            return AIMessage(
                content="Halo! Siap membantu.",
                usage_metadata={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14},
            )

    monkeypatch.setattr(agent, "_pm_llm", FakePM())
    return TestClient(server.app)


def _run_turn(ws, text):
    ws.send_json({"content": text})
    events = []
    while True:
        ev = ws.receive_json()
        events.append(ev)
        if ev["type"] == "done":
            return events


def test_new_session_reports_a_fresh_thread_id(client):
    with client, client.websocket_connect("/ws") as ws:
        session = ws.receive_json()
        assert session["type"] == "session"
        assert session["resumed"] is False
        assert session["thread_id"]


def test_reconnect_replays_the_whole_conversation(client):
    with client:
        with client.websocket_connect("/ws") as ws:
            tid = ws.receive_json()["thread_id"]
            ws.receive_json()  # language
            _run_turn(ws, "Buatkan aplikasi todo")

        # Reopen the same session.
        with client.websocket_connect(f"/ws?thread_id={tid}") as ws:
            session = ws.receive_json()
            assert session["resumed"] is True

            assert ws.receive_json()["type"] == "replay_start"
            replayed = []
            while True:
                ev = ws.receive_json()
                if ev["type"] == "replay_end":
                    break
                replayed.append(ev)

            kinds = [e["type"] for e in replayed]
            assert "user" in kinds          # the user's own message is restored
            assert "msg" in kinds           # the PM's reply is restored
            user_text = next(e["content"] for e in replayed if e["type"] == "user")
            assert user_text == "Buatkan aplikasi todo"

            assert ws.receive_json()["type"] == "language"
            assert ws.receive_json()["type"] == "done"   # not mid-question, so ready


def test_unknown_thread_id_starts_a_new_session_not_a_resume(client):
    with client, client.websocket_connect("/ws?thread_id=doesnotexist123") as ws:
        session = ws.receive_json()
        # A thread_id with no stored session is treated as a fresh one, not resumed.
        assert session["resumed"] is False
        assert session["thread_id"] != "doesnotexist123"


def test_sessions_endpoint_lists_saved_sessions_with_titles(client):
    with client:
        with client.websocket_connect("/ws") as ws:
            tid = ws.receive_json()["thread_id"]
            ws.receive_json()
            _run_turn(ws, "Bikin landing page")

        listing = client.get("/api/sessions").json()
        assert any(s["thread_id"] == tid for s in listing)
        row = next(s for s in listing if s["thread_id"] == tid)
        assert row["title"] == "Bikin landing page"


def test_a_second_new_session_is_independent(client):
    with client:
        with client.websocket_connect("/ws") as ws:
            tid1 = ws.receive_json()["thread_id"]
            ws.receive_json()
            _run_turn(ws, "Proyek satu")
        with client.websocket_connect("/ws") as ws:
            tid2 = ws.receive_json()["thread_id"]
        assert tid1 != tid2
        # The first session's log is untouched by the second.
        from app import session_store
        assert any(e["type"] == "user" for e in session_store.read_events(tid1))
        assert session_store.read_events(tid2) == []
