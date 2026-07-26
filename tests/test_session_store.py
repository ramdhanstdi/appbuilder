"""Durable session storage: the file-based layer behind resume-anytime."""

import importlib

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fresh session_store pointed at a temp directory (module-level dirs are rebound)."""
    monkeypatch.setenv("SESSIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "sessions" / "cp.sqlite"))
    from app import session_store
    importlib.reload(session_store)
    session_store.ensure_root()
    return session_store


def test_valid_thread_id_accepts_uuid_forms_and_rejects_traversal(store):
    assert store.valid_thread_id("abcDEF123")
    assert store.valid_thread_id("6f1e2d3c-4b5a-6789-0123-456789abcdef")  # dashed uuid
    assert not store.valid_thread_id("../escape")
    assert not store.valid_thread_id("a/b")
    assert not store.valid_thread_id("has space")
    assert not store.valid_thread_id("")
    assert not store.valid_thread_id(None)


def test_only_conversation_events_are_persisted(store):
    tid = "sess1"
    for event in [
        {"type": "msg", "content": "hi"},
        {"type": "working"},            # transient, not logged
        {"type": "metrics", "totals": {}},   # footer only, not logged
        {"type": "session", "thread_id": tid},
        {"type": "tool_call", "detail": {}},
        {"type": "user", "content": "build"},
    ]:
        store.append_event(tid, event)
    replayed = store.read_events(tid)
    assert [e["type"] for e in replayed] == ["msg", "tool_call", "user"]


def test_event_log_preserves_order_and_content(store):
    tid = "sess2"
    for i in range(5):
        store.append_event(tid, {"type": "msg", "content": f"m{i}"})
    assert [e["content"] for e in store.read_events(tid)] == ["m0", "m1", "m2", "m3", "m4"]


def test_read_events_is_empty_for_unknown_session(store):
    assert store.read_events("never-existed") == []


def test_specialists_round_trip(store):
    tid = "sess3"
    data = {"ba": [{"type": "human", "data": {"content": "spec it"}}]}
    store.save_specialists(tid, data)
    assert store.load_specialists(tid) == data


def test_load_specialists_missing_returns_none(store):
    assert store.load_specialists("nope") is None


def test_meta_merges_and_stamps_timestamps(store):
    tid = "sess4"
    store.update_meta(tid, title="My todo app", language="id")
    meta = store.update_meta(tid, language="en")  # merge, not replace
    assert meta["title"] == "My todo app"
    assert meta["language"] == "en"
    assert meta["updated_at"] >= meta["created_at"]


def test_pending_question_survives_in_meta(store):
    tid = "sess5"
    store.update_meta(tid, pending_question="What color?")
    assert store.load_meta(tid)["pending_question"] == "What color?"
    store.update_meta(tid, pending_question=None)
    assert store.load_meta(tid)["pending_question"] is None


def test_session_exists_reflects_directory(store):
    tid = "sess6"
    assert not store.session_exists(tid)
    store.session_dir(tid)
    assert store.session_exists(tid)
    assert not store.session_exists("../evil")


def test_list_sessions_orders_by_most_recent(store):
    store.update_meta("old", title="Old", updated_at=100)
    store.update_meta("new", title="New", updated_at=200)
    listed = store.list_sessions()
    ids = [s["thread_id"] for s in listed]
    assert ids.index("new") < ids.index("old")
    assert {s["title"] for s in listed} == {"Old", "New"}


def test_session_dir_rejects_unsafe_id(store):
    with pytest.raises(ValueError):
        store.session_dir("../../etc")
