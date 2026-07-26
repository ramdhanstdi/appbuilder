"""Durable session storage so a user can close the app and resume later.

A session is identified by a single stable `thread_id` that ties together three durable
layers plus the generated workspace:

- The PM graph checkpoint (LangGraph AsyncSqliteSaver, one shared DB) — the Project
  Manager's own conversation and any paused `ask_user` interrupt.
- `specialists.json` — each specialist's message history, so the team remembers what it
  already built and decided across restarts.
- `events.jsonl` — every message and activity the browser was shown, replayed verbatim on
  reconnect so no chat from the PM or the team is ever lost.
- `meta.json` — title, timestamps, response language, and cumulative metrics.

Everything lives under SESSIONS_DIR/<thread_id>/, with the shared checkpoint DB alongside.
"""

import json
import os
import re
import time

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.environ.get("SESSIONS_DIR", os.path.join(_BASE_DIR, "sessions"))
CHECKPOINT_DB = os.environ.get("CHECKPOINT_DB", os.path.join(SESSIONS_DIR, "checkpoints.sqlite"))

# A thread_id becomes a directory name and a checkpoint key, so it must be a single safe
# path segment. New ids are uuid hex; this also accepts the older dashed uuid form.
_THREAD_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Event types that carry conversation/activity and must be replayed on reconnect. Transient
# UI signals (working/done/session/language/replay markers) and live metrics are not stored;
# metrics are persisted as a cumulative total in meta instead.
_PERSISTED_EVENT_TYPES = frozenset({
    "user", "msg", "task", "peer_in", "peer_reply", "report",
    "question", "tool_call", "tool_result", "error", "retry",
})


def valid_thread_id(thread_id: str | None) -> bool:
    return bool(thread_id and _THREAD_ID_RE.match(thread_id))


def ensure_root() -> None:
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def session_dir(thread_id: str) -> str:
    """Absolute directory for a session, created on demand. Rejects unsafe ids."""
    if not valid_thread_id(thread_id):
        raise ValueError(f"invalid thread_id: {thread_id!r}")
    path = os.path.join(SESSIONS_DIR, thread_id)
    os.makedirs(path, exist_ok=True)
    return path


def session_exists(thread_id: str) -> bool:
    return valid_thread_id(thread_id) and os.path.isdir(os.path.join(SESSIONS_DIR, thread_id))


def should_persist(event: dict) -> bool:
    return event.get("type") in _PERSISTED_EVENT_TYPES


# --- event log ---------------------------------------------------------------
def append_event(thread_id: str, event: dict) -> None:
    """Append one outbound event to the session's replay log (best effort)."""
    if not should_persist(event):
        return
    try:
        path = os.path.join(session_dir(thread_id), "events.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Persistence must never take a live run down with it.
        pass


def read_events(thread_id: str) -> list[dict]:
    """The full ordered event log for a session, for replay. Empty if none/unreadable."""
    path = os.path.join(SESSIONS_DIR, thread_id, "events.jsonl")
    if not os.path.isfile(path):
        return []
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception:
        pass
    return events


# --- specialist history ------------------------------------------------------
def save_specialists(thread_id: str, data: dict) -> None:
    try:
        path = os.path.join(session_dir(thread_id), "specialists.json")
        _atomic_write(path, json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


def load_specialists(thread_id: str) -> dict | None:
    path = os.path.join(SESSIONS_DIR, thread_id, "specialists.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# --- metadata ----------------------------------------------------------------
def load_meta(thread_id: str) -> dict:
    path = os.path.join(SESSIONS_DIR, thread_id, "meta.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def update_meta(thread_id: str, **changes) -> dict:
    """Merge changes into a session's meta and stamp updated_at."""
    meta = load_meta(thread_id)
    meta.update(changes)
    now = time.time()
    meta.setdefault("created_at", now)
    meta["updated_at"] = now
    try:
        _atomic_write(
            os.path.join(session_dir(thread_id), "meta.json"),
            json.dumps(meta, ensure_ascii=False, indent=2),
        )
    except Exception:
        pass
    return meta


def list_sessions() -> list[dict]:
    """All resumable sessions, most recently updated first."""
    if not os.path.isdir(SESSIONS_DIR):
        return []
    sessions = []
    for name in os.listdir(SESSIONS_DIR):
        if not valid_thread_id(name) or not os.path.isdir(os.path.join(SESSIONS_DIR, name)):
            continue
        meta = load_meta(name)
        sessions.append({
            "thread_id": name,
            "title": meta.get("title") or "(untitled)",
            "updated_at": meta.get("updated_at", 0),
            "created_at": meta.get("created_at", 0),
        })
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return sessions


def _atomic_write(path: str, content: str) -> None:
    """Write via a temp file + replace so a crash never leaves a half-written JSON file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(tmp, path)
