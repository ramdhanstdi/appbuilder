"""Web server for the App Builder multi-agent team: FastAPI + WebSocket.

Flow:
- The browser connects over WebSocket /ws. Each connection resumes an existing session
  (its thread_id, sent as a query parameter) or starts a new one.
- A user message runs the PM graph, which delegates to the specialists. Two stream modes
  reach the browser: "updates" (the PM's own steps) and "custom" (each specialist's steps,
  emitted from inside assign_task via get_stream_writer).
- If the PM calls ask_user the graph pauses (interrupt) and the question is sent to the
  browser; the next message resumes it (Command(resume=...)).

Durability (see app/session_store.py): the PM graph uses a persistent AsyncSqliteSaver, the
team's specialist histories are written to disk, and every message and activity shown to
the browser is logged and replayed on reconnect — so a user can close the app and continue
later without losing any chat from the PM or the team.
"""

import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app import session_store
from app.agent import (
    WORKSPACE,
    _text_of,
    build_graph,
    cleanup_session,
    export_specialist_history,
    import_specialist_history,
    reset_task_budget,
    set_session_workspace,
    summarize_args,
)
from app.config import AGENTS
from app.language import (
    language_is_pinned,
    resolve_session_language,
    restore_language,
    start_session,
)
from app.metrics import METRICS
from app.protocol import is_failure

# Set during startup (lifespan) with a durable checkpointer, so PM sessions — including a
# paused ask_user interrupt — survive a server restart.
GRAPH = None

_METRIC_KEYS = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "llm_calls", "tool_calls", "retries", "latency_seconds", "cost_usd",
)


def _add_totals(base: dict, delta: dict) -> dict:
    """Add two metrics totals dicts, so cumulative session cost survives restarts."""
    out = {}
    for key in _METRIC_KEYS:
        value = (base.get(key, 0) or 0) + (delta.get(key, 0) or 0)
        out[key] = round(value, 6) if isinstance(value, float) else value
    return out


@asynccontextmanager
async def lifespan(_app: FastAPI):
    session_store.ensure_root()
    # The checkpointer connection stays open for the app's lifetime.
    async with AsyncSqliteSaver.from_conn_string(session_store.CHECKPOINT_DB) as saver:
        global GRAPH
        GRAPH = build_graph(saver)
        try:
            yield
        finally:
            GRAPH = None


app = FastAPI(title="App Builder — Multi-Agent Team", lifespan=lifespan)

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/api/agents")
async def agents():
    """Agent list for building the UI tabs (without leaking any API key)."""
    return JSONResponse([
        {"key": key, "name": cfg["name"], "emoji": cfg["emoji"], "model": cfg["model"]}
        for key, cfg in AGENTS.items()
    ])


@app.get("/api/sessions")
async def sessions():
    """Resumable sessions, most recently updated first, for the session switcher."""
    return JSONResponse(session_store.list_sessions())


def _session_dir(thread_id: str | None) -> str:
    """Resolve a session's workspace directory, jailed under WORKSPACE.

    A missing or unsafe thread_id falls back to the base workspace so traversal via the
    query string is impossible (only a single path segment inside WORKSPACE is honored).
    """
    if not thread_id:
        return WORKSPACE
    segment = os.path.basename(thread_id.strip())
    full = os.path.abspath(os.path.join(WORKSPACE, segment))
    if full == WORKSPACE or full.startswith(WORKSPACE + os.sep):
        return full
    return WORKSPACE


@app.get("/api/files")
async def files(thread_id: str | None = None):
    """File tree for the browser panel — this session's workspace only (thread_id)."""
    root_dir = _session_dir(thread_id)
    tree = []
    if not os.path.isdir(root_dir):
        return JSONResponse(tree)
    for root, dirs, filenames in os.walk(root_dir):
        dirs[:] = sorted(
            d for d in dirs if d not in ("node_modules", ".git", "dist", "__pycache__")
        )
        rel = os.path.relpath(root, root_dir)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > 5:
            dirs[:] = []
            continue
        if rel != ".":
            tree.append({"path": rel, "type": "dir", "depth": depth - 1})
        for fname in sorted(filenames):
            tree.append({
                "path": fname if rel == "." else os.path.join(rel, fname),
                "type": "file",
                "depth": depth,
            })
        if len(tree) > 500:
            break
    return JSONResponse(tree)


@app.get("/api/metrics/{thread_id}")
async def metrics(thread_id: str):
    """Token, cost, and latency summary for a session, broken down per agent."""
    return JSONResponse(METRICS.summary(thread_id))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    async def emit(event: dict, *, persist: bool = True) -> None:
        """Send an event to the browser and, unless replaying, log it for future replay."""
        await ws.send_json(event)
        if persist:
            session_store.append_event(thread_id, event)

    # Resume the requested session if it exists; otherwise mint a fresh one.
    requested = ws.query_params.get("thread_id")
    resuming = session_store.session_exists(requested)
    thread_id = requested if resuming else uuid.uuid4().hex
    session_store.session_dir(thread_id)  # ensure the directory exists

    # Same thread_id ties together the workspace, the checkpoint, and the stored history.
    set_session_workspace(thread_id)
    meta = session_store.load_meta(thread_id)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
    title_set = bool(meta.get("title"))
    metrics_base = meta.get("metrics") or {}

    if resuming:
        language = restore_language(
            thread_id, meta.get("language"), bool(meta.get("language_pinned"))
        )
        import_specialist_history(thread_id, session_store.load_specialists(thread_id))
    else:
        language = start_session(thread_id)

    # Tell the client which session it is and restore its cumulative metrics footer.
    await emit(
        {"type": "session", "thread_id": thread_id, "resumed": resuming, "metrics": metrics_base},
        persist=False,
    )

    # Replay the full stored conversation so no PM or team chat is lost on reconnect.
    if resuming:
        events = session_store.read_events(thread_id)
        await emit({"type": "replay_start", "count": len(events)}, persist=False)
        for event in events:
            await emit(event, persist=False)
        await emit({"type": "replay_end"}, persist=False)

    await emit({"type": "language", "code": language}, persist=False)

    # Restore the paused/ready state. A pending ask_user is tracked in meta (authoritative)
    # and the durable checkpoint holds the actual paused graph, so the next message resumes
    # it. Re-arm the input if the session stopped mid-question.
    waiting_for_answer = resuming and bool(meta.get("pending_question"))
    if resuming:
        await emit({"type": "awaiting" if waiting_for_answer else "done"}, persist=False)

    try:
        while True:
            data = await ws.receive_json()
            text = (data.get("content") or "").strip()
            if not text:
                continue

            # Log the user's own message so it reappears on replay (the live UI renders it
            # locally, so this is not echoed back now).
            session_store.append_event(thread_id, {"type": "user", "content": text})
            if not title_set:
                title_set = True
                session_store.update_meta(
                    thread_id, title=text[:80], language=language,
                    language_pinned=language_is_pinned(thread_id),
                )

            # Response language follows the language the user writes in, and is sticky: a
            # short ambiguous follow-up never flips it.
            resolved = resolve_session_language(text, thread_id)
            if resolved != language:
                language = resolved
                await emit({"type": "language", "code": language}, persist=False)
                session_store.update_meta(
                    thread_id, language=language,
                    language_pinned=language_is_pinned(thread_id),
                )

            if waiting_for_answer:
                graph_input = Command(resume=text)
                waiting_for_answer = False
                # The question has been answered; the session is no longer paused.
                session_store.update_meta(thread_id, pending_question=None)
            else:
                graph_input = {"messages": [HumanMessage(content=text)]}
                # A new user request resets the PM's delegation budget.
                reset_task_budget(thread_id)
                METRICS.start_request(thread_id, text)

            await emit({"type": "working"}, persist=False)

            try:
                async for mode, chunk in GRAPH.astream(
                    graph_input, config, stream_mode=["updates", "custom"]
                ):
                    # ---- Specialist steps (from inside assign_task) ----
                    if mode == "custom":
                        # Live metrics update the footer but are not part of the chat log;
                        # the cumulative total is persisted in meta instead.
                        await emit(chunk, persist=chunk.get("type") != "metrics")
                        continue

                    # ---- PM steps (mode "updates") ----
                    if "__interrupt__" in chunk:
                        intr = chunk["__interrupt__"][0]
                        await emit({
                            "agent": "pm", "type": "question", "content": str(intr.value),
                        })
                        waiting_for_answer = True
                        # Remember the pause so a reconnect re-arms the input for the answer.
                        session_store.update_meta(thread_id, pending_question=str(intr.value))
                        continue

                    if "pm" in chunk:
                        msg = chunk["pm"]["messages"][-1]
                        for tc in getattr(msg, "tool_calls", None) or []:
                            await emit({
                                "agent": "pm", "type": "tool_call",
                                "name": tc["name"],
                                "detail": summarize_args(tc["name"], tc.get("args") or {}),
                            })
                        content = _text_of(msg.content)
                        if content:
                            await emit({"agent": "pm", "type": "msg", "content": content})

                    elif "tools" in chunk:
                        for m in chunk["tools"]["messages"]:
                            name = getattr(m, "name", "tool")
                            if name == "ask_user":
                                continue
                            result = _text_of(m.content)
                            await emit({
                                "agent": "pm",
                                "type": "report" if name == "assign_task" else "tool_result",
                                "name": name,
                                "ok": not is_failure(result),
                                "content": result[:1500],
                            })
            except Exception as e:
                # The message is technical detail; the client prefixes it with a localized
                # label rather than receiving prose in a fixed language.
                await emit({"agent": "pm", "type": "error", "content": str(e)})

            # Persist the team's context so a later session resumes with full memory.
            session_store.save_specialists(thread_id, export_specialist_history(thread_id))

            if not waiting_for_answer:
                # One JSON Lines record per completed user request, under runs/.
                METRICS.finish_request(thread_id)
                grand = _add_totals(metrics_base, METRICS.totals(thread_id))
                session_store.update_meta(
                    thread_id, metrics=grand, language=language,
                    language_pinned=language_is_pinned(thread_id),
                )
                await emit(
                    {"type": "metrics", "totals": METRICS.totals(thread_id), "final": True},
                    persist=False,
                )
                await emit({"type": "done"}, persist=False)
    except WebSocketDisconnect:
        pass
    finally:
        # Release per-session in-process state on both normal disconnect and exception.
        # Durable state (checkpoint, specialist history, event log) is kept for resume.
        cleanup_session(thread_id)


if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8020, reload=False)
