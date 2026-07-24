"""
Web server untuk App Builder multi-agent: FastAPI + WebSocket.

Alur:
- Browser terhubung lewat WebSocket /ws (satu koneksi = satu sesi / thread_id).
- Pesan user masuk -> graph PM berjalan -> PM mendelegasikan ke spesialis.
- Stream ganda ke browser:
  * mode "updates" : langkah-langkah PM (pesan, tool call, interrupt/pertanyaan).
  * mode "custom"  : langkah-langkah tiap agent spesialis (dikirim dari dalam
    assign_task lewat get_stream_writer), sudah berlabel agent masing-masing.
- Jika PM memanggil ask_user, graph berhenti (interrupt) dan pertanyaan dikirim ke
  browser; pesan user berikutnya melanjutkan eksekusi (Command(resume=...)).
"""

import os
import uuid

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.agent import (
    WORKSPACE,
    _text_of,
    build_graph,
    cleanup_session,
    reset_task_budget,
    set_session_workspace,
    summarize_args,
)
from app.config import AGENTS
from app.language import resolve_session_language, start_session
from app.protocol import is_failure

app = FastAPI(title="App Builder — Tim Multi-Agent")
graph = build_graph()

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/api/agents")
async def agents():
    """Daftar agent untuk membangun tab di UI (tanpa membocorkan api key)."""
    return JSONResponse([
        {"key": key, "name": cfg["name"], "emoji": cfg["emoji"], "model": cfg["model"]}
        for key, cfg in AGENTS.items()
    ])


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
    """Struktur file untuk panel di browser — hanya milik sesi ini (thread_id)."""
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


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    thread_id = str(uuid.uuid4())
    # Isolate this session's generated files under WORKSPACE/<thread_id>/.
    set_session_workspace(thread_id)
    # Each handler runs in its own task, so this ContextVar write is session-local.
    language = start_session(thread_id)
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
    waiting_for_answer = False

    # Let the client scope its file-tree requests to this session.
    await ws.send_json({"type": "session", "thread_id": thread_id})
    await ws.send_json({"type": "language", "code": language})

    try:
        while True:
            data = await ws.receive_json()
            text = (data.get("content") or "").strip()
            if not text:
                continue

            # Response language follows the language the user actually writes in, and is
            # sticky: a short ambiguous follow-up never flips it.
            resolved = resolve_session_language(text, thread_id)
            if resolved != language:
                language = resolved
                await ws.send_json({"type": "language", "code": language})

            if waiting_for_answer:
                graph_input = Command(resume=text)
                waiting_for_answer = False
            else:
                graph_input = {"messages": [HumanMessage(content=text)]}
                # Permintaan baru dari user = jatah delegasi PM di-reset.
                reset_task_budget(thread_id)

            await ws.send_json({"type": "working"})

            try:
                async for mode, chunk in graph.astream(
                    graph_input, config, stream_mode=["updates", "custom"]
                ):
                    # ---- Langkah agent spesialis (dari dalam assign_task) ----
                    if mode == "custom":
                        await ws.send_json(chunk)
                        continue

                    # ---- Langkah PM (mode "updates") ----
                    if "__interrupt__" in chunk:
                        intr = chunk["__interrupt__"][0]
                        await ws.send_json({
                            "agent": "pm", "type": "question", "content": str(intr.value),
                        })
                        waiting_for_answer = True
                        continue

                    if "pm" in chunk:
                        msg = chunk["pm"]["messages"][-1]
                        for tc in getattr(msg, "tool_calls", None) or []:
                            await ws.send_json({
                                "agent": "pm", "type": "tool_call",
                                "name": tc["name"],
                                "detail": summarize_args(tc["name"], tc.get("args") or {}),
                            })
                        content = _text_of(msg.content)
                        if content:
                            await ws.send_json({"agent": "pm", "type": "msg", "content": content})

                    elif "tools" in chunk:
                        for m in chunk["tools"]["messages"]:
                            name = getattr(m, "name", "tool")
                            if name == "ask_user":
                                continue
                            result = _text_of(m.content)
                            await ws.send_json({
                                "agent": "pm",
                                "type": "report" if name == "assign_task" else "tool_result",
                                "name": name,
                                "ok": not is_failure(result),
                                "content": result[:1500],
                            })
            except Exception as e:
                # The message itself is technical detail; the client prefixes it with a
                # localized label rather than receiving prose in a fixed language.
                await ws.send_json({"agent": "pm", "type": "error", "content": str(e)})

            if not waiting_for_answer:
                await ws.send_json({"type": "done"})
    except WebSocketDisconnect:
        pass
    finally:
        # Release per-session in-process state on both normal disconnect and exception.
        cleanup_session(thread_id)


if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8020, reload=False)
