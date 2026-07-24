"""
Tim multi-agent: Project Manager (supervisor) + 4 spesialis (BA, Frontend, Backend, QA).

Arsitektur:
- User hanya bicara dengan PM (graph LangGraph utama, dengan checkpointer + interrupt).
- PM mendelegasikan pekerjaan lewat tool `assign_task(agent, task)`.
- assign_task menjalankan loop agent spesialis (LLM + tools masing-masing, dengan
  API base/key/model TERPISAH per agent dari app/config.py).
- Setiap langkah spesialis di-stream ke browser lewat custom stream writer LangGraph,
  sehingga percakapan tiap agent bisa dipantau di tab terpisah.
- `ask_user` (hanya milik PM) memakai interrupt(): eksekusi berhenti sampai user menjawab.
"""

import asyncio
import contextvars
import os
import random
import re
import subprocess
import time
import uuid
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from typing_extensions import TypedDict

from app.config import AGENTS, SPECIALISTS
from app.language import (
    effective_language,
    forget_session,
    language_directive,
    language_name,
    pin_language,
)
from app.metrics import METRICS
from app.protocol import STATUS_PARTIAL, failed, is_failure, ok

# ==========================================
# Workspace: semua app hasil buatan tim masuk ke sini
# ==========================================
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.environ.get("PM_WORKSPACE", os.path.join(_BASE_DIR, "workspace"))
os.makedirs(WORKSPACE, exist_ok=True)

# Per-session workspace root. A ContextVar (not a parameter) so the value propagates
# implicitly through async LangGraph tool execution without threading it everywhere.
# Defaults to the shared WORKSPACE so tools work outside a WebSocket session (e.g. tests).
_SESSION_ROOT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_SESSION_ROOT", default=WORKSPACE
)


def set_session_workspace(thread_id: str) -> str:
    """Create WORKSPACE/<thread_id>/ for a session, pin it on the ContextVar, return it."""
    root = os.path.abspath(os.path.join(WORKSPACE, thread_id))
    os.makedirs(root, exist_ok=True)
    _SESSION_ROOT.set(root)
    return root


_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def _safe_path(path: str) -> str:
    """Jail every file operation inside this session's workspace root.

    Separators are normalized before the check so traversal is rejected identically on
    POSIX and Windows: a backslash is an ordinary filename character on POSIX, which
    would otherwise let '..\\..\\etc' through as a literal name. A leading '/' means
    workspace-relative, not filesystem-absolute; drive-qualified and UNC paths are
    rejected outright since they can only be attempts to leave the sandbox.
    """
    root = os.path.abspath(_SESSION_ROOT.get())
    normalized = (path or "").replace("\\", "/")
    if _DRIVE_RE.match(normalized) or normalized.startswith("//"):
        raise ValueError(f"access denied: path '{path}' is outside the workspace.")
    segments = [p for p in normalized.split("/") if p]
    full = os.path.abspath(os.path.join(root, *segments))
    if not (full == root or full.startswith(root + os.sep)):
        raise ValueError(f"access denied: path '{path}' is outside the workspace.")
    return full


# ==========================================
# Tools dasar (dipakai lintas agent sesuai config)
# ==========================================
@tool
def write_code_file(file_path: str, content: str) -> str:
    """Create or overwrite a single file. Missing folders are created automatically.
    file_path is RELATIVE to the workspace, e.g. 'online-store/src/App.jsx'.
    """
    try:
        full = _safe_path(file_path)
        directory = os.path.dirname(full)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return ok(f"file saved at {os.path.relpath(full, _SESSION_ROOT.get())}")
    except Exception as e:
        return failed(f"could not write file: {e}")


@tool
def read_code_file(file_path: str) -> str:
    """Read the contents of a file in the workspace (path relative to the workspace)."""
    try:
        full = _safe_path(file_path)
        with open(full, encoding="utf-8") as f:
            content = f.read()
        if len(content) > 12000:
            content = content[:12000] + "\n... [truncated]"
        return content
    except Exception as e:
        return failed(f"could not read file: {e}")


@tool
def list_files(path: str = ".") -> str:
    """List the file/folder structure in the workspace (relative path, defaults to the root)."""
    try:
        full = _safe_path(path)
        if not os.path.exists(full):
            return ok(f"folder '{path}' does not exist yet — nothing here.")
        session_root = _SESSION_ROOT.get()
        lines = []
        for root, dirs, files in os.walk(full):
            dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "dist", "__pycache__")]
            rel_root = os.path.relpath(root, session_root)
            depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            if depth > 4:
                dirs[:] = []
                continue
            indent = "  " * depth
            lines.append(f"{indent}{os.path.basename(root) if rel_root != '.' else '.'}/")
            for fname in sorted(files):
                lines.append(f"{indent}  {fname}")
            if len(lines) > 200:
                lines.append("... [truncated]")
                break
        return "\n".join(lines)
    except Exception as e:
        return failed(f"could not list files: {e}")


# ==========================================
# Shell execution sandbox
# ==========================================
# Commands run inside a disposable container by default: only the session workspace is
# mounted, so a command cannot read the host filesystem, and networking is off unless
# explicitly enabled. ALLOW_UNSANDBOXED_COMMANDS=true restores direct host execution for
# local development — that is the posture the README documents as trusted-only.
SANDBOX_COMMANDS = os.environ.get(
    "ALLOW_UNSANDBOXED_COMMANDS", ""
).strip().lower() not in ("1", "true", "yes")
DOCKER_BIN = os.environ.get("DOCKER_BIN", "docker")
RUNNER_IMAGE = os.environ.get("RUNNER_IMAGE", "appbuilder-runner:latest")
# 'none' still gives the container its own loopback, so QA smoke tests against
# localhost work while outbound access does not. Set to 'bridge' when a task genuinely
# needs to install packages.
RUNNER_NETWORK = os.environ.get("RUNNER_NETWORK", "none")
RUNNER_MEMORY = os.environ.get("RUNNER_MEMORY", "512m")
RUNNER_CPUS = os.environ.get("RUNNER_CPUS", "2")
RUNNER_USER = os.environ.get("RUNNER_USER", "1000:1000")
# Set when the server itself runs in a container: the runner then reuses the server's
# mounts instead of bind-mounting a host path the server cannot name.
RUNNER_VOLUMES_FROM = os.environ.get("RUNNER_VOLUMES_FROM", "").strip()


def _runner_argv(container_name: str, command: str, cwd: str) -> list[str]:
    """Build the `docker run` argv that executes one agent command in isolation."""
    session_root = os.path.abspath(_SESSION_ROOT.get())
    argv = [
        DOCKER_BIN, "run", "--rm", "--name", container_name,
        "--network", RUNNER_NETWORK,
        "--memory", RUNNER_MEMORY,
        "--cpus", RUNNER_CPUS,
        "--pids-limit", "512",
        "--user", RUNNER_USER,
        "--security-opt", "no-new-privileges",
    ]
    if RUNNER_VOLUMES_FROM:
        # Same mount, same absolute path, so cwd needs no translation.
        argv += ["--volumes-from", RUNNER_VOLUMES_FROM, "-w", cwd]
    else:
        relative = os.path.relpath(cwd, session_root).replace("\\", "/")
        workdir = "/workspace" if relative == "." else f"/workspace/{relative}"
        argv += ["-v", f"{session_root}:/workspace", "-w", workdir]
    return argv + [RUNNER_IMAGE, "sh", "-lc", command]


def _kill_process_group(proc) -> None:
    """SIGKILL the command's whole process group; a no-op where that is unsupported."""
    if not hasattr(os, "killpg"):
        try:
            proc.kill()
        except Exception:
            pass
        return
    import signal
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _kill_container(name: str) -> None:
    try:
        subprocess.run(
            [DOCKER_BIN, "kill", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20,
        )
    except Exception:
        pass


@tool
def run_command(command: str, working_dir: str = ".", timeout_seconds: int = 120) -> str:
    """Run a shell command inside the workspace, e.g. 'npm install', 'npm run build',
    'node --check', or a smoke test. working_dir is relative to the workspace.
    timeout_seconds defaults to 120, maximum 300 (use large values only for slow
    installs or builds).

    SYSTEM GUARANTEE: the command runs inside an isolated container that can only see
    this session's workspace, and its whole process group is FORCE-KILLED on timeout
    AND when the command finishes — background processes never leak. To smoke-test a
    server, start it in the background with output redirected to a file, wait, then
    curl, all in ONE command. Example:
    'node backend/server.js > smoke.log 2>&1 & sleep 2; curl -s http://localhost:3000/api/health'
    Outbound network access is disabled by default (localhost still works).
    """
    container_name = None
    try:
        cwd = _safe_path(working_dir)
        timeout_seconds = max(5, min(int(timeout_seconds), 300))
        # start_new_session=True: perintah + semua anaknya masuk satu process group
        # baru, sehingga bisa dibunuh serentak (termasuk server yang di-background).
        # Ini tetap dipertahankan sebagai lapis kedua di atas isolasi container.
        if SANDBOX_COMMANDS:
            container_name = f"appbuilder-run-{uuid.uuid4().hex[:12]}"
            proc = subprocess.Popen(
                _runner_argv(container_name, command, cwd),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
        else:
            proc = subprocess.Popen(
                command, shell=True, cwd=cwd,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
            )
        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if container_name:
                _kill_container(container_name)
            _kill_process_group(proc)
            stdout, stderr = proc.communicate()
        finally:
            # Sapu bersih sisa proses background walau perintah utama selesai normal.
            if container_name:
                _kill_container(container_name)
            _kill_process_group(proc)

        out = (stdout or "") + ("\n" + stderr if stderr else "")
        if len(out) > 6000:
            out = out[:3000] + "\n... [truncated] ...\n" + out[-3000:]
        if timed_out:
            return failed(f"command exceeded the {timeout_seconds}s limit and its whole "
                          f"process group was killed.\nOutput so far:\n{out.strip()}")
        if proc.returncode == 0:
            return ok(f"command finished (exit code 0)\n{out.strip()}")
        return failed(f"command exited with code {proc.returncode}\n{out.strip()}")
    except FileNotFoundError as e:
        if SANDBOX_COMMANDS:
            return failed(
                f"the command sandbox needs Docker but '{DOCKER_BIN}' was not found. "
                f"Start the stack with docker compose, or set "
                f"ALLOW_UNSANDBOXED_COMMANDS=true to run commands directly on the host "
                f"(trusted local use only). Original error: {e}"
            )
        return failed(f"could not run command: {e}")
    except Exception as e:
        return failed(f"could not run command: {e}")


@tool
async def discuss_with(agent: str, message: str) -> str:
    """Talk DIRECTLY to another specialist agent and wait for their reply.
    agent: 'ba' | 'frontend' | 'backend' | 'qa' (never yourself).
    Use it to align work between specialists, e.g. the API contract between frontend and
    backend, or QA reporting a bug straight to the engineer who owns the code.
    message: a clear, specific message (name the file or endpoint being discussed).
    """
    # Note: this body never runs — discuss_with calls are handled specially in
    # _run_specialist so the sender's identity is known.
    return failed("discuss_with can only be used by specialist agents.")


BASE_TOOLS = {
    "write_code_file": write_code_file,
    "read_code_file": read_code_file,
    "list_files": list_files,
    "run_command": run_command,
    "discuss_with": discuss_with,
}

# Berapa lapis diskusi beruntun yang diizinkan (tugas PM = kedalaman 0).
# Mencegah dua agent saling memanggil tanpa henti.
MAX_DISCUSSION_DEPTH = 2

# Batas jumlah assign_task per satu permintaan user (di-reset tiap pesan user baru).
MAX_ASSIGN_TASKS_PER_REQUEST = int(os.environ.get("MAX_ASSIGN_TASKS", "12"))
_ASSIGN_BUDGET: dict[str, int] = {}


def reset_task_budget(thread_id: str) -> None:
    _ASSIGN_BUDGET[thread_id] = 0


def cleanup_session(thread_id: str) -> None:
    """Release all in-process state held for a finished session."""
    _SPECIALIST_HISTORY.pop(thread_id, None)
    _ASSIGN_BUDGET.pop(thread_id, None)
    forget_session(thread_id)
    METRICS.drop(thread_id)


# Zona kepemilikan file per agent, dicek DI LEVEL TOOL (bukan sekadar prompt).
# Path pertama = folder app; sisanya dicocokkan ke zona ini.
# Agent yang tidak terdaftar di sini (misal 'qa') sama sekali tidak boleh menulis.
_WRITE_ZONES: dict[str, tuple[str, ...]] = {
    "ba": ("docs/",),
    "frontend": ("frontend/", "README.md", ".env.example"),
    "backend": ("backend/", "docs/API_CONTRACT.md", "README.md", ".env.example"),
}


def _zone_error(agent_key: str, file_path: str) -> str | None:
    """None if the write is allowed; a FAILED result if the path is outside the agent's zone."""
    zones = _WRITE_ZONES.get(agent_key)
    if zones is None:
        return failed(f"{AGENTS[agent_key]['name']} is a pure reviewer and must not write "
                      f"files. Ask the engineer who owns the code via discuss_with.")
    parts = [p for p in file_path.replace("\\", "/").strip("/").split("/") if p]
    if len(parts) < 2:
        return failed("the file must live inside an app folder "
                      "(e.g. 'my-app/frontend/index.html'), not directly in the workspace root.")
    inner = "/".join(parts[1:])
    for zone in zones:
        if zone.endswith("/") and inner.startswith(zone):
            return None
        if inner == zone:
            return None
    allowed = ", ".join(f"<app>/{z}" for z in zones)
    return failed(f"path '{file_path}' is outside your ownership zone ({allowed}). "
                  f"Ask the owner of that path to change it via discuss_with.")


def summarize_args(name: str, args: dict) -> dict:
    """Structured description of a tool call for the UI.

    Returns a payload, not a sentence: the label is rendered client-side in the session
    language, so no server-side translation table is needed and presentation stays in the
    presentation layer.
    """
    args = args or {}
    if name in ("write_code_file", "read_code_file"):
        action = "write_file" if name == "write_code_file" else "read_file"
        return {"action": action, "target": args.get("file_path", "?")}
    if name == "list_files":
        return {"action": "list_files", "target": args.get("path", ".")}
    if name == "run_command":
        return {"action": "run_command", "target": args.get("command", "?")}
    if name == "ask_user":
        return {"action": "ask_user"}
    if name == "set_response_language":
        return {"action": "set_language", "target": args.get("language_code", "?")}
    if name in ("assign_task", "discuss_with"):
        key = args.get("agent", "")
        return {
            "action": "assign_task" if name == "assign_task" else "discuss_with",
            "target": key or "?",
            "target_name": AGENTS.get(key, {}).get("name", key or "?"),
        }
    return {"action": name}


def _text_of(content) -> str:
    """Normalize message content, which may be a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


# ==========================================
# LLM per agent (API base/key/model terpisah, dari app/config.py)
# ==========================================
def _make_llm(agent_key: str) -> ChatOpenAI:
    cfg = AGENTS[agent_key]
    return ChatOpenAI(
        openai_api_base=cfg["api_base"],
        openai_api_key=cfg["api_key"],
        model_name=cfg["model"],
        temperature=cfg["temperature"],
    )


# Dua versi binding per spesialis: lengkap, dan tanpa discuss_with (dipakai saat
# kedalaman diskusi sudah maksimal, agar rantai diskusi pasti berhenti).
_SPECIALIST_LLMS = {
    key: _make_llm(key).bind_tools([BASE_TOOLS[t] for t in AGENTS[key]["tools"]])
    for key in SPECIALISTS
}
_SPECIALIST_LLMS_NO_DISCUSS = {
    key: _make_llm(key).bind_tools(
        [BASE_TOOLS[t] for t in AGENTS[key]["tools"] if t != "discuss_with"]
    )
    for key in SPECIALISTS
}

# Memori percakapan tiap spesialis, per sesi (thread_id) — agar konteks antar tugas nyambung
_SPECIALIST_HISTORY: dict[str, dict[str, list]] = {}

MAX_SPECIALIST_STEPS = 40

# Batas jumlah pesan yang disimpan dalam history satu spesialis. History menumpuk lintas
# assign_task dalam satu sesi; tanpa batas, biaya token tumbuh tanpa henti.
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "60"))


def _trim_history(history: list) -> list:
    """Keep the most recent MAX_HISTORY_MESSAGES messages without orphaning tool plumbing.

    A ToolMessage must stay attached to the AIMessage.tool_calls that produced it. If the
    cut lands on an orphaned ToolMessage (its AIMessage was dropped), advance the boundary
    past those leading ToolMessages so the window starts on a clean pair boundary. The
    result therefore never begins with a ToolMessage and never contains an AIMessage whose
    tool_calls lost their replies.
    """
    if len(history) <= MAX_HISTORY_MESSAGES:
        return history
    start = len(history) - MAX_HISTORY_MESSAGES
    while start < len(history) and isinstance(history[start], ToolMessage):
        start += 1
    return history[start:]


# Base backoff delay (seconds). Overridable via env and monkeypatchable in tests so the
# suite stays fast; actual waits are _RETRY_BASE_DELAY * 2**attempt plus jitter.
_RETRY_BASE_DELAY = float(os.environ.get("LLM_RETRY_BASE_DELAY", "1.0"))


async def _invoke_with_retry(llm, messages, *, attempts: int = 3, agent_key: str = "pm",
                             writer=None, thread_id: str = "default", model: str = ""):
    """Invoke an LLM, retrying transient failures with exponential backoff + jitter.

    Retries on any exception up to `attempts` times (delays 1s, 2s, 4s * base plus jitter),
    emitting a `retry` stream event before each wait. Re-raises after the final attempt so
    the caller can surface a real error rather than silently degrading.

    Every attempt is instrumented: tokens, cost, latency, and retry count land in METRICS
    under (thread_id, agent_key).
    """
    for attempt in range(attempts):
        started = time.monotonic()
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            if attempt == attempts - 1:
                raise
            METRICS.record_retry(thread_id, agent_key)
            delay = _RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, _RETRY_BASE_DELAY)
            if writer is not None:
                try:
                    writer({
                        "agent": agent_key, "type": "retry",
                        "content": (f"LLM call failed ({e}); retrying in {delay:.1f}s "
                                    f"(attempt {attempt + 2}/{attempts})"),
                    })
                except Exception:
                    pass
            await asyncio.sleep(delay)
            continue

        METRICS.record_llm_call(
            thread_id, agent_key, model, response, time.monotonic() - started
        )
        _emit_metrics(writer, thread_id)
        return response


def _emit_metrics(writer, thread_id: str) -> None:
    """Push running session totals to the UI. Never allowed to break a run."""
    if writer is None:
        return
    try:
        writer({"type": "metrics", "totals": METRICS.totals(thread_id)})
    except Exception:
        pass


async def _run_discussion(sender_key: str, tc_args: dict, thread_id: str, chain: tuple) -> str:
    """Handle a discuss_with call made by a specialist.

    chain = the agents currently active on the call stack (excluding the sender). Used to
    reject cycles: A -> B -> A.
    """
    target = (tc_args.get("agent") or "").strip().lower()
    message = tc_args.get("message") or ""
    if target not in SPECIALISTS:
        return failed(f"unknown agent '{target}'. Choose one of: {', '.join(SPECIALISTS)}.")
    if target == sender_key:
        return failed("you cannot start a discussion with yourself.")
    if target in chain:
        return failed(f"{AGENTS[target]['name']} is your caller in this discussion chain — "
                      f"answer in your reply instead of starting a discussion back.")
    if len(chain) + 1 >= MAX_DISCUSSION_DEPTH:
        return failed("maximum discussion depth reached. Finish your task and raise the "
                      "rest in your report to the PM.")
    reply = await _run_specialist(
        target, message, thread_id, source=sender_key, chain=chain + (sender_key,)
    )
    return f"💬 Reply from {AGENTS[target]['name']}:\n{reply}"


async def _run_specialist(
    agent_key: str, content: str, thread_id: str, *, source: str = "pm", chain: tuple = ()
) -> str:
    """Run one specialist agent's work loop until it produces a final answer.

    source="pm"  : an official task from the Project Manager (assign_task).
    source=<key> : a discussion message from another specialist (discuss_with).
    chain        : agents waiting on the call stack (used by the cycle guard).
    """
    cfg = AGENTS[agent_key]
    # Di kedalaman maksimal, agent penerima tidak diberi tool discuss_with
    # sehingga rantai diskusi pasti berhenti.
    if len(chain) >= MAX_DISCUSSION_DEPTH:
        llm = _SPECIALIST_LLMS_NO_DISCUSS[agent_key]
    else:
        llm = _SPECIALIST_LLMS[agent_key]
    writer = get_stream_writer()

    bucket = _SPECIALIST_HISTORY.setdefault(thread_id, {})
    history = bucket.setdefault(agent_key, [])
    if source == "pm":
        history.append(HumanMessage(content=f"📥 Task from the Project Manager:\n{content}"))
        writer({"agent": agent_key, "type": "task", "content": content})
    else:
        sender_name = AGENTS[source]["name"]
        history.append(HumanMessage(
            content=f"💬 Discussion message from {sender_name} "
                    f"(answer directly and to the point):\n{content}"
        ))
        writer({"agent": agent_key, "type": "peer_in", "from": source, "content": content})

    for _ in range(MAX_SPECIALIST_STEPS):
        # Bound token growth: trim before each turn, keeping the shared bucket in sync so
        # later appends land on the same (trimmed) list.
        history = _trim_history(history)
        bucket[agent_key] = history
        # Language is appended per invocation, never written back into AGENTS[key]["prompt"]:
        # that dict is shared module state and mutating it would leak this session's
        # language into every other session.
        messages = [
            SystemMessage(
                content=cfg["prompt"] + language_directive(effective_language(thread_id))
            )
        ] + history
        response = await _invoke_with_retry(
            llm, messages, agent_key=agent_key, writer=writer,
            thread_id=thread_id, model=cfg["model"],
        )
        history.append(response)

        text = _text_of(response.content)
        if text:
            writer({"agent": agent_key, "type": "msg", "content": text})

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # Tidak ada tool call lagi = jawaban akhir (laporan ke PM / balasan diskusi)
            return text or f"STATUS: {STATUS_PARTIAL}\n(finished without a report)"

        for tc in tool_calls:
            METRICS.record_tool_call(thread_id, agent_key)
            writer({
                "agent": agent_key, "type": "tool_call",
                "detail": summarize_args(tc["name"], tc.get("args") or {}),
            })

            if tc["name"] == "discuss_with":
                result = await _run_discussion(agent_key, tc.get("args") or {}, thread_id, chain)
                succeeded = not is_failure(result)
                history.append(ToolMessage(
                    content=result[:8000], tool_call_id=tc["id"], name=tc["name"]
                ))
                # Untuk UI, buang baris prefix "Balasan dari ..." karena label bubble
                # sudah menyebutkan pengirimnya.
                shown = result.split("\n", 1)[1] if succeeded and "\n" in result else result
                writer({
                    "agent": agent_key,
                    "type": "peer_reply" if succeeded else "tool_result",
                    "name": tc["name"],
                    "from": (tc.get("args") or {}).get("agent", ""),
                    "ok": succeeded,
                    "content": shown[:2000],
                })
                continue

            # Validasi zona kepemilikan path DI LEVEL TOOL (aturan prompt saja bisa bocor).
            zone_err = None
            if tc["name"] == "write_code_file":
                zone_err = _zone_error(agent_key, (tc.get("args") or {}).get("file_path", ""))

            tool_fn = BASE_TOOLS.get(tc["name"])
            if zone_err is not None:
                result = zone_err
            elif tool_fn is None:
                result = failed(f"tool '{tc['name']}' is not available to you.")
            else:
                try:
                    result = await tool_fn.ainvoke(tc.get("args") or {})
                except Exception as e:
                    result = failed(str(e))
            result = str(result)
            history.append(ToolMessage(
                content=result[:8000], tool_call_id=tc["id"], name=tc["name"]
            ))
            writer({
                "agent": agent_key, "type": "tool_result",
                "name": tc["name"],
                "ok": not is_failure(result),
                "content": result[:400],
            })

    return (f"STATUS: {STATUS_PARTIAL}\nThe specialist stopped after reaching the step "
            f"limit ({MAX_SPECIALIST_STEPS}); the work may be incomplete.")


# ==========================================
# Tools khusus PM
# ==========================================
@tool
async def assign_task(agent: str, task: str, config: RunnableConfig) -> str:
    """Delegate a task to one specialist agent and WAIT for their final report.
    agent must be one of: 'ba' (Business Analyst), 'frontend' (Frontend Engineer),
    'backend' (Backend Engineer), 'qa' (Quality Assurance).
    task: a clear, complete task description (name the app folder and the spec files).
    """
    if agent not in SPECIALISTS:
        return failed(f"unknown agent '{agent}'. Choose one of: {', '.join(SPECIALISTS)}.")
    thread_id = (config.get("configurable") or {}).get("thread_id", "default")

    used = _ASSIGN_BUDGET.get(thread_id, 0) + 1
    _ASSIGN_BUDGET[thread_id] = used
    if used > MAX_ASSIGN_TASKS_PER_REQUEST:
        return failed(f"the budget of {MAX_ASSIGN_TASKS_PER_REQUEST} delegations for this "
                      f"request is exhausted. STOP delegating — summarize the real state of "
                      f"the work for the user (what is done and what is not).")

    report = await _run_specialist(agent, task, thread_id)
    return f"📤 Report from {AGENTS[agent]['name']}:\n{report}"


@tool
def ask_user(question: str) -> str:
    """Ask the user a question and WAIT for the answer. Execution pauses until they reply.
    Use this ONLY when a decision genuinely needs user input. Call this tool ALONE, never
    alongside other tool calls.
    """
    answer = interrupt(question)
    return str(answer)


@tool
def set_response_language(language_code: str, config: RunnableConfig) -> str:
    """Pin the language used for every human-facing reply, report, and generated document.
    Use ONLY when the user explicitly asks for a language ("reply in English",
    "pakai bahasa Indonesia saja"). Otherwise the language follows what the user writes.
    language_code: ISO 639-1 code, e.g. 'en', 'id', 'ja'.
    """
    code = (language_code or "").strip().lower()
    if not code or not code.replace("-", "").isalpha() or len(code) > 7:
        return failed(f"'{language_code}' is not a valid ISO 639-1 language code.")
    thread_id = (config.get("configurable") or {}).get("thread_id", "default")
    pin_language(thread_id, code)
    return ok(f"response language pinned to {language_name(code)} ({code}).")


PM_TOOLS = [assign_task, ask_user, list_files, read_code_file, set_response_language]

_pm_llm = _make_llm("pm").bind_tools(PM_TOOLS)


# ==========================================
# Graph utama (PM)
# ==========================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


async def project_manager(state: State, config: RunnableConfig):
    thread_id = (config.get("configurable") or {}).get("thread_id", "default")
    messages = [
        SystemMessage(
            content=AGENTS["pm"]["prompt"] + language_directive(effective_language(thread_id))
        )
    ] + state["messages"]
    try:
        writer = get_stream_writer()
    except Exception:
        writer = None
    response = await _invoke_with_retry(
        _pm_llm, messages, agent_key="pm", writer=writer,
        thread_id=thread_id, model=AGENTS["pm"]["model"],
    )
    for _ in getattr(response, "tool_calls", None) or []:
        METRICS.record_tool_call(thread_id, "pm")
    return {"messages": [response]}


def route_tools(state: State):
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END


def build_graph():
    workflow = StateGraph(State)
    workflow.add_node("pm", project_manager)
    workflow.add_node("tools", ToolNode(PM_TOOLS))
    workflow.add_edge(START, "pm")
    workflow.add_conditional_edges("pm", route_tools, {"tools": "tools", END: END})
    workflow.add_edge("tools", "pm")
    # MemorySaver wajib ada agar interrupt (ask_user) bisa pause & resume
    return workflow.compile(checkpointer=MemorySaver())
