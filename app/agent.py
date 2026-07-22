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

import os
import subprocess
from typing import Annotated

from typing_extensions import TypedDict

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

from app.config import AGENTS, SPECIALISTS

# ==========================================
# Workspace: semua app hasil buatan tim masuk ke sini
# ==========================================
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.environ.get("PM_WORKSPACE", os.path.join(_BASE_DIR, "workspace"))
os.makedirs(WORKSPACE, exist_ok=True)


def _safe_path(path: str) -> str:
    """Kunci semua operasi file agar tidak bisa keluar dari folder WORKSPACE."""
    cleaned = path.lstrip("/\\")
    full = os.path.abspath(os.path.join(WORKSPACE, cleaned))
    if not (full == WORKSPACE or full.startswith(WORKSPACE + os.sep)):
        raise ValueError(f"Akses ditolak: path '{path}' berada di luar workspace.")
    return full


# ==========================================
# Tools dasar (dipakai lintas agent sesuai config)
# ==========================================
@tool
def write_code_file(file_path: str, content: str) -> str:
    """Membuat atau menimpa satu file. Folder dibuat otomatis jika belum ada.
    file_path bersifat RELATIF terhadap workspace, contoh: 'toko-online/src/App.jsx'.
    """
    try:
        full = _safe_path(file_path)
        directory = os.path.dirname(full)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"SUKSES: File tersimpan di {os.path.relpath(full, WORKSPACE)}"
    except Exception as e:
        return f"GAGAL menulis file: {e}"


@tool
def read_code_file(file_path: str) -> str:
    """Membaca isi sebuah file di dalam workspace (path relatif terhadap workspace)."""
    try:
        full = _safe_path(file_path)
        with open(full, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 12000:
            content = content[:12000] + "\n... [terpotong]"
        return content
    except Exception as e:
        return f"GAGAL membaca file: {e}"


@tool
def list_files(path: str = ".") -> str:
    """Melihat struktur file/folder di dalam workspace (path relatif, default root workspace)."""
    try:
        full = _safe_path(path)
        if not os.path.exists(full):
            return f"Folder '{path}' belum ada. Workspace masih kosong di lokasi itu."
        lines = []
        for root, dirs, files in os.walk(full):
            dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "dist", "__pycache__")]
            rel_root = os.path.relpath(root, WORKSPACE)
            depth = 0 if rel_root == "." else rel_root.count(os.sep) + 1
            if depth > 4:
                dirs[:] = []
                continue
            indent = "  " * depth
            lines.append(f"{indent}{os.path.basename(root) if rel_root != '.' else '.'}/")
            for fname in sorted(files):
                lines.append(f"{indent}  {fname}")
            if len(lines) > 200:
                lines.append("... [terpotong]")
                break
        return "\n".join(lines)
    except Exception as e:
        return f"GAGAL melihat struktur: {e}"


@tool
def run_command(command: str, working_dir: str = ".") -> str:
    """Menjalankan perintah shell di dalam workspace, misal 'npm install' atau 'npm run build'.
    working_dir relatif terhadap workspace. Timeout 5 menit.
    JANGAN gunakan untuk perintah yang berjalan selamanya (npm run dev / server).
    """
    try:
        cwd = _safe_path(working_dir)
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=300,
        )
        out = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        if len(out) > 6000:
            out = out[:3000] + "\n... [terpotong] ...\n" + out[-3000:]
        status = "SUKSES" if result.returncode == 0 else f"GAGAL (exit code {result.returncode})"
        return f"{status}\n{out.strip()}"
    except subprocess.TimeoutExpired:
        return "GAGAL: Perintah melebihi batas waktu 5 menit."
    except Exception as e:
        return f"GAGAL menjalankan perintah: {e}"


@tool
async def discuss_with(agent: str, message: str) -> str:
    """Berdiskusi LANGSUNG dengan agent spesialis lain dan menunggu balasannya.
    agent: 'ba' | 'frontend' | 'backend' | 'qa' (tidak boleh dirimu sendiri).
    Gunakan untuk menyelaraskan pekerjaan antar spesialis, misal kontrak API antara
    frontend & backend, atau QA memberitahu bug langsung ke engineer terkait.
    message: pesan yang jelas dan spesifik (sebut file/endpoint yang dibahas).
    """
    # Catatan: tool ini tidak pernah dieksekusi lewat sini — pemanggilannya
    # ditangani khusus di _run_specialist agar identitas pengirim diketahui.
    return "GAGAL: discuss_with hanya bisa dipakai oleh agent spesialis."


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


def summarize_args(name: str, args: dict) -> str:
    """Ringkasan tool call yang enak dibaca di UI."""
    if name == "write_code_file":
        return f"menulis file: {args.get('file_path', '?')}"
    if name == "read_code_file":
        return f"membaca file: {args.get('file_path', '?')}"
    if name == "list_files":
        return f"melihat struktur: {args.get('path', '.')}"
    if name == "run_command":
        return f"menjalankan: {args.get('command', '?')}"
    if name == "ask_user":
        return "bertanya ke user"
    if name == "assign_task":
        target = AGENTS.get(args.get("agent", ""), {}).get("name", args.get("agent", "?"))
        return f"menugaskan {target}"
    if name == "discuss_with":
        target = AGENTS.get(args.get("agent", ""), {}).get("name", args.get("agent", "?"))
        return f"berdiskusi dengan {target}"
    return name


def _text_of(content) -> str:
    """Normalisasi content message (bisa string atau list of blocks)."""
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


async def _run_discussion(sender_key: str, tc_args: dict, thread_id: str, depth: int) -> str:
    """Tangani pemanggilan discuss_with oleh seorang spesialis."""
    target = (tc_args.get("agent") or "").strip().lower()
    message = tc_args.get("message") or ""
    if target not in SPECIALISTS:
        return f"GAGAL: agent '{target}' tidak dikenal. Pilihan: {', '.join(SPECIALISTS)}."
    if target == sender_key:
        return "GAGAL: kamu tidak bisa berdiskusi dengan dirimu sendiri."
    if depth + 1 > MAX_DISCUSSION_DEPTH:
        return "GAGAL: batas kedalaman diskusi tercapai. Selesaikan tugasmu dan sampaikan sisanya di laporan ke PM."
    reply = await _run_specialist(target, message, thread_id, source=sender_key, depth=depth + 1)
    return f"💬 Balasan dari {AGENTS[target]['name']}:\n{reply}"


async def _run_specialist(
    agent_key: str, content: str, thread_id: str, *, source: str = "pm", depth: int = 0
) -> str:
    """Jalankan loop kerja satu agent spesialis sampai ia memberikan jawaban akhir.

    source="pm"  : tugas resmi dari Project Manager (assign_task).
    source=<key> : pesan diskusi dari spesialis lain (discuss_with).
    """
    cfg = AGENTS[agent_key]
    # Saat kedalaman diskusi maksimal, agent penerima tidak diberi tool discuss_with
    # sehingga rantai diskusi pasti berhenti.
    if depth >= MAX_DISCUSSION_DEPTH:
        llm = _SPECIALIST_LLMS_NO_DISCUSS[agent_key]
    else:
        llm = _SPECIALIST_LLMS[agent_key]
    writer = get_stream_writer()

    history = _SPECIALIST_HISTORY.setdefault(thread_id, {}).setdefault(agent_key, [])
    if source == "pm":
        history.append(HumanMessage(content=f"📥 Tugas dari Project Manager:\n{content}"))
        writer({"agent": agent_key, "type": "task", "content": content})
    else:
        sender_name = AGENTS[source]["name"]
        history.append(HumanMessage(
            content=f"💬 Pesan diskusi dari {sender_name} (jawab langsung ke intinya):\n{content}"
        ))
        writer({"agent": agent_key, "type": "peer_in", "from": source, "content": content})

    for _ in range(MAX_SPECIALIST_STEPS):
        messages = [SystemMessage(content=cfg["prompt"])] + history
        response = await llm.ainvoke(messages)
        history.append(response)

        text = _text_of(response.content)
        if text:
            writer({"agent": agent_key, "type": "msg", "content": text})

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            # Tidak ada tool call lagi = jawaban akhir (laporan ke PM / balasan diskusi)
            return text or "(selesai tanpa laporan)"

        for tc in tool_calls:
            writer({
                "agent": agent_key, "type": "tool_call",
                "detail": summarize_args(tc["name"], tc.get("args") or {}),
            })

            if tc["name"] == "discuss_with":
                result = await _run_discussion(agent_key, tc.get("args") or {}, thread_id, depth)
                ok = not result.startswith("GAGAL")
                history.append(ToolMessage(content=result[:8000], tool_call_id=tc["id"], name=tc["name"]))
                # Untuk UI, buang baris prefix "Balasan dari ..." karena label bubble
                # sudah menyebutkan pengirimnya.
                shown = result.split("\n", 1)[1] if ok and "\n" in result else result
                writer({
                    "agent": agent_key,
                    "type": "peer_reply" if ok else "tool_result",
                    "name": tc["name"],
                    "from": (tc.get("args") or {}).get("agent", ""),
                    "ok": ok,
                    "content": shown[:2000],
                })
                continue

            tool_fn = BASE_TOOLS.get(tc["name"])
            if tool_fn is None:
                result = f"GAGAL: tool '{tc['name']}' tidak tersedia untukmu."
            else:
                try:
                    result = await tool_fn.ainvoke(tc.get("args") or {})
                except Exception as e:
                    result = f"GAGAL: {e}"
            result = str(result)
            history.append(ToolMessage(content=result[:8000], tool_call_id=tc["id"], name=tc["name"]))
            writer({
                "agent": agent_key, "type": "tool_result",
                "name": tc["name"],
                "ok": not result.startswith("GAGAL"),
                "content": result[:400],
            })

    return "PERINGATAN: spesialis berhenti karena mencapai batas iterasi. Hasil mungkin belum lengkap."


# ==========================================
# Tools khusus PM
# ==========================================
@tool
async def assign_task(agent: str, task: str, config: RunnableConfig) -> str:
    """Mendelegasikan tugas ke satu agent spesialis dan MENUNGGU laporan akhirnya.
    agent harus salah satu dari: 'ba' (Business Analyst), 'frontend' (Frontend Engineer),
    'backend' (Backend Engineer), 'qa' (Quality Assurance).
    task: deskripsi tugas yang jelas dan lengkap (sebut folder app & file spec bila ada).
    """
    if agent not in SPECIALISTS:
        return f"GAGAL: agent '{agent}' tidak dikenal. Pilihan: {', '.join(SPECIALISTS)}."
    thread_id = (config.get("configurable") or {}).get("thread_id", "default")
    report = await _run_specialist(agent, task, thread_id)
    return f"📤 Laporan dari {AGENTS[agent]['name']}:\n{report}"


@tool
def ask_user(question: str) -> str:
    """Bertanya kepada user dan MENUNGGU jawabannya. Eksekusi berhenti sampai user menjawab.
    Gunakan HANYA jika keputusan benar-benar butuh input user. Panggil tool ini SENDIRIAN,
    jangan bersamaan dengan tool lain.
    """
    answer = interrupt(question)
    return str(answer)


PM_TOOLS = [assign_task, ask_user, list_files, read_code_file]

_pm_llm = _make_llm("pm").bind_tools(PM_TOOLS)


# ==========================================
# Graph utama (PM)
# ==========================================
class State(TypedDict):
    messages: Annotated[list, add_messages]


async def project_manager(state: State):
    messages = [SystemMessage(content=AGENTS["pm"]["prompt"])] + state["messages"]
    response = await _pm_llm.ainvoke(messages)
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
