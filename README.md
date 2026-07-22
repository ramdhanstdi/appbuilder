# AppBuilder — Multi-Agent Development Team

A LangGraph-based multi-agent system where a Project Manager agent delegates software
development work to four specialists (Business Analyst, Frontend, Backend, QA) that build
real applications inside a sandboxed workspace — streamed live to a browser UI.

The interesting part is not that the agents write code. It is **how they are constrained**:
every collaboration rule in this system is enforced at the runtime/tool layer, not by
asking the model nicely in a prompt.

---

## Table of contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Agent roles](#agent-roles)
- [Design decisions](#design-decisions)
- [Getting started](#getting-started)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## Why this exists

Most multi-agent demos fail in the same predictable ways:

| Failure mode | How this project handles it |
|---|---|
| An agent runs `npm run dev` and hangs the orchestrator forever | Every shell command runs in its own process group and is force-killed on completion **and** on timeout |
| The QA agent silently fixes its own findings and reports "PASS" | QA has **no write tool at all** — it is a structurally pure reviewer |
| Two agents negotiate an API contract in chat, then both forget it | The contract is a **file** (`docs/API_CONTRACT.md`) with a single owning agent |
| Agent A asks B, B asks A, forever, burning tokens | Cycle detection + depth limit, enforced by **removing the tool from the model binding** |
| Agents overwrite each other's files | Per-agent write ownership zones validated before the write executes |
| The orchestrator loops indefinitely | Hard delegation budget per user request |

---

## Architecture

```
                                   ┌──────────────┐
                       WebSocket   │   Browser    │
                    ┌──────────────│  (tabs per   │
                    │              │    agent)    │
                    ▼              └──────────────┘
            ┌───────────────┐
            │   FastAPI     │  /ws  · /api/agents · /api/files
            │  app/server.py│
            └───────┬───────┘
                    │  stream_mode = ["updates", "custom"]
                    ▼
        ┌───────────────────────────┐
        │   LangGraph (PM graph)    │
        │   + MemorySaver           │◄──── interrupt() / Command(resume=…)
        │                           │      human-in-the-loop Q&A
        │   tools:                  │
        │    assign_task            │
        │    ask_user               │
        │    list_files             │
        │    read_code_file         │
        └────────────┬──────────────┘
                     │ assign_task(agent, task)
                     ▼
   ┌─────────────────────────────────────────────────┐
   │            Specialist agent loop                │
   │  (own API base / key / model / temperature)     │
   │                                                 │
   │   📊 BA        🎨 Frontend                      │
   │       ▲            ▲                            │
   │       │  discuss_with (peer-to-peer,            │
   │       ▼   depth-limited, cycle-guarded)         │
   │   ⚙️ Backend    🔍 QA                           │
   └─────────────────────┬───────────────────────────┘
                         │ write_code_file / run_command
                         ▼
              ┌──────────────────────┐
              │  workspace/          │  ← path-jailed sandbox
              │   my-app/            │
              │     docs/            │  ← BA owns
              │     frontend/        │  ← Frontend owns
              │     backend/         │  ← Backend owns
              └──────────────────────┘
```

### Communication topology

This is **not** a pure star topology. The PM delegates work, but specialists can talk to
each other laterally via `discuss_with` — frontend and backend negotiate their API
contract directly, and QA reports bugs straight to the engineer who owns the file.

That flexibility is what makes multi-agent systems useful and also what makes them
dangerous, so peer communication is bounded on three axes: depth, cycles, and topic count.

---

## Agent roles

| Agent | Role | Write access | Tools |
|---|---|---|---|
| 🧑‍💼 `pm` | Project Manager — the only agent the user talks to. Writes no code. | none | `assign_task`, `ask_user`, `list_files`, `read_code_file` |
| 📊 `ba` | Business Analyst / Tech Lead — owns the spec, the stack, and the API contract. | `<app>/docs/**` | `write_code_file`, `read_code_file`, `list_files`, `discuss_with` |
| 🎨 `frontend` | Frontend Engineer — builds the UI against the contract. | `<app>/frontend/**`, `README.md`, `.env.example` | + `run_command` |
| ⚙️ `backend` | Backend Engineer — owns `API_CONTRACT.md`; the only agent allowed to change it. | `<app>/backend/**`, `docs/API_CONTRACT.md`, `README.md`, `.env.example` | + `run_command` |
| 🔍 `qa` | Quality Assurance — verifies against acceptance criteria and smoke-tests real endpoints. | **none** | `read_code_file`, `list_files`, `run_command`, `discuss_with` |

### Standard workflow

1. PM clarifies ambiguity with the user via `ask_user` (execution pauses via `interrupt()`).
2. BA writes `<app>/docs/SPEC.md` with **objective acceptance criteria**, plus
   `<app>/docs/API_CONTRACT.md` when a backend exists.
3. Backend and Frontend implement — both are required to read the contract, and the
   frontend is forbidden from guessing endpoint shapes.
4. QA verifies criterion by criterion, smoke-tests live endpoints, and reports bugs to
   engineers directly. Maximum two repair rounds.
5. An engineer writes the app `README.md`.
6. PM reports back to the user.

Every specialist report must begin with `STATUS: SELESAI | PARSIAL | BLOKIR`, so the PM
cannot mistake partial work for completion.

---

## Design decisions

These are the choices worth reading the source for.

### 1. Guardrails are structural, not textual

When an agent reaches the maximum discussion depth, it is not *told* to stop. It is bound
to a different LLM instance that does not have the `discuss_with` tool at all:

```python
_SPECIALIST_LLMS_NO_DISCUSS = {
    key: _make_llm(key).bind_tools(
        [BASE_TOOLS[t] for t in AGENTS[key]["tools"] if t != "discuss_with"]
    )
    for key in SPECIALISTS
}
```

A prompt rule leaks eventually. A missing tool cannot be called.

The same principle applies to file ownership: `_zone_error()` runs *before*
`write_code_file` executes, and QA has no entry in `_WRITE_ZONES` — so "QA must not patch
its own findings" is a property of the system, not a request to the model.

### 2. Shell execution cannot leak processes

`run_command` starts every command in a new session (`start_new_session=True`), so the
command and all of its children share one process group. That group is `SIGKILL`ed on
timeout **and** in a `finally` block after normal completion.

The second kill is the one people forget: a command that "succeeds" can still leave a
backgrounded server running. Here it cannot.

This turns a constraint into a capability — QA can start a server, curl it, and read the
log in a single command, knowing cleanup is guaranteed:

```bash
node backend/server.js > qa-smoke.log 2>&1 & sleep 2; curl -s localhost:3000/api/health
```

So QA does real smoke testing, not just static analysis.

### 3. The API contract is an artifact, not a conversation

Each agent has its own context window. Anything agreed in a `discuss_with` exchange
evaporates when that exchange ends. So the contract lives in `docs/API_CONTRACT.md`,
owned exclusively by the backend agent. If the implementation must diverge, the backend
updates the file first, then notifies the frontend — the frontend can read it but never
write it.

### 4. Per-agent model routing

Every agent carries its own `api_base`, `api_key`, `model`, and `temperature`. Any agent
can run on a different provider entirely. This makes cost engineering possible: an
expensive reasoning model for the BA where architecture decisions are made, a cheap fast
model for mechanical review passes, without touching a line of orchestration code.

### 5. Human-in-the-loop uses real checkpointing

`ask_user` calls LangGraph's `interrupt()`, backed by a `MemorySaver` checkpointer. The
graph genuinely suspends mid-execution and resumes with `Command(resume=answer)` when the
next WebSocket message arrives. It is not a blocking `input()` call in disguise.

### 6. Full observability of nested agents

The server consumes two stream modes at once:

- `updates` — the PM graph's own steps
- `custom` — specialist steps, emitted from *inside* `assign_task` via `get_stream_writer()`

Without the custom channel, everything a specialist does would be invisible until its
final report. With it, each agent gets its own live tab in the browser.

### 7. Filesystem sandbox

`_safe_path()` normalizes and jails every file operation to `WORKSPACE`, rejecting
traversal attempts (`../`, absolute paths) before any I/O happens.

---

## Getting started

### Requirements

- Python 3.10+
- An OpenAI-compatible API endpoint (local gateway, or any hosted provider)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# edit .env with your API base, key, and model
```

### Run

```bash
python -m app.server
```

Open <http://localhost:8020>.

Generated applications appear under `workspace/`.

---

## Configuration

All agents fall back to shared defaults and can be overridden individually through
environment variables:

| Variable | Description | Default |
|---|---|---|
| `AGENTS_API_BASE` | OpenAI-compatible base URL | `http://localhost:20128/v1` |
| `AGENTS_API_KEY` | API key | — (required) |
| `AGENTS_MODEL` | Model name | — |
| `AGENTS_TEMPERATURE` | Sampling temperature | `0.1` |
| `PM_WORKSPACE` | Workspace root directory | `./workspace` |
| `MAX_ASSIGN_TASKS` | Delegation budget per user request | `12` |

Per-agent overrides use the agent's prefix — `PM_`, `BA_`, `FRONTEND_`, `BACKEND_`, `QA_`:

```bash
# Cheap model for QA, strong model for architecture
QA_MODEL=gpt-4o-mini
BA_MODEL=claude-sonnet-4-6
BA_API_BASE=https://api.anthropic.com/v1
```

Adding a new agent means adding one entry to the `AGENTS` dict in `app/config.py` —
`SPECIALISTS`, the collaboration rules, the LLM bindings, and the UI tabs all derive from it.

---

## Project structure

```
appbuilder/
├── app/
│   ├── agent.py        # tools, guardrails, specialist loop, PM graph
│   ├── config.py       # per-agent config + system prompts
│   ├── server.py       # FastAPI + WebSocket streaming
│   └── static/
│       └── index.html  # single-file UI: per-agent tabs, file tree
├── workspace/          # sandbox — generated apps live here (gitignored)
├── requirements.txt
└── .env.example
```

---

## Known limitations

Stated plainly, because these matter more than the feature list.

**`run_command` is arbitrary shell execution.** The `working_dir` argument is path-jailed,
but the command string itself is not sandboxed — an agent can reach outside the workspace.
**This tool is intended for trusted local use only.** Do not expose the server to a
network or run it against untrusted prompts without containerizing the workspace first.

**In-process state.** Conversation checkpoints (`MemorySaver`), specialist histories, and
delegation budgets live in process memory. Restarting the server loses all sessions.

**Single shared workspace.** All sessions write to the same directory, so concurrent
sessions building apps with the same name will collide.

**Unbounded specialist history.** Specialist context accumulates across tasks within a
session with no trimming or summarization, so cost grows over long sessions.

**No retry on transient provider errors.** A single failed LLM call aborts the current
specialist run.

**No cost or token instrumentation yet.** Per-agent model routing makes cost optimization
*possible*; it is not yet *measured*.

---

## Roadmap

- [ ] Token/cost/latency instrumentation per agent, persisted per run
- [ ] Containerized workspace execution
- [ ] Per-session workspace isolation
- [ ] Retry with exponential backoff on LLM calls
- [ ] History trimming / summarization for long sessions
- [ ] Persistent checkpointer (SQLite/Postgres)
- [ ] Benchmark harness: N identical tasks, measured variance and cost

---

## License

MIT
