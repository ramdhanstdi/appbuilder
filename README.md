# AppBuilder — Multi-Agent Development Team

[![CI](https://github.com/ramdhanstdi/appbuilder/actions/workflows/ci.yml/badge.svg)](https://github.com/ramdhanstdi/appbuilder/actions/workflows/ci.yml)

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
| An agent runs `npm run dev` and hangs the orchestrator forever | Every shell command runs in a disposable container and its own process group, force-killed on completion **and** on timeout |
| An agent's shell command reads your SSH keys | Commands execute in a container that mounts only the session workspace, with no network and no root |
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
        │   + AsyncSqliteSaver      │◄──── interrupt() / Command(resume=…)
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
| 🧑‍💼 `pm` | Project Manager — the only agent the user talks to. Writes no code. | none | `assign_task`, `ask_user`, `list_files`, `read_code_file`, `set_response_language` |
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

Every specialist report must begin with `STATUS: DONE | PARTIAL | BLOCKED`, so the PM
cannot mistake partial work for completion. Those tokens — and the `OK:` / `FAILED:`
prefixes on tool results — are **protocol, not prose**: they live in `app/protocol.py`,
stay English in every response language, and `is_failure()` is the only function allowed
to inspect a result prefix.

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

### 2. Shell execution is contained twice over

Agent commands run inside a one-shot container: only the session workspace is mounted, so
a command cannot read the host filesystem; networking is off by default (loopback still
works, so QA smoke tests do); memory, CPU, and pids are capped; and it runs as a non-root
user with `no-new-privileges`. The command string reaches Docker as a single argv element,
so nothing inside it can rewrite the invocation around it.

The process-group kill stays as the second layer. `run_command` starts every command in a
new session (`start_new_session=True`), so the command and all of its children share one
process group, and that group is `SIGKILL`ed on timeout **and** in a `finally` block after
normal completion — after the container itself is killed.

The second kill is the one people forget: a command that "succeeds" can still leave a
backgrounded server running. Here it cannot.

`ALLOW_UNSANDBOXED_COMMANDS=true` drops back to direct host execution for local
development without Docker. That is a real removal of the boundary, not a nicety — see
[Known limitations](#known-limitations).

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

### 4. Per-agent model routing — and the measurement to justify it

Every agent carries its own `api_base`, `api_key`, `model`, and `temperature`. Any agent
can run on a different provider entirely. This makes cost engineering possible: an
expensive reasoning model for the BA where architecture decisions are made, a cheap fast
model for mechanical review passes, without touching a line of orchestration code.

Routing without measurement is just a knob, so `app/metrics.py` records what each agent
actually consumed — prompt and completion tokens, LLM calls, tool calls, retries,
cumulative latency, and estimated cost from the `MODEL_PRICING` table in `app/config.py`.
Totals stream to a footer bar in the UI while the run is in flight, one JSON Lines record
per completed user request lands in `runs/<thread_id>.jsonl`, and
`GET /api/metrics/{thread_id}` returns the live per-agent breakdown.

Usage metadata is read from `usage_metadata` with a fallback to `response_metadata`;
a provider that reports neither contributes zero tokens rather than failing the run.
`benchmark/` (below) turns those records into a comparison between routing strategies.

### 5. Human-in-the-loop uses real checkpointing — and sessions are durable

`ask_user` calls LangGraph's `interrupt()`, backed by a checkpointer. The graph genuinely
suspends mid-execution and resumes with `Command(resume=answer)` when the next message
arrives. It is not a blocking `input()` call in disguise.

That checkpointer is a persistent **AsyncSqliteSaver**, so a session outlives the browser
tab and the server process. Each session has one stable `thread_id` (kept in the browser's
`localStorage`) that ties together three durable layers under `sessions/<thread_id>/`:

- the **PM graph checkpoint** — the Project Manager's own conversation;
- **`specialists.json`** — each specialist's message history, so the team remembers what it
  already built and decided;
- **`events.jsonl`** — every message and activity the browser was shown.

On reconnect the server replays `events.jsonl` verbatim, restores the specialist histories,
and continues from the checkpoint — so a user can close the app and pick up later without
losing any chat from the PM or the team. A session switcher in the header (`＋ New project`
and the dropdown, backed by `GET /api/sessions`) lists every saved project.

### 6. Full observability of nested agents

The server consumes two stream modes at once:

- `updates` — the PM graph's own steps
- `custom` — specialist steps, emitted from *inside* `assign_task` via `get_stream_writer()`

Without the custom channel, everything a specialist does would be invisible until its
final report. With it, each agent gets its own live tab in the browser.

### 7. The response language follows the user; the protocol does not

The system speaks the language the user writes in — chat replies, specialist reports, and
the contents of `SPEC.md`, `API_CONTRACT.md`, and the generated `README.md`. Detection is
deterministic (`langdetect` with a fixed seed) and **sticky**: it changes only on another
confident detection, so a short follow-up like `ok`, `lanjut`, or `next` can never flip a
session's language mid-build.

Three things stay English no matter what:

| Category | Language |
|---|---|
| Chat replies, reports, generated document contents | Session language |
| Code — identifiers, function names, comments, log strings | Always English |
| Protocol tokens (`STATUS:`, `OK:`, `FAILED:`) and **all file and folder names** | Always English |

The file-name rule is load-bearing rather than stylistic: `_WRITE_ZONES` matches literal
path prefixes like `docs/` and `docs/API_CONTRACT.md`. A translated filename would fail
the ownership check and the agent could not write its own spec.

The language is injected per invocation — `cfg["prompt"] + language_directive(...)` — and
never written back into the shared `AGENTS` dict, so two concurrent sessions in different
languages cannot leak into each other. `set_response_language` lets the user pin a
language explicitly, which then outranks detection for the rest of the session.

### 8. Filesystem sandbox

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
docker compose build     # builds the app image and the minimal runner image
docker compose up
```

Open <http://localhost:8020>.

Running with Docker is the default posture because agent shell commands execute in an
isolated runner container. To run the server directly on the host instead:

```bash
python -m app.server
```

That still uses containerized commands and therefore needs a reachable Docker daemon plus
the runner image (`docker build -f Dockerfile.runner -t appbuilder-runner:latest .`). With
no Docker at all, set `ALLOW_UNSANDBOXED_COMMANDS=true` — read the limitations first.

Generated applications appear under `workspace/<session-id>/`.

### Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The suite covers the guardrails — the filesystem jail, write-ownership zones, discussion
cycle/depth limits, the delegation budget, history trimming, session cleanup, and language
handling. Every LLM call is mocked: it needs no API key and no live endpoint, and runs in
about a second. CI runs the same two commands on Python 3.10, 3.11, and 3.12.

### Benchmarks

```bash
python -m benchmark.run_benchmark --runs 3
python -m benchmark.run_benchmark --runs 3 --config benchmark/configs/cheap-qa.yaml
```

Runs the same build tasks N times against the real graph and prints cost, tokens,
delegations, QA repair rounds, and the standard deviation across runs — so two per-agent
routing strategies can be compared on evidence rather than intuition. Unlike the test
suite this makes real LLM calls and costs real money. See [benchmark/README.md](benchmark/README.md).

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
| `MODEL_PRICING_JSON` | Price list override for cost estimation (JSON) | built-in table |
| `RUNS_DIR` | Where per-request metrics records are written | `./runs` |
| `MAX_HISTORY_MESSAGES` | Messages kept in a specialist's history before trimming | `60` |
| `DEFAULT_RESPONSE_LANGUAGE` | Response language before detection (ISO 639-1) | `id` |
| `SESSIONS_DIR` | Where durable session state (checkpoint, history, event log) lives | `./sessions` |
| `CHECKPOINT_DB` | SQLite file backing the PM graph checkpointer | `./sessions/checkpoints.sqlite` |
| `ALLOW_UNSANDBOXED_COMMANDS` | Run agent commands directly on the host instead of in a container | unset (sandboxed) |
| `RUNNER_IMAGE` | Image used to execute agent commands | `appbuilder-runner:latest` |
| `RUNNER_NETWORK` | Docker network for the runner (`bridge` to allow installs) | `none` |
| `RUNNER_MEMORY` / `RUNNER_CPUS` | Resource caps for the runner | `512m` / `2` |
| `RUNNER_USER` | uid:gid the command runs as | `1000:1000` |
| `RUNNER_VOLUMES_FROM` | Reuse this container's mounts (set by docker-compose) | unset |

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
│   ├── protocol.py     # OK/FAILED + DONE/PARTIAL/BLOCKED tokens (single source of truth)
│   ├── language.py     # response-language detection, stickiness, prompt directive
│   ├── metrics.py      # per-agent tokens, cost, latency; JSONL run records
│   ├── session_store.py# durable sessions: event log, meta, specialist history
│   ├── server.py       # FastAPI + WebSocket streaming
│   └── static/
│       └── index.html  # single-file UI: per-agent tabs, file tree, metrics bar
├── benchmark/          # cost & consistency harness (tasks.yaml, run_benchmark.py)
├── Dockerfile          # application image (server + orchestration)
├── Dockerfile.runner   # minimal image agent commands execute in
├── docker-compose.yml  # both images, shared workspace volume
├── tests/              # guardrail tests — no API key required
├── workspace/          # sandbox — generated apps live here (gitignored)
├── runs/               # per-request metrics records (gitignored)
├── sessions/           # durable per-session state for resume (gitignored)
├── requirements.txt
└── .env.example
```

---

## Known limitations

Stated plainly, because these matter more than the feature list.

**Agent shell commands are containerized, and the server is not.** `run_command` executes
in a one-shot runner container with only the session workspace mounted, no outbound
network, a memory cap, and a non-root user — so a command cannot read the host filesystem
or phone home. But launching those containers requires the Docker socket, which is a
host-level privilege granted to the *server* process. Isolation protects you from what the
agents do; it does not make the server itself safe to expose publicly.

**`ALLOW_UNSANDBOXED_COMMANDS=true` removes that boundary entirely.** It exists for local
development without Docker, and it restores the original posture: arbitrary shell
execution as your user, path-jailed `working_dir` but unjailed command string. Trusted
local use only.

**Installing packages requires opening the network.** The runner defaults to
`RUNNER_NETWORK=none`, which keeps loopback (so QA smoke tests still work) but blocks
outbound access — `npm install` fails until you set `RUNNER_NETWORK=bridge`. That is the
intended tradeoff, not an oversight.

**Sessions are durable, with one edge case.** The PM checkpoint, specialist histories,
and the chat event log are persisted per session, so closing the app and reopening resumes
the same project with its full history. The one soft spot is reconnecting at the *exact*
moment the PM is waiting on an `ask_user` question: `AsyncSqliteSaver` commits the paused
checkpoint on a background thread, and a disconnect can race that commit. The pending
question itself is stored durably (in `meta.json`) so it is always re-shown, but resuming
it may cost one extra answer round. Delegation budgets and the live metrics *timer* remain
in-process and reset on restart (cumulative token/cost totals are persisted).

**History trimming is a window, not a summary.** Old messages past
`MAX_HISTORY_MESSAGES` are dropped, not compressed, so a very long session forgets its
early context rather than paying for it.

**Cost estimates are only as good as the price table.** A model absent from
`MODEL_PRICING` still has its tokens counted but contributes `$0.00`, and providers that
omit usage metadata contribute zero tokens.

**Language detection can be wrong on borderline input.** Detection is deterministic and
sticky with a confidence floor, but a confident wrong guess (short mixed-language text
above the threshold) will switch the session until the next confident message.

---

## Roadmap

- [x] Per-session workspace isolation
- [x] Retry with exponential backoff on LLM calls
- [x] History trimming for long sessions
- [x] Test suite for the guardrails, plus CI on 3.10–3.12
- [x] Token/cost/latency instrumentation per agent, persisted per run
- [x] Benchmark harness: N identical tasks, measured variance and cost
- [x] Containerized command execution
- [x] Response language mirrors the user's
- [x] Persistent checkpointer (SQLite) + resumable sessions with full chat replay
- [ ] History summarization instead of a plain window
- [ ] Rootless/gVisor runner so the server no longer needs the Docker socket

---

## License

MIT
