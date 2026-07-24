# IMPROVEMENTS — Executable Task List

Engineering backlog for the `appbuilder` repository, ordered by priority.
Each task is self-contained: context, target files, required change, and acceptance criteria.

**Execution rules**

- Work through tasks in order. P0 before P1 before P2.
- One task = one commit. Use the commit message provided in each task.
- Do not refactor code outside the listed target files unless a task says so.
- Language convention: **code, comments, identifiers, commit messages, protocol tokens,
  and file names are always English.** Agent system prompts stay Indonesian until TASK-15,
  which makes human-facing output follow the user's language.
- After each task, verify the acceptance criteria before moving on.
- Never commit real secrets. `.env` must stay gitignored.

---

## P0 — Repository credibility

These determine whether a reviewer keeps reading. Do these first.

### TASK-01 — Remove the legacy prototype

**Context.** `main.py` at the repository root is an obsolete single-agent prototype with a
hardcoded API key and an architecture superseded by `app/`. The README instructs users to
run it, so anyone following the README never reaches the real system. It is the first file
a reviewer opens and it misrepresents the project.

**Files.** `main.py` (delete), `README.md` (replaced separately in TASK-02)

**Change.**
1. Delete `main.py`.
2. Confirm nothing imports from it: `grep -rn "main" --include="*.py" .` should return no
   import of the deleted module.

**Acceptance criteria.**
- `main.py` no longer exists.
- `python -m app.server` still starts successfully.
- No hardcoded API key string remains anywhere outside `app/config.py`.

**Commit.** `Remove obsolete single-agent prototype`

---

### TASK-02 — Replace the README

**Context.** The current README is titled `# crewai-env`, describes no architecture, and
gives the wrong run command. CrewAI is not used anywhere — this is LangGraph.

**Files.** `README.md`

**Change.** Replace the entire file with the provided `README.md`. Then verify each claim
against the source and correct any drift:

- Run command matches how `app/server.py` actually starts (`python -m app.server`).
- Port matches `uvicorn.run(...)` in `app/server.py`.
- Environment variable names match `app/config.py` and `app/agent.py`.
- The agent/tool table matches the `AGENTS` dict and `_WRITE_ZONES`.

**Acceptance criteria.**
- Following the README from a clean clone produces a running server.
- No reference to CrewAI remains.
- Known limitations section is present and accurate.

**Commit.** `Rewrite README with architecture, design decisions, and limitations`

---

### TASK-03 — Externalize configuration

**Context.** The API key is hardcoded five times in `app/config.py`. Even for a local
endpoint, a duplicated literal credential in a public repo reads as a finding to any
reviewer, and duplication makes provider changes error-prone.

**Files.** `app/config.py`, `.env.example` (new), `requirements.txt`, `.gitignore`

**Change.**
1. Add `python-dotenv` to `requirements.txt`. Load `.env` at the top of `app/config.py`
   with `load_dotenv()` before reading any variables.
2. Introduce shared fallbacks read from the environment:
   - `AGENTS_API_BASE` (default `http://localhost:20128/v1`)
   - `AGENTS_API_KEY` (no default)
   - `AGENTS_MODEL` (no default)
   - `AGENTS_TEMPERATURE` (default `0.1`)
3. Each entry in `AGENTS` uses those shared values as its default, keeping the existing
   per-agent `<PREFIX>_*` override behavior unchanged.
4. If `AGENTS_API_KEY` resolves empty for any agent, raise a `RuntimeError` at import time
   with a message naming the missing variable. Fail loudly, not at first request.
5. Create `.env.example` documenting every supported variable with placeholder values and
   a commented example of per-agent routing.
6. Confirm `.env` and `.env.*` are gitignored (they already are — verify).

**Acceptance criteria.**
- No credential literal remains in any tracked file.
- Starting the server without `AGENTS_API_KEY` fails immediately with a clear message.
- Setting only `AGENTS_*` variables configures all five agents.
- Setting `QA_MODEL` overrides the QA agent alone.

**Commit.** `Move agent credentials and model config to environment variables`

---

### TASK-04 — Add a license

**Files.** `LICENSE` (new)

**Change.** Add a standard MIT license file. Copyright holder: the repository owner.

**Acceptance criteria.** GitHub detects and displays the license.

**Commit.** `Add MIT license`

---

## P1 — Correctness and resource safety

Real bugs. A reviewer who reads the source will find these.

### TASK-05 — Fix the specialist history memory leak

**Context.** In `app/agent.py`, `_SPECIALIST_HISTORY` is a module-level dict keyed by
`thread_id`. Every WebSocket connection generates a fresh UUID `thread_id`, and nothing
ever removes the entry. A long-running server grows without bound. `_ASSIGN_BUDGET` has
the same problem.

**Files.** `app/agent.py`, `app/server.py`

**Change.**
1. In `app/agent.py`, add a public function:
   ```python
   def cleanup_session(thread_id: str) -> None:
       """Release all in-process state held for a finished session."""
   ```
   It must remove the `thread_id` key from both `_SPECIALIST_HISTORY` and `_ASSIGN_BUDGET`,
   tolerating missing keys.
2. In `app/server.py`, call `cleanup_session(thread_id)` from a `finally` block wrapping
   the WebSocket handler loop, so it runs on both normal disconnect and exception.

**Acceptance criteria.**
- After a client connects and disconnects, neither dict retains that `thread_id`.
- Add a test asserting this (see TASK-09).

**Commit.** `Release per-session agent state on disconnect`

---

### TASK-06 — Bound specialist context growth

**Context.** `MAX_SPECIALIST_STEPS` caps iterations inside a single `_run_specialist`
call, but `history` persists and accumulates across every `assign_task` in a session.
Token cost grows linearly and long sessions will eventually exceed the context window.

**Files.** `app/agent.py`

**Change.**
1. Add a module constant `MAX_HISTORY_MESSAGES`, configurable via the `MAX_HISTORY_MESSAGES`
   environment variable, default `60`.
2. Add a helper `_trim_history(history: list) -> list` that, when the list exceeds the
   limit, keeps the most recent `MAX_HISTORY_MESSAGES` entries.
3. **Critical:** trimming must not orphan tool-call plumbing. An `AIMessage` containing
   `tool_calls` must never be separated from its corresponding `ToolMessage` replies, or
   the provider will reject the request. If the trim boundary would split such a pair,
   move the boundary earlier until it lands on a clean boundary.
4. Call `_trim_history` at the start of each iteration of the specialist loop, before
   building the `messages` list.

**Acceptance criteria.**
- A synthetic history of 200 messages trims to at most 60.
- No trimmed result begins with a `ToolMessage`.
- No `AIMessage` with `tool_calls` survives without its matching `ToolMessage`.
- Covered by tests in TASK-09.

**Commit.** `Trim specialist conversation history to bound token growth`

---

### TASK-07 — Isolate workspaces per session

**Context.** `WORKSPACE` is a single module-level directory shared by every session.
Two concurrent sessions generating an app with the same name overwrite each other.

**Files.** `app/agent.py`, `app/server.py`

**Change.**
1. Add a `contextvars.ContextVar[str]` named `_SESSION_ROOT`, defaulting to the base
   `WORKSPACE`. A ContextVar is required rather than a parameter because the value must
   propagate implicitly through async LangGraph tool execution.
2. Add `set_session_workspace(thread_id: str) -> str` that creates
   `WORKSPACE/<thread_id>/`, sets the ContextVar, and returns the path.
3. Change `_safe_path()` to jail against `_SESSION_ROOT.get()` instead of the global
   `WORKSPACE`. Traversal rejection logic stays identical.
4. In `app/server.py`, call `set_session_workspace(thread_id)` once when the WebSocket
   connection is accepted.
5. Update the `/api/files` endpoint to list the current session's directory rather than
   the global workspace root.

**Acceptance criteria.**
- Two concurrent sessions writing `my-app/backend/server.js` produce two independent files.
- Path traversal is still rejected (`../../etc/passwd`, absolute paths, `..\\` on Windows).
- The file tree panel shows only the current session's files.

**Commit.** `Isolate generated app workspaces per session`

---

### TASK-08 — Retry transient LLM failures

**Context.** `llm.ainvoke` in `_run_specialist` and `project_manager` has no error
handling. A single transient provider error (429, 500, timeout) aborts the entire run and
the user sees a raw exception.

**Files.** `app/agent.py`

**Change.**
1. Add an async helper `_invoke_with_retry(llm, messages, *, attempts: int = 3)`.
2. Retry on exception with exponential backoff (`1s`, `2s`, `4s`) plus jitter.
3. After the final failed attempt, re-raise so the caller can surface a real error.
4. Use it for both the specialist loop and the `project_manager` node.
5. Emit a stream event on each retry so the UI can show that a retry is happening — reuse
   the existing writer channel with `{"agent": <key>, "type": "retry", "content": ...}`.

**Acceptance criteria.**
- A mocked LLM failing twice then succeeding completes the run without error.
- A mocked LLM failing three times raises, and the WebSocket handler reports it cleanly
  rather than crashing the connection.
- Backoff delays are patchable in tests so the suite stays fast.

**Commit.** `Add retry with exponential backoff for LLM calls`

---

## P2 — Verification

The security boundaries in this project are currently untested. This is the highest
value-per-effort work in the backlog.

### TASK-09 — Test suite for guardrails

**Context.** `_safe_path`, `_zone_error`, and the discussion cycle/depth guards are the
system's actual security and correctness boundaries. They are pure or near-pure functions
and are trivially testable. They currently have zero tests.

**Files.** `tests/__init__.py`, `tests/test_safe_path.py`, `tests/test_write_zones.py`,
`tests/test_discussion_guard.py`, `tests/test_session_state.py`, `requirements-dev.txt`

**Change.** Add `pytest` and `pytest-asyncio` to `requirements-dev.txt`, then write:

**`tests/test_safe_path.py`**
- Accepts a normal relative path.
- Accepts a nested relative path.
- Rejects `../` traversal.
- Rejects deep traversal (`a/../../../etc/passwd`).
- Rejects an absolute path escaping the workspace.
- Rejects backslash traversal (`..\\..\\windows`).
- Treats a leading `/` as workspace-relative, not filesystem-absolute.
- The returned path always resolves inside the workspace root.

**`tests/test_write_zones.py`**
- `ba` may write `my-app/docs/SPEC.md`.
- `ba` may **not** write `my-app/backend/server.js`.
- `frontend` may write `my-app/frontend/App.jsx` and `my-app/README.md`.
- `frontend` may **not** write `my-app/docs/API_CONTRACT.md`.
- `backend` **may** write `my-app/docs/API_CONTRACT.md` (sole owner).
- `qa` may write **nothing** — assert the error message identifies QA as a pure reviewer.
- A path at the workspace root with no app folder is rejected.
- A zone prefix must not match by substring (`my-app/frontend-old/x.js` is rejected).

**`tests/test_discussion_guard.py`** (async)
- An unknown target agent key is rejected.
- An agent addressing itself is rejected.
- A target already in the call chain is rejected (cycle guard).
- Exceeding `MAX_DISCUSSION_DEPTH` is rejected.
- At maximum depth, `_run_specialist` selects the binding **without** `discuss_with` —
  assert against `_SPECIALIST_LLMS_NO_DISCUSS`, since this is the structural guarantee.
- The `assign_task` budget rejects the call once `MAX_ASSIGN_TASKS_PER_REQUEST` is
  exceeded, and `reset_task_budget` restores it.

**`tests/test_session_state.py`**
- `cleanup_session` removes both history and budget entries (TASK-05).
- `_trim_history` respects the cap and never orphans a tool-call pair (TASK-06).

Mock all LLM calls. The suite must not require a live API endpoint and must run in under
ten seconds.

**Acceptance criteria.**
- `pytest` passes from a clean clone with no API key set.
- Every test above exists and asserts real behavior, not just "does not raise".

**Commit.** `Add test suite for path, ownership, and discussion guardrails`

---

### TASK-10 — Continuous integration

**Files.** `.github/workflows/ci.yml` (new)

**Change.** GitHub Actions workflow triggered on push and pull request:
1. Matrix over Python 3.10, 3.11, 3.12.
2. Install `requirements.txt` and `requirements-dev.txt`.
3. Run `ruff check .` (add `ruff` to dev requirements).
4. Run `pytest`.

Add the resulting status badge to the top of `README.md`.

**Acceptance criteria.** The workflow passes on the default branch and the badge renders.

**Commit.** `Add CI workflow for lint and tests`

---

## P3 — Differentiation

This is what turns the repository from a working demo into evidence of engineering
judgment. Per-agent model routing is currently a *capability* with no *measurement*.

### TASK-11 — Instrument cost and latency

**Context.** Each agent can run a different model, which makes cost optimization possible.
Nothing currently records tokens, cost, or latency, so the benefit cannot be demonstrated.

**Files.** `app/metrics.py` (new), `app/agent.py`, `app/server.py`, `app/static/index.html`

**Change.**
1. Create `app/metrics.py` with a `RunMetrics` collector holding, per `thread_id` and per
   agent: prompt tokens, completion tokens, LLM call count, tool call count, retry count,
   cumulative wall-clock latency.
2. Read token counts from the `usage_metadata` / `response_metadata` on LangChain
   responses. Handle their absence gracefully — not every provider returns usage.
3. Add an optional `MODEL_PRICING` map in `app/config.py` (model name → input and output
   cost per 1M tokens). Compute estimated cost when a model is present in the map.
4. Record metrics in `_run_specialist`, `_invoke_with_retry`, and the `project_manager` node.
5. Persist one JSON Lines record per completed user request to `runs/<thread_id>.jsonl`.
   Gitignore `runs/`.
6. Add `GET /api/metrics/{thread_id}` returning the current session summary.
7. Add a footer bar in the UI showing live totals: tokens, estimated cost, elapsed time.

**Acceptance criteria.**
- After one full build request, `runs/<thread_id>.jsonl` contains a per-agent breakdown.
- The UI shows a non-zero token total during execution.
- A provider that omits usage metadata does not crash the run.

**Commit.** `Add per-agent token, cost, and latency instrumentation`

---

### TASK-12 — Benchmark harness

**Context.** With TASK-11 in place, the system can answer questions that most multi-agent
projects cannot: how much does a build actually cost, how consistent is it across runs,
and where does the multi-agent split help versus hurt.

**Files.** `benchmark/run_benchmark.py` (new), `benchmark/tasks.yaml` (new),
`benchmark/README.md` (new)

**Change.**
1. `benchmark/tasks.yaml` defines a set of build prompts with expected artifacts
   (for example: an app folder must contain `docs/SPEC.md`, `backend/`, and `frontend/`).
2. `run_benchmark.py` executes each task `N` times against the graph, capturing per run:
   success/failure, wall-clock time, total tokens, estimated cost, number of delegations,
   number of QA repair rounds, and final QA status.
3. Output a summary table plus variance (standard deviation) across runs.
4. Support `--config` to point at an alternative per-agent model configuration, so two
   routing strategies can be compared directly.
5. `benchmark/README.md` documents how to run it and how to interpret the output.

**Acceptance criteria.**
- `python -m benchmark.run_benchmark --runs 3` completes and prints a summary table.
- Results are written to a timestamped file for later comparison.

**Commit.** `Add benchmark harness for cost and consistency measurement`

---

### TASK-13 — Containerize command execution

**Context.** `run_command` executes arbitrary shell commands. `working_dir` is path-jailed
but the command string is not — an agent can read files outside the workspace or reach the
network. This is documented as a limitation in the README; this task removes it.

**Files.** `Dockerfile` (new), `docker-compose.yml` (new), `app/agent.py`, `README.md`

**Change.**
1. `Dockerfile` for the application, and a separate minimal runner image containing only
   Node.js and Python.
2. Change `run_command` to execute inside the runner container with the session workspace
   bind-mounted, no network access by default, a memory limit, and a non-root user.
3. Keep the existing process-group kill behavior as a second layer of defense.
4. Make containerized execution the default with an explicit
   `ALLOW_UNSANDBOXED_COMMANDS=true` escape hatch for local development.
5. Update the README limitations section to reflect the new posture.

**Acceptance criteria.**
- A command attempting to read a file outside the workspace fails.
- A command attempting outbound network access fails when networking is disabled.
- Timeout still force-kills the container.
- Existing QA smoke-test workflow (background server plus curl) still works inside the
  container.

**Commit.** `Execute agent shell commands in an isolated container`

---

## P1b — Language handling

> **Ordering note.** TASK-14 changes string constants that TASK-09 tests assert against.
> Execute TASK-14 **before** TASK-09, or budget time to update those assertions.

### TASK-14 — Separate machine-readable protocol from human-facing prose

**Context.** Tool results currently begin with Indonesian tokens: `write_code_file` returns
`"SUKSES: File tersimpan di ..."` and failures return `"GAGAL: ..."`. Both `app/agent.py`
and `app/server.py` decide the UI success indicator with `result.startswith("GAGAL")`.
The specialist report contract is likewise Indonesian: `STATUS: SELESAI | PARSIAL | BLOKIR`.

These are **protocol tokens**, not prose — the system branches on them. Once agent output
becomes language-dependent (TASK-15), any token that drifts with the response language
breaks silently: a failed tool call would render as a success, and the PM would read a
`BLOCKED` report as complete. Normalize the protocol before making prose variable.

**Files.** `app/protocol.py` (new), `app/agent.py`, `app/config.py`, `app/server.py`

**Change.**
1. Create `app/protocol.py` containing the single source of truth:
   - Result tokens `RESULT_OK = "OK"` and `RESULT_FAILED = "FAILED"`.
   - Status tokens `STATUS_DONE = "DONE"`, `STATUS_PARTIAL = "PARTIAL"`,
     `STATUS_BLOCKED = "BLOCKED"`.
   - Builders `ok(detail: str) -> str` and `failed(detail: str) -> str` returning
     `f"{TOKEN}: {detail}"`.
   - A predicate `is_failure(result: str) -> bool`. **No other module may inspect the
     prefix directly.**
2. Replace every `"SUKSES..."` / `"GAGAL..."` literal in `app/agent.py` with `ok()` /
   `failed()`. This covers `write_code_file`, `read_code_file`, `list_files`,
   `run_command`, `discuss_with`, `assign_task`, `_zone_error`, and `_safe_path`.
3. Replace every `startswith("GAGAL")` in `app/agent.py` and `app/server.py` with
   `is_failure(...)`.
4. Update the report contract in `_GENERAL_RULES` (`app/config.py`) to the English tokens,
   and update the PM prompt so it reads `PARTIAL` / `BLOCKED` rather than the Indonesian
   equivalents.
5. Document the rule explicitly in the prompts: **protocol tokens are always English
   regardless of the response language.** Only the prose after the colon is localizable.

**Acceptance criteria.**
- `git grep -n "SUKSES\|GAGAL\|SELESAI\|PARSIAL\|BLOKIR" app/` returns nothing.
- `is_failure` is the only function that inspects a result prefix.
- The UI still marks failed tool calls as failures.
- A specialist report beginning `STATUS: BLOCKED` is not treated as completed work.

**Commit.** `Normalize tool result and report status tokens to a stable protocol`

---

### TASK-15 — Mirror the user's language in all human-facing output

**Context.** `_GENERAL_RULES` in `app/config.py` hardcodes Indonesian for every report and
document. A user who writes in English receives Indonesian reports, Indonesian
`SPEC.md`, and an Indonesian UI. Response language should follow the language the user
actually writes in.

**Files.** `app/language.py` (new), `app/config.py`, `app/agent.py`, `app/server.py`,
`app/static/index.html`, `requirements.txt`

**Change.**

**1. Detection module — `app/language.py`**

- `detect_language(text: str) -> str | None` returning an ISO 639-1 code.
  Use `langdetect` with `DetectorFactory.seed = 0` so results are deterministic; add
  `langdetect` to `requirements.txt`.
- Return `None` when the input is too short or the detection confidence is low
  (suggested floor: fewer than 12 characters, or probability below `0.85`). Short
  follow-ups like `ok`, `lanjut`, `yes`, `thanks` must **not** produce a guess.
- `_SESSION_LANGUAGE: ContextVar[str]` — same propagation pattern as the workspace
  ContextVar in TASK-07.
- `resolve_session_language(text: str) -> str` with **sticky** semantics: set on the first
  confident detection and changed only by another confident detection that differs.
  Ambiguous input leaves the current value untouched. Initial default comes from the
  `DEFAULT_RESPONSE_LANGUAGE` environment variable (default `id`).
- `language_name(code: str) -> str` mapping at minimum `id`, `en`, `ms`, `ja`, `zh`, `ar`,
  `es`, `fr`, `de`, falling back to the raw code.
- `language_directive(code: str) -> str` returning a short English instruction naming the
  target language explicitly, for appending to a system prompt.

**2. Injection — `app/agent.py`**

Append `language_directive(...)` to the system prompt at invocation time, in both
`project_manager` and `_run_specialist`:

```python
messages = [SystemMessage(content=cfg["prompt"] + language_directive(current_language()))] + history
```

**Do not mutate `AGENTS[key]["prompt"]`.** That dict is module-level shared state; mutating
it would leak one session's language into every other session.

**3. Session wiring — `app/server.py`**

- Call `resolve_session_language(text)` when a user message arrives, before invoking the
  graph.
- Emit a `{"type": "language", "code": "<iso>"}` WebSocket event so the client can
  localize its own chrome.

**4. Prompt rules — `app/config.py`**

Replace the current Indonesian language rule in `_GENERAL_RULES` with a three-way split:

| Category | Language |
|---|---|
| Chat replies, specialist reports, and generated documents (`SPEC.md`, `API_CONTRACT.md`, app `README.md`) | **Session language** |
| Code — identifiers, function names, comments, log strings | **Always English** |
| Protocol tokens (`STATUS:`, `OK:`, `FAILED:`) and **all file and folder names** | **Always English** |

The file-name rule is load-bearing: `_WRITE_ZONES` matches literal path prefixes such as
`docs/` and `docs/API_CONTRACT.md`. A localized filename would fail the ownership check
and the agent would be unable to write its own spec.

**5. Explicit override**

Add a `set_response_language(language_code: str)` tool to `PM_TOOLS` so an explicit user
request ("reply in English", "pakai bahasa Indonesia saja") pins the language regardless
of detection. It writes through the same ContextVar and takes precedence over detection
for the remainder of the session.

**6. UI localization — `app/static/index.html`**

- Move hardcoded Indonesian interface strings into a keyed dictionary with `id` and `en`
  entries, switched by the `language` event. Fall back to `en` for unmapped languages.
- `summarize_args()` in `app/agent.py` currently returns Indonesian display labels
  (`"menulis file: ..."`, `"berdiskusi dengan ..."`). Change it to return a structured
  payload — `{"action": "write_file", "target": "..."}` — and render the label on the
  client. This removes the need for a server-side translation table and keeps
  presentation concerns in the presentation layer.

**Acceptance criteria.**
- An English request produces: English PM replies, English specialist reports, English
  `SPEC.md` and `README.md`, and English UI labels.
- An Indonesian request produces all of the above in Indonesian.
- In **both** cases, code identifiers and comments are English.
- In **both** cases, generated paths are identical — `docs/SPEC.md`, never
  `docs/SPESIFIKASI.md`.
- In **both** cases, `STATUS:` values remain `DONE` / `PARTIAL` / `BLOCKED`.
- A short ambiguous follow-up (`ok`, `lanjut`, `next`) does not flip the session language.
- After a completed run, `AGENTS["pm"]["prompt"]` is byte-identical to its value at import
  time — assert this in a test.
- Two concurrent sessions in different languages do not interfere.

**Extend TASK-09 with:**
- `detect_language` returns `None` for short or ambiguous input.
- `resolve_session_language` stickiness: confident → ambiguous → confident-different.
- `language_directive` injection does not mutate shared `AGENTS` state.
- Protocol tokens survive a simulated non-Indonesian, non-English session.

**Commit.** `Mirror user language across agent responses, documents, and UI`

---

## Verification checklist

Run before declaring the backlog complete.

- [ ] Clean clone, follow the README, server starts on the documented port
- [ ] Server refuses to start without `AGENTS_API_KEY` and says which variable is missing
- [ ] `pytest` passes with no API key configured
- [ ] `ruff check .` is clean
- [ ] CI is green on the default branch
- [ ] No credential literal in any tracked file (`git grep -i "sk-"` returns nothing)
- [ ] Two concurrent browser sessions do not share a workspace
- [ ] Connect then disconnect leaves no residual entry in the session state dicts
- [ ] A full build produces a metrics record under `runs/`
- [ ] An English request yields fully English output; an Indonesian request yields fully
      Indonesian output
- [ ] Code identifiers and generated file paths are English in both cases
- [ ] `git grep -n "SUKSES\|GAGAL\|SELESAI\|PARSIAL\|BLOKIR" app/` returns nothing
- [ ] README limitations match the code as it now stands
