# Benchmark harness

Runs the same build tasks N times against the real graph and measures what they cost.
The point is not a leaderboard — it is answering three questions this project can answer
and most multi-agent demos cannot:

1. **What does a build actually cost?** In tokens and dollars, per agent.
2. **How consistent is it?** Same prompt, N runs, standard deviation across them.
3. **Does a routing change pay for itself?** Two `--config` files, two tables, compared.

## Requirements

A working `.env` (the harness drives the real graph, so it makes real LLM calls and
costs real money). Cost figures only appear for models listed in `MODEL_PRICING` in
`app/config.py`; token counts are always recorded.

## Running

```bash
# baseline: every task 3 times with the configuration in .env
python -m benchmark.run_benchmark --runs 3

# a single task, once, while iterating
python -m benchmark.run_benchmark --runs 1 --task notes-api

# an alternative routing strategy, to compare against the baseline
python -m benchmark.run_benchmark --runs 3 --config benchmark/configs/cheap-qa.yaml
```

| Flag | Meaning | Default |
|---|---|---|
| `--runs` | Runs per task | `3` |
| `--task` | Only this task id; repeatable | all |
| `--tasks` | Alternative `tasks.yaml` | `benchmark/tasks.yaml` |
| `--config` | YAML of per-agent env overrides | none |
| `--timeout` | Seconds before a run is abandoned | `900` |

Every run gets its own `thread_id` and therefore its own isolated session workspace under
`workspace/`, so concurrent artifacts never collide. Whenever the PM calls `ask_user`, the
task's canned `answer` is supplied automatically — a benchmark run never blocks on a human.

## Defining tasks

`tasks.yaml` holds one entry per task:

```yaml
tasks:
  - id: notes-api
    prompt: Build a backend-only notes API in the folder 'notes-api' …
    answer: Use your best judgment and proceed. Express, in-memory storage, no frontend.
    expect:
      - notes-api/docs/SPEC.md
      - notes-api/backend/          # trailing '/' = directory must exist and be non-empty
```

`expect` is what makes a run pass or fail, so keep it objective: files that must exist and
be non-empty. Make `answer` decisive — a vague answer inflates variance for reasons that
have nothing to do with the routing you are measuring.

## Reading the output

```
Routing configuration: cheap-qa
task              runs   ok%     sec       ±        tokens     ±         cost$     ±         deleg   qa-fix
todo-app          3      100%    182.4     21.7     148302     9611      0.4821    0.0312    5.0     1.3
```

- **ok%** — fraction of runs where every expected artifact was produced and no error was raised.
- **sec / tokens / cost$** — means across runs; the `±` column next to each is the standard
  deviation. A large `±` means a single run tells you nothing, and any comparison between
  two configurations needs more runs before it means anything either.
- **deleg** — `assign_task` calls, i.e. how much delegation the PM needed.
- **qa-fix** — QA repair rounds: how many times QA sent an engineer back to fix something.
  Rising `qa-fix` with a cheaper engineer model is exactly the tradeoff worth seeing.

The full per-run detail, including the per-agent token and cost breakdown, is written to
`benchmark/results/<timestamp>-<config>.json` (gitignored) so two runs can be diffed later.
Per-request records also accumulate in `runs/<thread_id>.jsonl` as usual.

## Interpreting a comparison

Run the baseline and the alternative with the same `--runs` and compare the same task row.
A routing change is worth keeping when cost drops **and** `ok%` holds **and** `qa-fix` does
not climb — a cheap engineer model that doubles the repair rounds usually costs more in
total than the model it replaced, which is precisely the thing that is invisible without
this harness.
