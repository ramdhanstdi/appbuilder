"""Benchmark harness: run the same build tasks N times and measure what they cost.

With app/metrics.py in place the system can answer questions most multi-agent projects
cannot — what does a build actually cost, how consistent is it run to run, and does a
given per-agent routing strategy pay for itself.

    python -m benchmark.run_benchmark --runs 3
    python -m benchmark.run_benchmark --runs 3 --config benchmark/configs/cheap-qa.yaml

`--config` applies environment overrides *before* the app is imported (agent config is
resolved at import time), so two routing strategies can be compared directly.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone

import yaml

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TASKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks.yaml")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# How many times a run may be resumed past an ask_user interrupt before giving up.
MAX_RESUMES = 12


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_config(path: str | None) -> str:
    """Apply a routing configuration to the environment. Must run before importing app."""
    if not path:
        return "default"
    config = load_yaml(path)
    for name, value in (config.get("env") or {}).items():
        os.environ[str(name)] = str(value)
    return str(config.get("name") or os.path.basename(path))


def artifacts_present(root: str, expected: list[str]) -> tuple[bool, list[str]]:
    """Check the artifacts a successful run must produce. Returns (ok, missing)."""
    missing = []
    for item in expected or []:
        full = os.path.join(root, *[p for p in item.replace("\\", "/").split("/") if p])
        if item.rstrip().endswith("/"):
            present = os.path.isdir(full) and any(os.scandir(full))
        else:
            present = os.path.isfile(full) and os.path.getsize(full) > 0
        if not present:
            missing.append(item)
    return (not missing), missing


async def run_once(task: dict, run_index: int, timeout: float) -> dict:
    """Execute one task once against the graph and return its measurements."""
    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    from app.agent import (
        build_graph,
        cleanup_session,
        reset_task_budget,
        set_session_workspace,
    )
    from app.language import start_session
    from app.metrics import METRICS
    from app.protocol import report_status

    graph = _GRAPH or build_graph()
    task_id = task.get("id", "task")
    thread_id = f"bench-{task_id}-{run_index}-{uuid.uuid4().hex[:6]}"
    root = set_session_workspace(thread_id)
    start_session(thread_id)
    reset_task_budget(thread_id)
    METRICS.start_request(thread_id, task.get("prompt", ""))

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
    answer = task.get("answer") or "Use your best judgment and proceed."

    counters = {"delegations": 0, "qa_repair_rounds": 0, "questions": 0, "tool_failures": 0}
    qa_status = None
    error = None
    started = time.monotonic()

    async def drive():
        nonlocal qa_status
        graph_input = {"messages": [HumanMessage(content=task.get("prompt", ""))]}
        for _ in range(MAX_RESUMES + 1):
            interrupted = False
            async for mode, chunk in graph.astream(
                graph_input, config, stream_mode=["updates", "custom"]
            ):
                if mode == "custom":
                    _count_custom(chunk, counters)
                    if chunk.get("agent") == "qa" and chunk.get("type") == "msg":
                        status = report_status(chunk.get("content", ""))
                        if status:
                            qa_status = status
                    continue
                if "__interrupt__" in chunk:
                    counters["questions"] += 1
                    interrupted = True
                    continue
                if "pm" in chunk:
                    message = chunk["pm"]["messages"][-1]
                    for call in getattr(message, "tool_calls", None) or []:
                        if call["name"] == "assign_task":
                            counters["delegations"] += 1
            if not interrupted:
                return
            graph_input = Command(resume=answer)
        raise RuntimeError(f"run kept asking questions after {MAX_RESUMES} answers")

    try:
        await asyncio.wait_for(drive(), timeout=timeout)
    except Exception as exc:  # a failed run is a data point, not a crash
        error = f"{type(exc).__name__}: {exc}"

    duration = time.monotonic() - started
    record = METRICS.finish_request(thread_id) or {}
    totals = record.get("totals", {})
    ok, missing = artifacts_present(root, task.get("expect") or [])

    result = {
        "task": task_id,
        "run": run_index,
        "thread_id": thread_id,
        "success": ok and error is None,
        "error": error,
        "missing_artifacts": missing,
        "duration_seconds": round(duration, 2),
        "total_tokens": totals.get("total_tokens", 0),
        "prompt_tokens": totals.get("prompt_tokens", 0),
        "completion_tokens": totals.get("completion_tokens", 0),
        "cost_usd": totals.get("cost_usd", 0.0),
        "llm_calls": totals.get("llm_calls", 0),
        "retries": totals.get("retries", 0),
        "delegations": counters["delegations"],
        "qa_repair_rounds": counters["qa_repair_rounds"],
        "questions": counters["questions"],
        "tool_failures": counters["tool_failures"],
        "qa_status": qa_status,
        "agents": record.get("agents", {}),
    }
    cleanup_session(thread_id)
    return result


def _count_custom(chunk: dict, counters: dict) -> None:
    event = chunk.get("type")
    # A QA repair round is QA telling an engineer to fix something, which arrives at the
    # engineer as an inbound discussion message from 'qa'.
    if event == "peer_in" and chunk.get("from") == "qa":
        counters["qa_repair_rounds"] += 1
    elif event == "tool_result" and chunk.get("ok") is False:
        counters["tool_failures"] += 1


_GRAPH = None


def summarize(results: list[dict]) -> list[dict]:
    """Per-task aggregates with standard deviation across runs."""
    rows = []
    by_task: dict[str, list[dict]] = {}
    for result in results:
        by_task.setdefault(result["task"], []).append(result)

    for task_id, runs in by_task.items():
        row = {"task": task_id, "runs": len(runs),
               "success_rate": sum(1 for r in runs if r["success"]) / len(runs)}
        for field in ("duration_seconds", "total_tokens", "cost_usd", "delegations",
                      "qa_repair_rounds", "retries"):
            values = [r[field] for r in runs]
            row[f"{field}_mean"] = round(statistics.fmean(values), 4)
            row[f"{field}_stdev"] = round(
                statistics.stdev(values) if len(values) > 1 else 0.0, 4
            )
        row["qa_statuses"] = [r["qa_status"] for r in runs]
        rows.append(row)
    return rows


def print_table(rows: list[dict], config_name: str) -> None:
    headers = ["task", "runs", "ok%", "sec", "±", "tokens", "±", "cost$", "±", "deleg", "qa-fix"]
    widths = [16, 5, 6, 8, 7, 9, 8, 9, 8, 6, 7]
    print(f"\nRouting configuration: {config_name}")
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True)))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        cells = [
            row["task"],
            str(row["runs"]),
            f"{row['success_rate'] * 100:.0f}%",
            f"{row['duration_seconds_mean']:.1f}",
            f"{row['duration_seconds_stdev']:.1f}",
            f"{row['total_tokens_mean']:.0f}",
            f"{row['total_tokens_stdev']:.0f}",
            f"{row['cost_usd_mean']:.4f}",
            f"{row['cost_usd_stdev']:.4f}",
            f"{row['delegations_mean']:.1f}",
            f"{row['qa_repair_rounds_mean']:.1f}",
        ]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths, strict=True)))
    print(
        "\n'±' is the standard deviation across runs — the number that says whether a "
        "single run means anything."
    )


def write_results(payload: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(RESULTS_DIR, f"{stamp}-{payload['config']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


async def main_async(args, config_name: str) -> int:
    global _GRAPH
    from app.agent import build_graph

    _GRAPH = build_graph()

    tasks = load_yaml(args.tasks).get("tasks") or []
    if args.task:
        tasks = [t for t in tasks if t.get("id") in args.task]
    if not tasks:
        print("No tasks selected.", file=sys.stderr)
        return 1

    results = []
    for task in tasks:
        for index in range(1, args.runs + 1):
            label = f"{task.get('id')} run {index}/{args.runs}"
            print(f"▶ {label} …", flush=True)
            result = await run_once(task, index, args.timeout)
            status = "ok" if result["success"] else f"FAILED ({result['error'] or 'artifacts'})"
            print(
                f"  {status} · {result['duration_seconds']:.1f}s · "
                f"{result['total_tokens']} tokens · ${result['cost_usd']:.4f} · "
                f"{result['delegations']} delegations",
                flush=True,
            )
            results.append(result)

    rows = summarize(results)
    print_table(rows, config_name)
    payload = {
        "config": config_name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs_per_task": args.runs,
        "summary": rows,
        "results": results,
    }
    print(f"\nResults written to {write_results(payload)}")
    return 0 if all(r["success"] for r in results) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=3, help="runs per task (default 3)")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="path to tasks.yaml")
    parser.add_argument("--task", action="append", help="run only this task id (repeatable)")
    parser.add_argument("--config", help="YAML file of per-agent env overrides to compare")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per run")
    args = parser.parse_args()

    # Environment overrides must land before app.config resolves agent configuration.
    config_name = apply_config(args.config)
    if _BASE_DIR not in sys.path:
        sys.path.insert(0, _BASE_DIR)
    return asyncio.run(main_async(args, config_name))


if __name__ == "__main__":
    raise SystemExit(main())
