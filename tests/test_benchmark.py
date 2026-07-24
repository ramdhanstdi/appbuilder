"""The benchmark harness's measurement logic, tested without touching a provider."""

import os

import yaml

from benchmark.run_benchmark import (
    DEFAULT_TASKS,
    _count_custom,
    apply_config,
    artifacts_present,
    load_yaml,
    summarize,
)


def _counters():
    return {"delegations": 0, "qa_repair_rounds": 0, "questions": 0, "tool_failures": 0}


def test_shipped_tasks_file_is_valid_and_complete():
    tasks = load_yaml(DEFAULT_TASKS)["tasks"]
    assert tasks
    for task in tasks:
        assert task["id"] and task["prompt"] and task["expect"]
        # A canned answer is what keeps a run from blocking on ask_user.
        assert task.get("answer")


def test_artifacts_present_accepts_a_complete_run(tmp_path):
    (tmp_path / "todo-app" / "docs").mkdir(parents=True)
    (tmp_path / "todo-app" / "docs" / "SPEC.md").write_text("criteria", encoding="utf-8")
    (tmp_path / "todo-app" / "backend").mkdir()
    (tmp_path / "todo-app" / "backend" / "server.js").write_text("code", encoding="utf-8")

    ok, missing = artifacts_present(
        str(tmp_path), ["todo-app/docs/SPEC.md", "todo-app/backend/"]
    )
    assert ok and missing == []


def test_artifacts_present_reports_what_is_missing(tmp_path):
    (tmp_path / "todo-app" / "docs").mkdir(parents=True)
    (tmp_path / "todo-app" / "docs" / "SPEC.md").write_text("criteria", encoding="utf-8")

    ok, missing = artifacts_present(
        str(tmp_path), ["todo-app/docs/SPEC.md", "todo-app/backend/", "todo-app/README.md"]
    )
    assert not ok
    assert missing == ["todo-app/backend/", "todo-app/README.md"]


def test_an_empty_file_does_not_count_as_produced(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "README.md").write_text("", encoding="utf-8")
    ok, missing = artifacts_present(str(tmp_path), ["app/README.md"])
    assert not ok and missing == ["app/README.md"]


def test_an_empty_directory_does_not_count_as_produced(tmp_path):
    (tmp_path / "app" / "frontend").mkdir(parents=True)
    ok, missing = artifacts_present(str(tmp_path), ["app/frontend/"])
    assert not ok and missing == ["app/frontend/"]


def test_qa_repair_rounds_count_only_messages_originating_from_qa():
    counters = _counters()
    _count_custom({"type": "peer_in", "from": "qa", "agent": "backend"}, counters)
    _count_custom({"type": "peer_in", "from": "frontend", "agent": "backend"}, counters)
    _count_custom({"type": "peer_reply", "from": "qa", "agent": "backend"}, counters)
    assert counters["qa_repair_rounds"] == 1


def test_failed_tool_results_are_counted():
    counters = _counters()
    _count_custom({"type": "tool_result", "ok": False}, counters)
    _count_custom({"type": "tool_result", "ok": True}, counters)
    assert counters["tool_failures"] == 1


def _run(task, success, duration, tokens, cost, delegations=4, repairs=1, retries=0):
    return {
        "task": task, "success": success, "duration_seconds": duration,
        "total_tokens": tokens, "cost_usd": cost, "delegations": delegations,
        "qa_repair_rounds": repairs, "retries": retries, "qa_status": "DONE",
    }


def test_summarize_reports_means_and_variance_per_task():
    results = [
        _run("todo-app", True, 100.0, 1000, 0.10),
        _run("todo-app", True, 120.0, 1400, 0.14),
        _run("todo-app", False, 110.0, 1200, 0.12),
    ]
    row = summarize(results)[0]

    assert row["task"] == "todo-app"
    assert row["runs"] == 3
    assert row["success_rate"] == 2 / 3
    assert row["duration_seconds_mean"] == 110.0
    assert row["total_tokens_mean"] == 1200.0
    assert row["total_tokens_stdev"] > 0
    assert row["cost_usd_mean"] == 0.12


def test_a_single_run_has_zero_variance_not_an_error():
    row = summarize([_run("solo", True, 90.0, 800, 0.08)])[0]
    assert row["duration_seconds_stdev"] == 0.0


def test_summarize_keeps_tasks_separate():
    rows = summarize([
        _run("a", True, 10.0, 100, 0.01),
        _run("b", True, 20.0, 200, 0.02),
    ])
    assert {row["task"] for row in rows} == {"a", "b"}


def test_apply_config_sets_environment_overrides(tmp_path, monkeypatch):
    config = tmp_path / "routing.yaml"
    config.write_text(
        yaml.safe_dump({"name": "cheap-qa", "env": {"QA_MODEL": "gpt-4o-mini"}}),
        encoding="utf-8",
    )
    # setenv (not delenv) so monkeypatch records the original state and restores it —
    # apply_config writes os.environ directly and would otherwise leak into later tests.
    monkeypatch.setenv("QA_MODEL", "placeholder")

    assert apply_config(str(config)) == "cheap-qa"
    assert os.environ["QA_MODEL"] == "gpt-4o-mini"


def test_no_config_means_the_baseline():
    assert apply_config(None) == "default"
