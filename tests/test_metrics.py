"""Instrumentation: usage extraction, cost estimation, and per-request persistence."""

import json
import os

from langchain_core.messages import AIMessage

from app.config import MODEL_PRICING
from app.metrics import RunMetrics, estimate_cost, extract_usage


class _Response:
    """A provider response carrying usage in one of the shapes LangChain surfaces."""

    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


def test_extract_usage_reads_usage_metadata():
    response = _Response(usage_metadata={"input_tokens": 120, "output_tokens": 45})
    assert extract_usage(response) == (120, 45)


def test_extract_usage_falls_back_to_response_metadata():
    response = _Response(
        response_metadata={"token_usage": {"prompt_tokens": 10, "completion_tokens": 3}}
    )
    assert extract_usage(response) == (10, 3)


def test_extract_usage_returns_zero_when_the_provider_omits_it():
    # Not every provider reports usage; that must not crash a run.
    assert extract_usage(_Response()) == (0, 0)
    assert extract_usage(AIMessage(content="hi")) == (0, 0)


def test_estimate_cost_uses_the_price_list():
    model = next(iter(MODEL_PRICING))
    pricing = MODEL_PRICING[model]
    expected = pricing["input"] + 500_000 / 1_000_000 * pricing["output"]
    assert estimate_cost(model, 1_000_000, 500_000) == expected


def test_unpriced_model_costs_nothing_but_still_counts_tokens(tmp_path):
    collector = RunMetrics(runs_dir=str(tmp_path))
    collector.start_request("t1", "build a todo app")
    collector.record_llm_call(
        "t1", "ba", "some-local-model",
        _Response(usage_metadata={"input_tokens": 100, "output_tokens": 20}), 0.5,
    )
    totals = collector.totals("t1")
    assert totals["total_tokens"] == 120
    assert totals["cost_usd"] == 0.0


def test_metrics_are_broken_down_per_agent(tmp_path):
    collector = RunMetrics(runs_dir=str(tmp_path))
    collector.start_request("t1", "build it")
    collector.record_llm_call(
        "t1", "pm", "gpt-4o-mini",
        _Response(usage_metadata={"input_tokens": 10, "output_tokens": 2}), 0.1,
    )
    collector.record_llm_call(
        "t1", "ba", "gpt-4o",
        _Response(usage_metadata={"input_tokens": 50, "output_tokens": 30}), 0.4,
    )
    collector.record_tool_call("t1", "ba", 3)
    collector.record_retry("t1", "ba")

    summary = collector.summary("t1")
    assert summary["agents"]["pm"]["prompt_tokens"] == 10
    assert summary["agents"]["ba"]["total_tokens"] == 80
    assert summary["agents"]["ba"]["tool_calls"] == 3
    assert summary["agents"]["ba"]["retries"] == 1
    assert summary["totals"]["llm_calls"] == 2
    assert summary["totals"]["cost_usd"] > 0


def test_finish_request_writes_one_jsonl_record_per_request(tmp_path):
    collector = RunMetrics(runs_dir=str(tmp_path))

    collector.start_request("thread-a", "first request")
    collector.record_llm_call(
        "thread-a", "pm", "gpt-4o",
        _Response(usage_metadata={"input_tokens": 5, "output_tokens": 1}), 0.2,
    )
    collector.finish_request("thread-a")

    collector.start_request("thread-a", "second request")
    collector.record_llm_call(
        "thread-a", "qa", "gpt-4o",
        _Response(usage_metadata={"input_tokens": 7, "output_tokens": 2}), 0.3,
    )
    collector.finish_request("thread-a")

    path = os.path.join(str(tmp_path), "thread-a.jsonl")
    lines = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert len(lines) == 2
    # Each record covers its own request…
    assert set(lines[0]["agents"]) == {"pm"}
    assert set(lines[1]["agents"]) == {"qa"}
    assert lines[1]["request"] == "second request"
    # …while session_totals accumulates across the whole session.
    assert lines[1]["session_totals"]["total_tokens"] == 15


def test_finish_request_without_any_activity_writes_nothing(tmp_path):
    collector = RunMetrics(runs_dir=str(tmp_path))
    collector.start_request("quiet", "hello")
    assert collector.finish_request("quiet") is None
    assert not os.path.exists(os.path.join(str(tmp_path), "quiet.jsonl"))


def test_drop_releases_session_metrics(tmp_path):
    collector = RunMetrics(runs_dir=str(tmp_path))
    collector.start_request("t1", "x")
    collector.record_tool_call("t1", "pm")
    collector.drop("t1")
    assert collector.summary("t1")["agents"] == {}


def test_a_hostile_thread_id_cannot_escape_the_runs_directory(tmp_path):
    collector = RunMetrics(runs_dir=str(tmp_path))
    collector.start_request("../../escaped", "x")
    collector.record_tool_call("../../escaped", "pm")
    collector.finish_request("../../escaped")
    assert os.path.exists(os.path.join(str(tmp_path), "escaped.jsonl"))
