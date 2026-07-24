"""Per-agent token, cost, and latency instrumentation.

Per-agent model routing makes cost engineering *possible*; without measurement it stays a
capability nobody can evaluate. This module records what each agent actually consumed so a
routing change can be judged instead of guessed.

Two scopes are tracked per session: cumulative totals for the whole session, and a bucket
for the current user request that is reset on each new request and written to
`runs/<thread_id>.jsonl` when the request completes.

Nothing here may break a run: providers that omit usage metadata simply contribute zero
tokens, and a failed write to `runs/` is swallowed.
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from app.config import MODEL_PRICING

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.environ.get("RUNS_DIR", os.path.join(_BASE_DIR, "runs"))


@dataclass
class AgentMetrics:
    """What one agent consumed within one scope."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    retries: int = 0
    latency_seconds: float = 0.0
    cost_usd: float = 0.0
    models: list[str] = field(default_factory=list)

    def add_llm_call(self, prompt_tokens, completion_tokens, latency, cost, model):
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.llm_calls += 1
        self.latency_seconds += latency
        self.cost_usd += cost
        if model and model not in self.models:
            self.models.append(model)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def extract_usage(response) -> tuple[int, int]:
    """Read (prompt_tokens, completion_tokens) off a LangChain response.

    Providers disagree on where usage lives and some omit it entirely; an absent count is
    zero, never an error.
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)

    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    if isinstance(token_usage, dict) and token_usage:
        prompt = token_usage.get("prompt_tokens", token_usage.get("input_tokens"))
        completion = token_usage.get("completion_tokens", token_usage.get("output_tokens"))
        return int(prompt or 0), int(completion or 0)
    return 0, 0


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimated USD cost, or 0.0 when the model has no entry in MODEL_PRICING."""
    pricing = MODEL_PRICING.get((model or "").strip())
    if not pricing:
        return 0.0
    return (
        prompt_tokens / 1_000_000 * float(pricing.get("input", 0.0))
        + completion_tokens / 1_000_000 * float(pricing.get("output", 0.0))
    )


class RunMetrics:
    """Thread-safe collector, keyed by thread_id and then agent key."""

    def __init__(self, runs_dir: str = RUNS_DIR):
        self._lock = threading.Lock()
        self._session: dict[str, dict[str, AgentMetrics]] = {}
        self._request: dict[str, dict[str, AgentMetrics]] = {}
        self._request_started: dict[str, float] = {}
        self._request_prompt: dict[str, str] = {}
        self._runs_dir = runs_dir

    # -- recording ----------------------------------------------------------
    def _bucket(self, scope: dict, thread_id: str, agent_key: str) -> AgentMetrics:
        return scope.setdefault(thread_id, {}).setdefault(agent_key, AgentMetrics())

    def record_llm_call(self, thread_id, agent_key, model, response, latency_seconds):
        prompt_tokens, completion_tokens = extract_usage(response)
        cost = estimate_cost(model, prompt_tokens, completion_tokens)
        with self._lock:
            for scope in (self._session, self._request):
                self._bucket(scope, thread_id, agent_key).add_llm_call(
                    prompt_tokens, completion_tokens, latency_seconds, cost, model
                )
        return prompt_tokens, completion_tokens, cost

    def record_tool_call(self, thread_id: str, agent_key: str, count: int = 1) -> None:
        with self._lock:
            for scope in (self._session, self._request):
                self._bucket(scope, thread_id, agent_key).tool_calls += count

    def record_retry(self, thread_id: str, agent_key: str) -> None:
        with self._lock:
            for scope in (self._session, self._request):
                self._bucket(scope, thread_id, agent_key).retries += 1

    # -- request lifecycle --------------------------------------------------
    def start_request(self, thread_id: str, prompt: str = "") -> None:
        with self._lock:
            self._request[thread_id] = {}
            self._request_started[thread_id] = time.monotonic()
            self._request_prompt[thread_id] = (prompt or "")[:200]

    def finish_request(self, thread_id: str) -> dict | None:
        """Build the record for the request that just finished and append it to runs/."""
        with self._lock:
            agents = self._request.get(thread_id)
            if not agents:
                return None
            started = self._request_started.get(thread_id, time.monotonic())
            record = {
                "thread_id": thread_id,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "request": self._request_prompt.get(thread_id, ""),
                "duration_seconds": round(time.monotonic() - started, 3),
                "totals": _totals(agents),
                "agents": {key: _clean(asdict(m)) for key, m in agents.items()},
                "session_totals": _totals(self._session.get(thread_id, {})),
            }
        self._append(thread_id, record)
        return record

    def _append(self, thread_id: str, record: dict) -> None:
        try:
            os.makedirs(self._runs_dir, exist_ok=True)
            safe_name = os.path.basename(thread_id.strip()) or "unknown"
            path = os.path.join(self._runs_dir, f"{safe_name}.jsonl")
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Instrumentation must never take a run down with it.
            pass

    # -- reading ------------------------------------------------------------
    def summary(self, thread_id: str) -> dict:
        with self._lock:
            agents = self._session.get(thread_id, {})
            return {
                "thread_id": thread_id,
                "totals": _totals(agents),
                "agents": {key: _clean(asdict(m)) for key, m in agents.items()},
            }

    def totals(self, thread_id: str) -> dict:
        with self._lock:
            return _totals(self._session.get(thread_id, {}))

    def drop(self, thread_id: str) -> None:
        """Release in-memory metrics for a finished session; the runs/ file stays."""
        with self._lock:
            self._session.pop(thread_id, None)
            self._request.pop(thread_id, None)
            self._request_started.pop(thread_id, None)
            self._request_prompt.pop(thread_id, None)


def _totals(agents: dict[str, AgentMetrics]) -> dict:
    return {
        "prompt_tokens": sum(m.prompt_tokens for m in agents.values()),
        "completion_tokens": sum(m.completion_tokens for m in agents.values()),
        "total_tokens": sum(m.total_tokens for m in agents.values()),
        "llm_calls": sum(m.llm_calls for m in agents.values()),
        "tool_calls": sum(m.tool_calls for m in agents.values()),
        "retries": sum(m.retries for m in agents.values()),
        "latency_seconds": round(sum(m.latency_seconds for m in agents.values()), 3),
        "cost_usd": round(sum(m.cost_usd for m in agents.values()), 6),
    }


def _clean(data: dict) -> dict:
    data["latency_seconds"] = round(data["latency_seconds"], 3)
    data["cost_usd"] = round(data["cost_usd"], 6)
    data["total_tokens"] = data["prompt_tokens"] + data["completion_tokens"]
    return data


# Single process-wide collector; the server and the benchmark harness share it.
METRICS = RunMetrics()
