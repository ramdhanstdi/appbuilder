"""Shared test setup.

app.config fails loudly at import time when no API key resolves, and app.agent builds a
ChatOpenAI client per agent at import time. Neither should require a live endpoint or a
real credential to run the suite, so placeholder values are pinned here — before any app
module is imported — and a local .env can never override them (load_dotenv does not
overwrite variables that are already set).
"""

import os

os.environ.setdefault("AGENTS_API_BASE", "http://localhost:1/v1")
os.environ["AGENTS_API_KEY"] = "test-key-not-a-real-credential"
os.environ["AGENTS_MODEL"] = "test-model"
for _prefix in ("PM", "BA", "FRONTEND", "BACKEND", "QA"):
    os.environ.pop(f"{_prefix}_API_KEY", None)
    os.environ.pop(f"{_prefix}_MODEL", None)
    os.environ.pop(f"{_prefix}_API_BASE", None)

import pytest  # noqa: E402

from app import agent as agent_module  # noqa: E402
from app import language as language_module  # noqa: E402


@pytest.fixture
def workspace(tmp_path):
    """Point the session workspace ContextVar at an isolated temp directory."""
    token = agent_module._SESSION_ROOT.set(str(tmp_path))
    try:
        yield str(tmp_path)
    finally:
        agent_module._SESSION_ROOT.reset(token)


@pytest.fixture(autouse=True)
def clean_session_state():
    """Keep module-level session dicts from leaking between tests."""
    yield
    agent_module._SPECIALIST_HISTORY.clear()
    agent_module._ASSIGN_BUDGET.clear()
    language_module._PINNED.clear()
    language_module._SESSION_LANGUAGE.set(language_module.DEFAULT_RESPONSE_LANGUAGE)


@pytest.fixture
def no_stream_writer(monkeypatch):
    """Replace LangGraph's stream writer, which only exists inside a graph run."""
    events = []
    monkeypatch.setattr(agent_module, "get_stream_writer", lambda: events.append)
    return events
