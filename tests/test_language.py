"""Response language: detection floors, stickiness, and non-mutation of shared state."""

import contextvars

import pytest
from langchain_core.messages import AIMessage

from app import agent as agent_module
from app import language as language_module
from app.agent import _run_specialist
from app.config import AGENTS
from app.language import (
    DEFAULT_RESPONSE_LANGUAGE,
    detect_language,
    effective_language,
    language_directive,
    language_name,
    pin_language,
    resolve_session_language,
    start_session,
)
from app.protocol import RESULT_FAILED, RESULT_OK, STATUS_BLOCKED, STATUS_DONE, STATUS_PARTIAL

_ENGLISH = "Build me a simple todo list application with React and an Express backend."
_INDONESIAN = "Buatkan aplikasi daftar tugas sederhana dengan React dan backend Express."


@pytest.mark.parametrize("text", ["ok", "lanjut", "yes", "thanks", "next", "", "   ", "123 456"])
def test_short_or_ambiguous_input_produces_no_guess(text):
    assert detect_language(text) is None


def test_confident_input_is_detected():
    assert detect_language(_ENGLISH) == "en"
    assert detect_language(_INDONESIAN) == "id"


def test_session_language_is_sticky_across_an_ambiguous_follow_up():
    start_session("t1")
    assert resolve_session_language(_ENGLISH, "t1") == "en"
    # Ambiguous follow-ups must not flip a session mid-build.
    assert resolve_session_language("ok", "t1") == "en"
    assert resolve_session_language("lanjut", "t1") == "en"
    # A second confident detection that differs does switch it.
    assert resolve_session_language(_INDONESIAN, "t1") == "id"


def test_new_session_starts_at_the_configured_default():
    pin_language("t1", "ja")
    assert start_session("t1") == DEFAULT_RESPONSE_LANGUAGE
    assert effective_language("t1") == DEFAULT_RESPONSE_LANGUAGE


def test_an_explicit_pin_outranks_detection():
    start_session("t1")
    pin_language("t1", "en")
    # Even a confident Indonesian message leaves the pinned language alone.
    assert resolve_session_language(_INDONESIAN, "t1") == "en"
    assert effective_language("t1") == "en"


def test_language_name_maps_known_codes_and_falls_back_to_the_raw_code():
    assert language_name("id") == "Indonesian"
    assert language_name("en") == "English"
    assert language_name("ja") == "Japanese"
    assert language_name("xx") == "xx"


def test_directive_names_the_language_and_pins_the_protocol_to_english():
    directive = language_directive("ja")
    assert "Japanese" in directive
    for token in (STATUS_DONE, STATUS_PARTIAL, STATUS_BLOCKED, RESULT_OK, RESULT_FAILED):
        assert token in directive
    assert "docs/SPEC.md" in directive


def test_protocol_tokens_survive_a_non_indonesian_non_english_session():
    start_session("t1")
    pin_language("t1", "ja")
    directive = language_directive(effective_language("t1"))
    assert f"{STATUS_DONE} / {STATUS_PARTIAL} / {STATUS_BLOCKED}" in directive
    assert f"'{RESULT_OK}:' / '{RESULT_FAILED}:'" in directive


async def test_language_injection_does_not_mutate_shared_agent_prompts(
    monkeypatch, no_stream_writer
):
    """AGENTS is module-level shared state; mutating it would leak across sessions."""
    before = {key: cfg["prompt"] for key, cfg in AGENTS.items()}

    captured = {}

    class _CapturingLLM:
        async def ainvoke(self, messages):
            captured["system"] = messages[0].content
            return AIMessage(content=f"STATUS: {STATUS_DONE}\ndone")

    monkeypatch.setitem(agent_module._SPECIALIST_LLMS, "ba", _CapturingLLM())
    pin_language("lang-test", "en")
    await _run_specialist("ba", "write the spec", "lang-test")

    assert "English" in captured["system"]
    assert captured["system"].startswith(AGENTS["ba"]["prompt"])
    for key, prompt in before.items():
        assert AGENTS[key]["prompt"] == prompt


def test_concurrent_sessions_do_not_share_a_language():
    """Each WebSocket handler runs in its own context copy, like these two."""
    results = {}

    def session(thread_id, text):
        start_session(thread_id)
        results[thread_id] = resolve_session_language(text, thread_id)

    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()
    ctx_a.run(session, "a", _ENGLISH)
    ctx_b.run(session, "b", _INDONESIAN)

    assert results == {"a": "en", "b": "id"}
    # Neither leaked into the other's context, nor into this one.
    assert language_module._SESSION_LANGUAGE.get() == DEFAULT_RESPONSE_LANGUAGE
