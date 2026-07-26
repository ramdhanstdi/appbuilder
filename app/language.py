"""Session response language: detection, stickiness, and prompt injection.

Human-facing prose (chat replies, specialist reports, generated documents) follows the
language the user actually writes in. Code identifiers, file/folder names, and the
protocol tokens in app/protocol.py stay English always — see `language_directive()`.

The active language lives in a ContextVar so it propagates implicitly through async
LangGraph tool execution, exactly like the per-session workspace root in app/agent.py.
Each WebSocket handler runs in its own task with its own context copy, so two concurrent
sessions in different languages cannot interfere.
"""

import contextvars
import os

from langdetect import DetectorFactory, detect_langs

# Deterministic results: langdetect is randomized by default and would otherwise return
# different codes for the same input across runs.
DetectorFactory.seed = 0

# Language used before anything confident has been detected.
DEFAULT_RESPONSE_LANGUAGE = os.environ.get("DEFAULT_RESPONSE_LANGUAGE", "id") or "id"

# Detection floors. Short follow-ups ('ok', 'lanjut', 'yes', 'thanks') carry no reliable
# signal, and a wrong guess would flip the whole session's language.
MIN_DETECT_CHARS = 12
MIN_DETECT_PROBABILITY = 0.85

_SESSION_LANGUAGE: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_SESSION_LANGUAGE", default=DEFAULT_RESPONSE_LANGUAGE
)

# Languages pinned by an explicit user request, keyed by thread_id. A ContextVar alone is
# not enough here: set_response_language runs inside a LangGraph tool, whose context is a
# copy — the write would not be visible to the WebSocket handler that owns the session.
_PINNED: dict[str, str] = {}

_LANGUAGE_NAMES = {
    "id": "Indonesian",
    "en": "English",
    "ms": "Malay",
    "ja": "Japanese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "ar": "Arabic",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


def detect_language(text: str) -> str | None:
    """Return an ISO 639-1 code, or None when the input is too short or too ambiguous."""
    cleaned = (text or "").strip()
    if len(cleaned) < MIN_DETECT_CHARS:
        return None
    try:
        candidates = detect_langs(cleaned)
    except Exception:
        # langdetect raises on input it cannot profile at all (digits, emoji, symbols).
        return None
    if not candidates:
        return None
    best = candidates[0]
    if best.prob < MIN_DETECT_PROBABILITY:
        return None
    return str(best.lang).lower()


def current_language() -> str:
    """The language currently active for this session."""
    return _SESSION_LANGUAGE.get()


def effective_language(thread_id: str | None = None) -> str:
    """The language to use for this invocation: an explicit pin wins over detection.

    Reads the pin registry as well as the ContextVar because `set_response_language` runs
    inside a LangGraph tool — its ContextVar write lives in a child context and would not
    be visible to the node that runs next in the same graph invocation.
    """
    pinned = _PINNED.get(thread_id) if thread_id else None
    return pinned or _SESSION_LANGUAGE.get()


def resolve_session_language(text: str, thread_id: str | None = None) -> str:
    """Update and return the session language for an incoming user message.

    Sticky: the value changes only on a *confident* detection that differs from the
    current one. Ambiguous input leaves it untouched. An explicit
    `set_response_language()` pin wins over detection for the rest of the session.
    """
    pinned = _PINNED.get(thread_id) if thread_id else None
    if pinned:
        _SESSION_LANGUAGE.set(pinned)
        return pinned

    detected = detect_language(text)
    if detected and detected != _SESSION_LANGUAGE.get():
        _SESSION_LANGUAGE.set(detected)
    return _SESSION_LANGUAGE.get()


def start_session(thread_id: str) -> str:
    """Reset a new session to the configured default language."""
    _PINNED.pop(thread_id, None)
    _SESSION_LANGUAGE.set(DEFAULT_RESPONSE_LANGUAGE)
    return DEFAULT_RESPONSE_LANGUAGE


def pin_language(thread_id: str | None, code: str) -> str:
    """Pin a language explicitly requested by the user; takes precedence over detection."""
    normalized = (code or "").strip().lower()
    if thread_id:
        _PINNED[thread_id] = normalized
    _SESSION_LANGUAGE.set(normalized)
    return normalized


def forget_session(thread_id: str) -> None:
    """Drop the pinned language held for a finished session."""
    _PINNED.pop(thread_id, None)


def language_is_pinned(thread_id: str | None) -> bool:
    """Whether the session's language was set by an explicit user request."""
    return bool(thread_id) and thread_id in _PINNED


def restore_language(thread_id: str, code: str | None, pinned: bool = False) -> str:
    """Re-apply a persisted session language when resuming a saved session.

    Sets the ContextVar for this handler's context and, if the language had been pinned by
    an explicit request, restores that pin too. Falls back to the configured default.
    """
    normalized = (code or "").strip().lower() or DEFAULT_RESPONSE_LANGUAGE
    if pinned and thread_id:
        _PINNED[thread_id] = normalized
    _SESSION_LANGUAGE.set(normalized)
    return normalized


def language_name(code: str) -> str:
    """Human-readable English name of a language code; falls back to the raw code."""
    key = (code or "").strip().lower()
    return _LANGUAGE_NAMES.get(key, key or DEFAULT_RESPONSE_LANGUAGE)


def language_directive(code: str) -> str:
    """A short English instruction to append to a system prompt at invocation time.

    Appended per invocation — never written back into AGENTS[key]["prompt"], which is
    shared module-level state that would leak one session's language into every other.
    """
    name = language_name(code)
    return (
        f"\n\nRESPONSE LANGUAGE: {name} ({code}).\n"
        f"- Write ALL human-facing prose in {name}: chat replies, reports to the Project "
        f"Manager, discussion messages, and the contents of generated documents "
        f"(SPEC.md, API_CONTRACT.md, the app README.md).\n"
        f"- ALWAYS English regardless of the above: code identifiers, function and "
        f"variable names, code comments, log strings, and every file and folder name "
        f"(always 'docs/SPEC.md', never a translated filename — the ownership check "
        f"matches literal paths and a translated name would be rejected).\n"
        f"- ALWAYS English: protocol tokens. The 'STATUS:' value stays DONE / PARTIAL / "
        f"BLOCKED and tool results stay 'OK:' / 'FAILED:'. Only the prose after the "
        f"colon follows the response language."
    )
