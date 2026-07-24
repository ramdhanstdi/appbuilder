"""Machine-readable protocol tokens exchanged between tools, agents, and the UI.

These tokens are **not prose**. The runtime branches on them: the UI decides whether a
tool call rendered as success or failure, and the Project Manager decides whether a
specialist actually finished its work. They must therefore stay stable and English even
when the human-facing response language changes (see app/language.py).

Only the detail *after* the colon is localizable prose. No module outside this one may
inspect a result prefix directly — use `is_failure()`.
"""

# --- Tool result tokens ------------------------------------------------------
RESULT_OK = "OK"
RESULT_FAILED = "FAILED"

# --- Specialist report status tokens ----------------------------------------
STATUS_DONE = "DONE"
STATUS_PARTIAL = "PARTIAL"
STATUS_BLOCKED = "BLOCKED"

REPORT_STATUSES = (STATUS_DONE, STATUS_PARTIAL, STATUS_BLOCKED)


def ok(detail: str) -> str:
    """Build a successful tool result."""
    return f"{RESULT_OK}: {detail}"


def failed(detail: str) -> str:
    """Build a failed tool result."""
    return f"{RESULT_FAILED}: {detail}"


def is_failure(result: str) -> bool:
    """True when a tool result carries the failure token.

    The single place in the codebase allowed to inspect a result prefix.
    """
    return str(result or "").lstrip().startswith(RESULT_FAILED)


def report_status(report: str) -> str | None:
    """Extract the STATUS token from the first line of a specialist report, if present."""
    first_line = str(report or "").strip().splitlines()[:1]
    if not first_line:
        return None
    head = first_line[0].strip().upper()
    if not head.startswith("STATUS:"):
        return None
    value = head[len("STATUS:"):].strip()
    return value if value in REPORT_STATUSES else None
