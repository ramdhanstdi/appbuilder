"""The protocol layer: the runtime branches on these tokens, so they must not drift."""

from app.protocol import (
    RESULT_FAILED,
    RESULT_OK,
    STATUS_BLOCKED,
    STATUS_DONE,
    STATUS_PARTIAL,
    failed,
    is_failure,
    ok,
    report_status,
)


def test_builders_use_the_documented_shape():
    assert ok("file saved") == f"{RESULT_OK}: file saved"
    assert failed("no such file") == f"{RESULT_FAILED}: no such file"


def test_is_failure_distinguishes_the_two_results():
    assert is_failure(failed("boom"))
    assert not is_failure(ok("fine"))


def test_is_failure_tolerates_empty_and_non_string_input():
    assert not is_failure("")
    assert not is_failure(None)
    assert is_failure(f"  {RESULT_FAILED}: leading whitespace")


def test_is_failure_is_not_fooled_by_a_mention_further_in_the_text():
    assert not is_failure(ok("the previous attempt returned FAILED: retried and worked"))


def test_report_status_reads_the_first_line_only():
    assert report_status(f"STATUS: {STATUS_BLOCKED}\nneed credentials") == STATUS_BLOCKED
    assert report_status(f"status: {STATUS_DONE.lower()}") == STATUS_DONE
    assert report_status(f"here is my report\nSTATUS: {STATUS_PARTIAL}") is None


def test_report_status_rejects_an_unknown_token():
    assert report_status("STATUS: SELESAI") is None
    assert report_status("") is None
