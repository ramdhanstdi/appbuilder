"""Per-agent file ownership. Enforced before the write executes, not by prompt."""

import pytest

from app.agent import _zone_error
from app.protocol import is_failure


def test_ba_may_write_its_spec():
    assert _zone_error("ba", "my-app/docs/SPEC.md") is None


def test_ba_may_not_write_backend_code():
    error = _zone_error("ba", "my-app/backend/server.js")
    assert is_failure(error)
    assert "docs/" in error


@pytest.mark.parametrize("path", ["my-app/frontend/App.jsx", "my-app/README.md"])
def test_frontend_owns_its_ui_and_the_app_readme(path):
    assert _zone_error("frontend", path) is None


def test_frontend_may_not_write_the_api_contract():
    # The contract has exactly one owner; the frontend reads it but never changes it.
    assert is_failure(_zone_error("frontend", "my-app/docs/API_CONTRACT.md"))


def test_backend_is_the_sole_owner_of_the_api_contract():
    assert _zone_error("backend", "my-app/docs/API_CONTRACT.md") is None
    assert is_failure(_zone_error("ba", "my-app/backend/server.js"))


def test_backend_may_not_write_the_spec():
    assert is_failure(_zone_error("backend", "my-app/docs/SPEC.md"))


def test_qa_may_write_nothing_and_is_told_it_is_a_reviewer():
    error = _zone_error("qa", "my-app/backend/server.js")
    assert is_failure(error)
    assert "reviewer" in error.lower()
    assert "discuss_with" in error


def test_qa_cannot_even_write_inside_docs():
    assert is_failure(_zone_error("qa", "my-app/docs/SPEC.md"))


def test_workspace_root_file_without_an_app_folder_is_rejected():
    error = _zone_error("frontend", "README.md")
    assert is_failure(error)
    assert "app folder" in error


def test_zone_prefix_does_not_match_by_substring():
    # 'frontend-old/' must not satisfy the 'frontend/' zone.
    assert is_failure(_zone_error("frontend", "my-app/frontend-old/x.js"))
    assert is_failure(_zone_error("backend", "my-app/backend2/server.js"))


def test_exact_file_zone_does_not_match_a_longer_name():
    assert is_failure(_zone_error("frontend", "my-app/README.md.bak"))
