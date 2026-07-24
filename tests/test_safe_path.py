"""The filesystem sandbox: every agent file operation is jailed to the session workspace."""

import os

import pytest

from app.agent import _safe_path


def _inside(root: str, resolved: str) -> bool:
    return resolved == root or resolved.startswith(os.path.abspath(root) + os.sep)


def test_accepts_relative_path(workspace):
    resolved = _safe_path("my-app/README.md")
    assert resolved == os.path.join(os.path.abspath(workspace), "my-app", "README.md")


def test_accepts_nested_relative_path(workspace):
    resolved = _safe_path("my-app/backend/src/routes/users.js")
    assert _inside(workspace, resolved)
    assert resolved.endswith(os.path.join("routes", "users.js"))


def test_rejects_parent_traversal(workspace):
    with pytest.raises(ValueError, match="outside the workspace"):
        _safe_path("../secrets.txt")


def test_rejects_deep_traversal(workspace):
    with pytest.raises(ValueError, match="outside the workspace"):
        _safe_path("a/../../../etc/passwd")


def test_rejects_drive_qualified_absolute_path(workspace):
    # A drive-qualified path can only be an attempt to leave the sandbox, on any platform.
    with pytest.raises(ValueError, match="outside the workspace"):
        _safe_path("C:\\Windows\\System32\\drivers\\etc\\hosts")


def test_rejects_unc_path(workspace):
    with pytest.raises(ValueError, match="outside the workspace"):
        _safe_path("//server/share/payload.txt")


def test_rejects_backslash_traversal(workspace):
    # On POSIX a backslash is a legal filename character; the jail must still reject this.
    with pytest.raises(ValueError, match="outside the workspace"):
        _safe_path("..\\..\\windows")


def test_leading_slash_is_workspace_relative_not_filesystem_absolute(workspace):
    resolved = _safe_path("/etc/passwd")
    assert _inside(workspace, resolved)
    assert resolved == os.path.join(os.path.abspath(workspace), "etc", "passwd")


def test_bare_root_resolves_to_the_workspace_itself(workspace):
    assert _safe_path(".") == os.path.abspath(workspace)


@pytest.mark.parametrize(
    "path",
    [".", "my-app", "my-app/docs/SPEC.md", "/my-app/frontend/App.jsx", "a/b/../c.txt"],
)
def test_accepted_paths_always_resolve_inside_the_workspace(workspace, path):
    assert _inside(workspace, _safe_path(path))
