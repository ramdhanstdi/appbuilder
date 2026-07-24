"""Containerized command execution: what the runner is allowed to see and do."""

import os

from app import agent as agent_module
from app.agent import _runner_argv, run_command
from app.protocol import is_failure


def _argv(monkeypatch, workspace, command="node --version", subdir=None, **env):
    for name, value in env.items():
        monkeypatch.setattr(agent_module, name, value)
    cwd = os.path.join(workspace, subdir) if subdir else workspace
    return _runner_argv("appbuilder-run-test", command, cwd)


def test_only_the_session_workspace_is_mounted(monkeypatch, workspace):
    argv = _argv(monkeypatch, workspace, RUNNER_VOLUMES_FROM="")
    mounts = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-v"]
    assert mounts == [f"{os.path.abspath(workspace)}:/workspace"]


def test_the_working_directory_is_mapped_inside_the_mount(monkeypatch, workspace):
    os.makedirs(os.path.join(workspace, "my-app", "backend"), exist_ok=True)
    argv = _argv(monkeypatch, workspace, subdir="my-app/backend", RUNNER_VOLUMES_FROM="")
    assert argv[argv.index("-w") + 1] == "/workspace/my-app/backend"


def test_networking_is_disabled_by_default(monkeypatch, workspace):
    argv = _argv(monkeypatch, workspace, RUNNER_VOLUMES_FROM="", RUNNER_NETWORK="none")
    assert argv[argv.index("--network") + 1] == "none"


def test_network_can_be_enabled_explicitly(monkeypatch, workspace):
    argv = _argv(monkeypatch, workspace, RUNNER_VOLUMES_FROM="", RUNNER_NETWORK="bridge")
    assert argv[argv.index("--network") + 1] == "bridge"


def test_the_container_is_memory_capped_and_non_root(monkeypatch, workspace):
    argv = _argv(
        monkeypatch, workspace,
        RUNNER_VOLUMES_FROM="", RUNNER_MEMORY="256m", RUNNER_USER="1000:1000",
    )
    assert argv[argv.index("--memory") + 1] == "256m"
    assert argv[argv.index("--user") + 1] == "1000:1000"
    assert "--security-opt" in argv and "no-new-privileges" in argv
    assert "--rm" in argv


def test_the_command_is_passed_as_an_argument_not_interpolated(monkeypatch, workspace):
    # The command reaches the runner as a single argv element, so nothing in it can
    # break out into the docker invocation itself.
    hostile = 'echo hi" --privileged -v /:/host "'
    argv = _argv(monkeypatch, workspace, command=hostile, RUNNER_VOLUMES_FROM="")
    assert argv[-3:] == ["sh", "-lc", hostile]
    assert "--privileged" not in argv[:-1]


def test_when_the_server_is_containerized_the_runner_reuses_its_mounts(monkeypatch, workspace):
    argv = _argv(monkeypatch, workspace, RUNNER_VOLUMES_FROM="appbuilder")
    assert argv[argv.index("--volumes-from") + 1] == "appbuilder"
    # Same mount at the same path, so the working directory needs no translation.
    assert argv[argv.index("-w") + 1] == os.path.abspath(workspace)
    assert "-v" not in argv


def test_a_missing_docker_binary_reports_the_escape_hatch(monkeypatch, workspace):
    monkeypatch.setattr(agent_module, "SANDBOX_COMMANDS", True)
    monkeypatch.setattr(agent_module, "DOCKER_BIN", "docker-not-installed-xyz")
    result = run_command.invoke({"command": "echo hi", "working_dir": "."})
    assert is_failure(result)
    assert "ALLOW_UNSANDBOXED_COMMANDS" in result


def test_the_unsandboxed_escape_hatch_still_runs_commands(monkeypatch, workspace):
    monkeypatch.setattr(agent_module, "SANDBOX_COMMANDS", False)
    result = run_command.invoke(
        {"command": "python -c \"print('sandbox-off')\"", "working_dir": "."}
    )
    assert not is_failure(result)
    assert "sandbox-off" in result


def test_run_command_still_rejects_a_working_dir_outside_the_workspace(monkeypatch, workspace):
    monkeypatch.setattr(agent_module, "SANDBOX_COMMANDS", False)
    result = run_command.invoke({"command": "echo hi", "working_dir": "../.."})
    assert is_failure(result)
    assert "outside the workspace" in result
