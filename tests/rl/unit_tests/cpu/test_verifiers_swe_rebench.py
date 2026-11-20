# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the SWE-rebench V2.1 Verifiers example and its recipe."""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers")
pytest.importorskip("swerebench_v2")

import verifiers.v1 as vf
from swerebench_v2.taskset import build_eval_script

from torchtitan.rl.examples.verifiers.swe_rebench.taskset import (
    BASE_COMMIT_KEY,
    project_name,
    read_task_directory,
    run_grader,
    strip_script_markers,
    SWERebenchV21Task,
    SWERebenchV21Taskset,
    SWERebenchV21TasksetConfig,
)
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.state import state_cls
from verifiers.v1.trace import Trace, TraceTask

FAIL_TO_PASS = ["tests/test_app.py::test_fix"]
PASS_TO_PASS = ["tests/test_app.py::test_old_a", "tests/test_app.py::test_old_b"]
RESOURCES = vf.TaskResources(cpu=1, memory=1, disk=1)
# Patches to files with CRLF line endings, as some V2.1 repositories have.
TEST_PATCH = "diff --git a/tests/test_app.py b/tests/test_app.py\n-old\r\n+new\r\n"
GOLD_PATCH = "diff --git a/app.py b/app.py\n-old\r\n+new\r\n"

# A stand-in for V2.1's tests/grade.py with the same command line and report: a
# test passes when the log has a "PASSED <name>" line.
GRADER = """
import json, sys
config_path, log_path, report_path = sys.argv[1:4]
config = json.load(open(config_path))
lines = open(log_path).read().splitlines()
# V2.1 grades the test command's output alone.
assert not any(line.startswith("SWEREBENCH_V2_TEST_OUTPUT") for line in lines)
passed = {line.split()[1] for line in lines if line.startswith("PASSED ")}
failed = {line.split()[1] for line in lines if line.startswith("FAILED ")}
report = {
    "resolved": passed >= set(config["FAIL_TO_PASS"] + config["PASS_TO_PASS"]),
    "parsed_counts": {"PASSED": len(passed), "FAILED": len(failed)},
    "fail_to_pass": {name: name in passed for name in config["FAIL_TO_PASS"]},
    "pass_to_pass": {name: name in passed for name in config["PASS_TO_PASS"]},
}
json.dump(report, open(report_path, "w"))
sys.exit(0 if report["resolved"] else 1)
"""


def _write_task(
    root: Path,
    name: str,
    *,
    repo: str = "owner/app",
    language: str = "python",
    quality_flags: tuple[str, ...] = (),
    fail_to_pass: list[str] = FAIL_TO_PASS,
    pass_to_pass: list[str] = PASS_TO_PASS,
    test_command: str = "pytest -rA tests/test_app.py",
    test_env: str = "",
    container_env: str = "",
) -> Path:
    """Write one task directory laid out like SWE-rebench V2.1's Harbor export.
    `test_env` holds tests/test.sh's export lines; `container_env` is task.toml's
    `env` table."""
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    flags = ", ".join(f'"{flag}"' for flag in quality_flags)
    (task_dir / "task.toml").write_text(
        f"""[task]
name = "org/{name}"

[metadata]
repo = "{repo}"
language = "{language}"
log_parser = "parse_log_pytest"
quality_flags = [{flags}]

[environment]
workdir = "/app"
docker_image = "docker.io/swerebenchv2/app:1-abc@sha256:0123"
{container_env}
"""
    )
    (task_dir / "instruction.md").write_text(f"Fix {name}.")
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        "\n"
        "WORKDIR=/app\n"
        "TEST_FILES=(tests/test_app.py)\n"
        f"{test_env}IFS= read -r -d '' HARBOR_TEST_CMD <<'HARBOR_TEST_CMD_EOF'\n"
        f"{test_command}\n"
        "HARBOR_TEST_CMD_EOF\n"
        "\n"
        'cd "$WORKDIR"\n'
        "export HARBOR_TEST_CMD=\"${HARBOR_TEST_CMD%$'\\n'}\"\n"
    )
    (task_dir / "tests" / "grade.py").write_text(GRADER)
    (task_dir / "tests" / "config.json").write_text(
        json.dumps({"FAIL_TO_PASS": fail_to_pass, "PASS_TO_PASS": pass_to_pass})
    )
    (task_dir / "tests" / "test.patch").write_bytes(TEST_PATCH.encode())
    (task_dir / "solution" / "gold.patch").write_bytes(GOLD_PATCH.encode())
    return task_dir


def test_a_task_directory_becomes_a_prime_envs_task(tmp_path) -> None:
    task_dir = read_task_directory(_write_task(tmp_path, "t-1"), resources=RESOURCES)

    data = task_dir.data
    assert (task_dir.repo, data.name, data.prompt) == ("owner/app", "t-1", "Fix t-1.")
    assert data.image == "docker.io/swerebenchv2/app:1-abc@sha256:0123"
    assert data.workdir == "/app"
    assert (data.fail_to_pass, data.pass_to_pass) == (FAIL_TO_PASS, PASS_TO_PASS)
    # Byte for byte: a patch that lost its "\r" no longer applies.
    assert (data.test_patch, data.gold_patch) == (TEST_PATCH, GOLD_PATCH)
    assert (data.grader_source, data.container_env) == (GRADER, {})
    # V2.1 has no base commit; setup reads it from the image.
    assert data.base_commit == ""


def test_a_task_without_fail_to_pass_tests_is_rejected(tmp_path) -> None:
    """Every rollout would pass an empty FAIL_TO_PASS list."""
    with pytest.raises(ValueError, match="no FAIL_TO_PASS tests"):
        read_task_directory(
            _write_task(tmp_path, "t-1", fail_to_pass=[]), resources=RESOURCES
        )


def test_the_test_command_runs_as_in_v21s_test_script(tmp_path) -> None:
    """The task's test environment reaches the command, which runs in a login shell;
    a multi-line command with shell variables survives prime-envs' `set -u` script."""
    task_dir = _write_task(
        tmp_path,
        "t-1",
        language="rust",
        test_command=(
            'printf "threads=%s\\n" "$RUST_TEST_THREADS"\n'
            "shopt -q login_shell && echo login_shell"
        ),
        test_env="export RUST_TEST_THREADS=1\n",
        container_env='env = { _JAVA_OPTIONS = "-Xmx1g" }',
    )
    data = read_task_directory(task_dir, resources=RESOURCES).data
    task = SWERebenchV21Task(data, vf.TaskConfig())

    script = build_eval_script(data.install_config["test_cmd"])
    result = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert strip_script_markers(result.stdout).splitlines() == [
        "threads=1",
        "login_shell",
    ]
    assert task.runtime_env() == {"_JAVA_OPTIONS": "-Xmx1g"}


def _load(root: Path, **options) -> list[SWERebenchV21Task]:
    config = SWERebenchV21TasksetConfig(dataset_dir=root, **options)
    return list(SWERebenchV21Taskset(config).load())


def test_training_and_validation_never_share_a_project(tmp_path) -> None:
    for index in range(20):
        _write_task(tmp_path, f"t-{index}", repo=f"owner/repo-{index % 10}")
    # The same ten projects under a former owner, capitalized differently.
    for index in range(10):
        _write_task(tmp_path, f"renamed-{index}", repo=f"old-owner/Repo-{index}")
    _write_task(tmp_path, "weak", repo="owner/repo-0", quality_flags=("weak_f2p",))

    def projects(tasks: list[SWERebenchV21Task]) -> set[str]:
        repos = {
            read_task_directory(tmp_path / task.data.name, resources=RESOURCES).repo
            for task in tasks
        }
        return {repo.split("/")[1].lower() for repo in repos}

    training = _load(tmp_path, partition="train")
    validation = _load(tmp_path, partition="validation")
    assert projects(training).isdisjoint(projects(validation))
    # 10% of the 10 projects: one project, with its 3 tasks.
    assert (len(projects(validation)), len(validation), len(training)) == (1, 3, 27)
    assert "weak" not in {task.data.name for task in training + validation}
    assert project_name("old-owner/Repo-3") == project_name("owner/repo-3")


def test_a_language_filter_keeps_the_split(tmp_path) -> None:
    for index in range(10):
        _write_task(tmp_path, f"py-{index}", repo=f"owner/repo-{index}")
        _write_task(tmp_path, f"go-{index}", repo=f"owner/repo-{index}", language="go")

    def names(tasks: list[SWERebenchV21Task]) -> set[str]:
        return {task.data.name for task in tasks}

    validation = names(_load(tmp_path, partition="validation"))
    python_validation = names(
        _load(tmp_path, partition="validation", languages=["python"])
    )
    go_training = names(_load(tmp_path, partition="train", languages=["go"]))

    assert len(validation) == 2 and python_validation < validation
    assert all(name.startswith("py-") for name in python_validation)
    assert len(go_training) == 9 and all(name.startswith("go-") for name in go_training)


class _ScriptedRuntime:
    """Returns `test_log` as prime-envs' test output and records each command the
    task runs."""

    def __init__(self, test_log: str) -> None:
        self.test_log = test_log
        self.commands: list[tuple[list[str], dict[str, str]]] = []

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append((argv, env))
        if argv == ["cat", "/test_output.txt"]:
            return ProgramResult(exit_code=0, stdout=self.test_log, stderr="")
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def write(self, path: str, data: bytes) -> None:
        pass


# The commit setup recorded; HEAD may differ by scoring time, since the agent can
# commit.
BASE_AT_SETUP = "base-at-setup"


# What prime-envs' test script writes: its markers around the test command's output.
TEST_LOG = """SWEREBENCH_V2_TEST_OUTPUT_START
PASSED tests/test_app.py::test_fix
PASSED tests/test_app.py::test_old_a
FAILED tests/test_app.py::test_old_b - AssertionError
SWEREBENCH_V2_TEST_OUTPUT_END
"""


def test_scoring_restores_tests_from_the_recorded_commit(tmp_path) -> None:
    """The fix passes, but it breaks one of the two tests that passed before."""
    data = read_task_directory(_write_task(tmp_path, "t-1"), resources=RESOURCES).data
    runtime = _ScriptedRuntime(TEST_LOG)
    trace = SimpleNamespace(info={BASE_COMMIT_KEY: BASE_AT_SETUP}, metrics={})
    trace.record_metrics = trace.metrics.update

    score = asyncio.run(
        SWERebenchV21Task(data, vf.TaskConfig()).solved(runtime=runtime, trace=trace)
    )

    assert score == 0.75
    restore_env = next(env for argv, env in runtime.commands if "base" in env)
    assert restore_env["base"] == BASE_AT_SETUP
    assert trace.metrics == {
        "resolved": 0.0,
        "fail_to_pass_fraction": 1.0,
        "pass_to_pass_fraction": 0.5,
        "parsed_tests": 3.0,
    }


@pytest.mark.parametrize(
    "pass_to_pass",
    [
        # One of two FAIL_TO_PASS tests passes; the one PASS_TO_PASS test still passes.
        ["tests/test_app.py::test_old_a"],
        # No PASS_TO_PASS tests: only FAIL_TO_PASS counts.
        [],
    ],
)
def test_verifiers_scores_the_task_with_its_one_reward(
    tmp_path, pass_to_pass: list[str]
) -> None:
    """Score through Verifiers, which finds rewards by their decorator and name."""
    fail_to_pass = ["tests/test_app.py::test_fix", "tests/test_app.py::test_fix_2"]
    task_dir = _write_task(
        tmp_path, "t-1", fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass
    )
    data = read_task_directory(task_dir, resources=RESOURCES).data
    task = SWERebenchV21Task(data, vf.TaskConfig())
    trace = Trace(
        task=TraceTask(
            type="SWERebenchV21Task", data=data, key=task.key, hash=task.hash
        ),
        state=state_cls(type(task))(),
        agent=vf.AgentInfo(config=vf.AgentConfig(), name="test", trainable=False),
    )
    trace.info[BASE_COMMIT_KEY] = BASE_AT_SETUP
    test_log = (
        "SWEREBENCH_V2_TEST_OUTPUT_START\n"
        "PASSED tests/test_app.py::test_fix\n"
        "FAILED tests/test_app.py::test_fix_2 - AssertionError\n"
        "PASSED tests/test_app.py::test_old_a\n"
        "SWEREBENCH_V2_TEST_OUTPUT_END\n"
    )

    asyncio.run(task.score(trace, _ScriptedRuntime(test_log)))

    # The bridge trains on trace.reward.
    assert {name: r.score for name, r in trace.rewards.items()} == {"solved": 0.5}
    assert trace.reward == 0.5


def test_a_crashed_grader_fails_scoring_instead_of_scoring_zero() -> None:
    """A grader exits 1 both for an unresolved task and on a crash."""
    with pytest.raises(RuntimeError, match="grader failed"):
        asyncio.run(
            run_grader(
                "raise SystemExit(1)",
                fail_to_pass=FAIL_TO_PASS,
                pass_to_pass=PASS_TO_PASS,
                test_log="",
            )
        )


def test_cancelling_scoring_stops_the_grader(tmp_path) -> None:
    """Verifiers' scoring timeout cancels scoring; the grader must not outlive it."""
    pid_path = tmp_path / "grader.pid"
    grader = (
        "import os, pathlib, time\n"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(600)\n"
    )

    async def cancel_while_grading() -> None:
        grading = asyncio.ensure_future(
            run_grader(
                grader,
                fail_to_pass=FAIL_TO_PASS,
                pass_to_pass=PASS_TO_PASS,
                test_log="",
            )
        )
        while not (pid_path.exists() and pid_path.read_text()):
            await asyncio.sleep(0.05)
        grading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await grading

    asyncio.run(cancel_while_grading())
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_path.read_text()), 0)
