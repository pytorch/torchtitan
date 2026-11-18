# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the SWE-rebench V2.1 Verifiers example and recipes."""

import asyncio
import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers")
pytest.importorskip("swerebench_v2")

import verifiers.v1 as vf

from torchtitan.config import ConfigLoader
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.examples.verifiers import env_server
from torchtitan.rl.examples.verifiers.swe_rebench import taskset
from torchtitan.rl.examples.verifiers.swe_rebench.taskset import (
    ANSWER_KEY_DIR,
    BASE_COMMIT_KEY,
    parse_test_statuses,
    read_task_directory,
    SWERebenchV21Task,
    SWERebenchV21TaskConfig,
    SWERebenchV21Taskset,
    SWERebenchV21TasksetConfig,
)
from verifiers.v1.harnesses import BashHarnessConfig
from verifiers.v1.runtimes import ProgramResult
from verifiers.v1.serve import env_config_data

FAIL_TO_PASS = ["tests/test_app.py::test_fix"]
PASS_TO_PASS = ["tests/test_app.py::test_old_a", "tests/test_app.py::test_old_b"]
RESOURCES = vf.TaskResources(cpu=1, memory=1, disk=1)


def _write_task(
    root: Path,
    name: str,
    *,
    repo: str = "owner/app",
    quality_flags: tuple[str, ...] = (),
    log_parser: str = "parse_log_pytest",
) -> Path:
    """Write one task directory laid out like SWE-rebench V2.1's Harbor export."""
    task_dir = root / name
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    flags = ", ".join(f'"{flag}"' for flag in quality_flags)
    (task_dir / "task.toml").write_text(
        f"""[task]
name = "org/{name}"

[metadata]
repo = "{repo}"
language = "python"
log_parser = "{log_parser}"
quality_flags = [{flags}]

[environment]
workdir = "/app"
docker_image = "docker.io/swerebenchv2/app:1-abc@sha256:0123"
"""
    )
    (task_dir / "instruction.md").write_text(f"Fix {name}.")
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/bash\n"
        "IFS= read -r -d '' HARBOR_TEST_CMD <<'HARBOR_TEST_CMD_EOF'\n"
        "pytest -rA tests/test_app.py\n"
        "HARBOR_TEST_CMD_EOF\n"
        'cd "$WORKDIR"\n'
    )
    # V2.1 stores some name lists as the Python repr of a list.
    (task_dir / "tests" / "config.json").write_text(
        json.dumps({"FAIL_TO_PASS": FAIL_TO_PASS, "PASS_TO_PASS": str(PASS_TO_PASS)})
    )
    (task_dir / "tests" / "test.patch").write_text("diff --git a/tests/test_app.py\n")
    (task_dir / "solution" / "gold.patch").write_text("diff --git a/app.py\n")
    return task_dir


def test_a_task_directory_becomes_a_prime_envs_task(tmp_path) -> None:
    task_dir = read_task_directory(_write_task(tmp_path, "t-1"), resources=RESOURCES)

    data = task_dir.data
    assert (task_dir.repo, data.name, data.prompt) == ("owner/app", "t-1", "Fix t-1.")
    assert data.image == "docker.io/swerebenchv2/app:1-abc@sha256:0123"
    assert data.workdir == "/app"
    assert data.install_config == {
        "test_cmd": ["pytest -rA tests/test_app.py"],
        "log_parser": "parse_log_pytest",
    }
    assert (data.fail_to_pass, data.pass_to_pass) == (FAIL_TO_PASS, PASS_TO_PASS)
    assert data.test_patch.startswith("diff --git a/tests/")
    assert data.gold_patch.startswith("diff --git a/app.py")
    # V2.1 has no base commit; setup reads it from the image.
    assert data.base_commit == ""


def test_an_unknown_log_parser_fails_at_load(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown log parser"):
        read_task_directory(
            _write_task(tmp_path, "t-1", log_parser="parse_log_nothing"),
            resources=RESOURCES,
        )


def _load(root: Path, **options) -> list[SWERebenchV21Task]:
    config = SWERebenchV21TasksetConfig(dataset_dir=root, **options)
    return list(SWERebenchV21Taskset(config).load())


def test_training_and_validation_never_share_a_repository(tmp_path) -> None:
    for index in range(20):
        _write_task(tmp_path, f"t-{index}", repo=f"owner/repo-{index % 10}")
    _write_task(tmp_path, "weak", repo="owner/repo-0", quality_flags=("weak_f2p",))

    def repos(tasks: list[SWERebenchV21Task]) -> set[str]:
        return {
            read_task_directory(tmp_path / task.data.name, resources=RESOURCES).repo
            for task in tasks
        }

    training = _load(tmp_path, partition="train")
    validation = _load(tmp_path, partition="validation")
    assert repos(training).isdisjoint(repos(validation))
    assert len(training) + len(validation) == 20
    assert "weak" not in {task.data.name for task in training + validation}
    assert [task.data.idx for task in training] == list(range(len(training)))


def test_a_task_list_selects_its_tasks_and_fails_on_a_missing_one(tmp_path) -> None:
    for index in range(3):
        _write_task(tmp_path, f"t-{index}")
    listed = tmp_path / "list.txt"
    listed.write_text("t-0  # a comment\nt-2\n")
    assert [
        t.data.name for t in _load(tmp_path, partition="all", task_list=listed)
    ] == [
        "t-0",
        "t-2",
    ]

    with pytest.raises(ValueError, match="partition='all'"):
        _load(tmp_path, partition="train", task_list=listed)
    listed.write_text("t-0\nt-9\n")
    with pytest.raises(ValueError, match="t-9"):
        _load(tmp_path, partition="all", task_list=listed)


class _ScriptedRuntime:
    """Answers each command the task runs, and records them in order."""

    def __init__(self, test_log: str = "") -> None:
        self.test_log = test_log
        self.commands: list[tuple[list[str], dict[str, str]]] = []

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.commands.append((argv, env))
        if argv == ["git", "rev-parse", "HEAD"]:
            return ProgramResult(exit_code=0, stdout="abc123\n", stderr="")
        if argv == ["cat", "/test_output.txt"]:
            return ProgramResult(exit_code=0, stdout=self.test_log, stderr="")
        if ANSWER_KEY_DIR in " ".join(argv):
            return ProgramResult(exit_code=0, stdout="removed\n", stderr="")
        return ProgramResult(exit_code=0, stdout="", stderr="")

    async def write(self, path: str, data: bytes) -> None:
        pass


def _trace() -> SimpleNamespace:
    trace = SimpleNamespace(info={}, metrics={})
    trace.record_metrics = trace.metrics.update
    return trace


def _task(tmp_path: Path, reward_mode: str = "f2p_regression") -> SWERebenchV21Task:
    data = read_task_directory(_write_task(tmp_path, "t-1"), resources=RESOURCES).data
    return SWERebenchV21Task(data, SWERebenchV21TaskConfig(reward_mode=reward_mode))


def test_setup_removes_the_answer_key_and_records_the_base_commit(tmp_path) -> None:
    runtime, trace = _ScriptedRuntime(), _trace()

    asyncio.run(_task(tmp_path).setup(trace=trace, runtime=runtime))

    assert trace.info[BASE_COMMIT_KEY] == "abc123"
    assert trace.metrics["answer_key_present"] == 1.0
    assert "rm -rf /swebench_matterhorn" in " ".join(runtime.commands[0][0])


TEST_LOG = """setup noise PASSED tests/test_app.py::test_old_b
SWEREBENCH_V2_TEST_OUTPUT_START
PASSED tests/test_app.py::test_fix
PASSED tests/test_app.py::test_old_a
FAILED tests/test_app.py::test_old_b - AssertionError
SWEREBENCH_V2_TEST_OUTPUT_END
"""


def test_only_the_marked_region_of_the_test_log_is_parsed() -> None:
    statuses = parse_test_statuses(TEST_LOG, {"log_parser": "parse_log_pytest"})
    assert statuses == {
        "tests/test_app.py::test_fix": "PASSED",
        "tests/test_app.py::test_old_a": "PASSED",
        "tests/test_app.py::test_old_b": "FAILED",
    }


@pytest.mark.parametrize(
    ("reward_mode", "reward"), [("f2p_regression", 0.75), ("resolved", 0.0)]
)
def test_scoring_restores_tests_from_the_recorded_commit(
    tmp_path, reward_mode: str, reward: float
) -> None:
    """The fix passes, but it breaks one of the two tests that passed before."""
    runtime, trace = _ScriptedRuntime(TEST_LOG), _trace()
    trace.info[BASE_COMMIT_KEY] = "abc123"

    score = asyncio.run(
        _task(tmp_path, reward_mode).solved(runtime=runtime, trace=trace)
    )

    assert score == reward
    restore_env = next(env for argv, env in runtime.commands if "base" in env)
    assert restore_env["base"] == "abc123"
    assert trace.metrics == {
        "resolved": 0.0,
        "f2p_fraction": 1.0,
        "p2p_fraction": 0.5,
        "parsed_tests": 3.0,
    }


@pytest.mark.parametrize(
    ("test_log", "valid"),
    [(TEST_LOG.replace("FAILED", "PASSED"), True), (TEST_LOG, False)],
    ids=["all_tests_pass", "a_passing_test_breaks"],
)
def test_the_gold_check_reads_the_base_commit_itself(
    tmp_path, test_log: str, valid: bool
) -> None:
    """Verifiers' gold check passes only the runtime, so there is no trace to read."""
    runtime = _ScriptedRuntime(test_log)

    assert asyncio.run(_task(tmp_path).validate(runtime)) is valid

    argvs = [argv for argv, _ in runtime.commands]
    assert argvs.index(["git", "rev-parse", "HEAD"]) == 0
    restore_env = next(env for argv, env in runtime.commands if "base" in env)
    assert restore_env["base"] == "abc123"


def _recipe(name: str) -> Controller.Config:
    return ConfigLoader().load(
        ["--module", "torchtitan_recipes.rl.verifiers_swe_rebench", "--config", name]
    )


@pytest.mark.parametrize(
    "name", ["rl_grpo_qwen38_27b_swe_rebench", "rl_grpo_qwen35_9b_swe_rebench"]
)
def test_recipes_validate_on_repositories_they_do_not_train_on(name: str) -> None:
    rollouter = _recipe(name).rollouter
    training = rollouter.training_dataloader.dataset.verifiers_taskset
    validation = rollouter.validation_dataset.verifiers_taskset
    environment = rollouter.verifiers_env_server.environment
    assert (training.partition, validation.partition) == ("train", "validation")
    assert environment.taskset == training
    assert isinstance(environment.agent.harness, BashHarnessConfig)
    assert environment.agent.harness.compaction is None
    assert isinstance(environment.agent.runtime, vf.DockerConfig)
    assert rollouter.verifiers_env_server.local_taskset_module == taskset.__name__


def test_env_server_workers_load_the_swe_rebench_env(monkeypatch) -> None:
    """A pool worker runs the pickled `log_setup`, then resolves the env config, whose
    taskset id is a local alias that only this test's process registered. It must find
    the taskset and the harness without importing torch, vLLM or Monarch.
    """
    captured = {}

    def serve_env(**kwargs) -> None:
        captured["log_setup"] = pickle.dumps(kwargs["log_setup"])
        captured["config_data"] = pickle.dumps(kwargs["config_data"])

    monkeypatch.setattr(env_server, "serve_env", serve_env)
    # _run_env_server_process replaces __main__ and edits no_proxy; restore both after the test.
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    config = _recipe("rl_grpo_qwen35_9b_swe_rebench").rollouter.verifiers_env_server
    env_server._run_env_server_process(
        environment=env_config_data(config.environment),
        serve=config.serve,
        local_taskset_module=config.local_taskset_module,
        address_queue=None,
        death_pipe=None,
    )

    worker = f"""
import pickle, sys
from verifiers.v1.utils.loaders import load_harness, load_taskset, resolve_env_config
pickle.loads({captured["log_setup"]!r})()
env_config = resolve_env_config(pickle.loads({captured["config_data"]!r}))
print(type(load_taskset(env_config.taskset)).__name__)
print(type(load_harness(env_config.agent.harness)).__name__)
print([name for name in ("torch", "vllm", "monarch") if name in sys.modules])
"""
    result = subprocess.run(
        [sys.executable, "-c", worker], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split() == ["SWERebenchV21Taskset", "BashHarness", "[]"]
