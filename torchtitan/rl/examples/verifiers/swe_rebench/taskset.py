# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE-rebench V2.1 tasks, built on prime-envs' SWE-rebench V2 task.

Each task directory of the V2.1 Harbor dataset holds one GitHub issue: the
task's Docker image, the issue text, the test command, a held-out test patch,
the FAIL_TO_PASS and PASS_TO_PASS test names, the gold patch and the task's
grader (``tests/grade.py``, V2.1's log parser and test-name matching for the
task's test framework). This module turns each directory into a
``SWERebenchV21Task``, a subclass of prime-envs' ``SWERebenchV2Task``. At
scoring time prime-envs restores the files the test patch touches to the base
commit and applies the test patch; the task runs the test command the way
V2.1's ``tests/test.sh`` does; the env-server worker then runs the task's
grader on the log outside the container, so the agent cannot alter it.

Verifiers' env-server workers are spawned processes that import the taskset by
module name, and the bridge keeps torch out of them, so the taskset lives in
this torch-free module rather than in the recipe.
"""

import asyncio
import functools
import hashlib
import json
import re
import shlex
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field
from swerebench_v2.taskset import (
    ENV,
    SOLVER_SYSTEM_PROMPT,
    SWERebenchV2Data,
    SWERebenchV2Task,
)
from verifiers.v1 import capture_patch
from verifiers.v1.tasksets.harbor import HarborConfig
from verifiers.v1.tasksets.harbor.taskset import dataset_dir as harbor_dataset_dir
from verifiers.v1.utils.git import resolve_head

DATASET = (
    "ibragim-badertdinov/swe-rebench-v2.1-rl"
    "@sha256:0dcba0d99e8c2bc7f1a8aefdd2e3cf8f5aae2387659d6390a06446ed5f3612b3"
)
"""The SWE-rebench V2.1 RL Harbor dataset, pinned to its revision 3: 13,337 tasks
in 20 languages, 3,990 of them Python.
TODO: move to v1.1.0 (revision 4, which adds 7,935 Python tasks from the first
SWE-rebench) after gold-checking it."""

ANSWER_KEY_DIR = "/swebench_matterhorn"
"""Some task images ship the reference solution here; setup deletes it."""

BASE_COMMIT_KEY = "swe_rebench_base_commit"
"""`trace.info` key for the repository commit the agent started from."""

# V2.1's tests/test.sh exports the task's test environment variables, if any,
# then reads the test command from this heredoc.
_TEST_COMMAND = re.compile(
    r"^((?:export \w+=[^\n]*\n)*)IFS= read -r -d '' HARBOR_TEST_CMD "
    r"<<'HARBOR_TEST_CMD_EOF'\n(.*?)\nHARBOR_TEST_CMD_EOF\n",
    re.DOTALL | re.MULTILINE,
)

# How V2.1's tests/test.sh runs the test command: in a login shell, whose PATH
# is the image's own rather than prime-envs' fixed one, with an 1800 s limit.
_RUN_TEST_COMMAND = "bash -lc 'timeout -k 10 1800 bash -lc \"$HARBOR_TEST_CMD\"'"

# prime-envs' test script prints these around the test command's output.
_TEST_OUTPUT_START = "SWEREBENCH_V2_TEST_OUTPUT_START"
_TEST_OUTPUT_END = "SWEREBENCH_V2_TEST_OUTPUT_END"


class SWERebenchV21TaskConfig(vf.TaskConfig):
    reward_mode: Literal["f2p_regression", "resolved"] = "f2p_regression"
    """`resolved`: 1.0 when every FAIL_TO_PASS and PASS_TO_PASS test passes, else
    0.0 (V2.1's criterion). `f2p_regression` gives partial credit:
    f2p * (0.5 + 0.5 * p2p), where f2p and p2p are the pass fractions of the
    FAIL_TO_PASS and PASS_TO_PASS tests. Each broken PASS_TO_PASS test lowers the
    reward; breaking all of them halves it. The env server builds every task,
    validation included, with the training taskset's value."""


class SWERebenchV21Data(SWERebenchV2Data):
    grader_source: str
    """The task's tests/grade.py."""

    container_env: dict[str, str] = Field(default_factory=dict)
    """task.toml's environment variables for the task container, e.g. Java options."""


def strip_script_markers(test_output: str) -> str:
    """The test command's output, without the lines prime-envs' script adds around
    it; the same log V2.1's tests/test.sh grades."""
    region = test_output.split(_TEST_OUTPUT_START + "\n", 1)[-1]
    return region.rsplit(_TEST_OUTPUT_END + "\n", 1)[0]


@functools.cache
def load_grader(grader_source: str) -> Callable[[dict, str], dict]:
    """Compile a task's tests/grade.py and return its `grade` function. Tasks with
    the same log parser ship the same grader, so each distinct grader loads once.

    What is specific to SWE-rebench V2.1: each task ships tests/grade.py, a
    standalone script that SWE-rebench-V2's lib/harbor_tasks generates for the
    task's log parser (there are 32 parsers). It defines
    ``grade(config, test_log)``, where `config` holds the FAIL_TO_PASS and
    PASS_TO_PASS lists, and returns the report `run_grader` describes. V2.1's
    tests/test.sh calls it through the script's command line
    (``grade.py CONFIG LOG REPORT``), which only reads and writes those files around
    `grade`; this calls `grade` directly. The script imports only the standard
    library and runs its command line only as ``__main__``.
    """
    namespace = {"__name__": "swe_rebench_v21_grader"}
    exec(compile(grader_source, "tests/grade.py", "exec"), namespace)
    return namespace["grade"]


async def run_grader(
    grader_source: str,
    *,
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    test_log: str,
) -> dict:
    """Grade a test log with a task's grader, as V2.1's tests/test.sh does, and
    return its report: `resolved`, `parsed_counts`, and `fail_to_pass` and
    `pass_to_pass` mapping each expected test name to whether it passed."""
    config = {"FAIL_TO_PASS": fail_to_pass, "PASS_TO_PASS": pass_to_pass}

    def grade() -> dict:
        return load_grader(grader_source)(config, test_log)

    try:
        # In a worker thread, so parsing a large log does not stall the event loop
        # that serves the env server's other rollouts.
        return await asyncio.to_thread(grade)
    except Exception as error:
        # The grader is the dataset's code: fail scoring rather than score 0.
        raise RuntimeError(f"the task's grader failed: {error!r}") from error


def pass_fraction(passed: dict[str, bool]) -> float:
    """Fraction of a grader report's tests that passed; 1.0 when there are none."""
    return sum(passed.values()) / len(passed) if passed else 1.0


class SWERebenchV21Task(
    SWERebenchV2Task,
    vf.Task[SWERebenchV21Data, vf.State, SWERebenchV21TaskConfig],
):
    """prime-envs' SWE-rebench V2 task, adapted to V2.1 task directories.

    V2.1 tasks carry no base commit, which prime-envs needs to restore the test
    patch's files before applying it and to diff the agent's changes. The image's
    HEAD is that commit, so setup records it before the agent can move HEAD.
    Setup also deletes the answer key some images ship. Scoring grades with the
    task's own V2.1 grader instead of prime-envs' parsers, adds the
    `f2p_regression` reward mode and records the FAIL_TO_PASS and PASS_TO_PASS
    pass fractions as metrics. The second base binds the data and config types;
    prime-envs' base binds only its own data type.
    """

    @property
    def key(self) -> str:
        # Identify the task by its V2.1 name (e.g. aio-libs__aiohttp-7889) on traces
        # and in rollout logs. The default hashes all task data, so it would change
        # with the task's position after filtering (idx) or with `resources`.
        return self.data.name

    def runtime_env(self) -> dict[str, str]:
        # Verifiers adds these to every process in the task container, the agent's
        # included, as Harbor does with task.toml's environment variables.
        return dict(self.data.container_env)

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        removed = await runtime.run(
            [
                "sh",
                "-c",
                f"if [ -e {ANSWER_KEY_DIR} ]; then rm -rf {ANSWER_KEY_DIR} && echo removed; fi",
            ],
            ENV,
        )
        if removed.exit_code != 0:
            raise RuntimeError(
                f"cannot remove {ANSWER_KEY_DIR}: {(removed.stderr or removed.stdout)[-500:]}"
            )
        trace.record_metrics({"answer_key_present": float("removed" in removed.stdout)})
        trace.info[BASE_COMMIT_KEY] = await self._read_head(runtime)
        await super().setup(runtime)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # prime-envs diffs against the task data's base commit, which V2.1 tasks
        # lack; without it Verifiers diffs against HEAD and misses commits the agent
        # made. Runs before scoring restores the test files, so the patch keeps the
        # agent's test edits.
        await capture_patch(
            trace, runtime, base_commit=trace.info[BASE_COMMIT_KEY], env=ENV
        )

    # Same name as prime-envs' reward, so it replaces that reward instead of adding a
    # second one.
    @vf.reward(weight=1.0)
    async def solved(self, runtime: vf.Runtime, trace: vf.Trace) -> float:
        metrics = await self._grade(runtime, base_commit=trace.info[BASE_COMMIT_KEY])
        trace.record_metrics(metrics)
        if self.config.reward_mode == "resolved":
            return metrics["resolved"]
        return metrics["fail_to_pass_fraction"] * (
            0.5 + 0.5 * metrics["pass_to_pass_fraction"]
        )

    async def validate(self, runtime: vf.Runtime) -> bool:
        # Verifiers calls this with the runtime only, right after setup, so HEAD is
        # still the base commit.
        base_commit = await self._read_head(runtime)
        await self.apply_gold_patch(runtime)
        metrics = await self._grade(runtime, base_commit=base_commit)
        return metrics["resolved"] == 1.0

    async def _read_head(self, runtime: vf.Runtime) -> str:
        head = await resolve_head(runtime, ENV)
        # An empty base makes prime-envs skip restoring the test files, so the
        # agent's edits to them would survive grading.
        if not head:
            raise RuntimeError(f"cannot read the base commit of {self.data.name}")
        return head

    async def _grade(
        self, runtime: vf.Runtime, *, base_commit: str
    ) -> dict[str, float]:
        """Grade the current tree: restore the test patch's files to `base_commit`,
        apply the test patch, run the tests, and return `resolved`, the FAIL_TO_PASS
        and PASS_TO_PASS pass fractions and the number of parsed tests."""
        # prime-envs reads the base commit from the task data. It returns an empty
        # log when the test patch does not apply, which grades as unresolved.
        test_output = await self.with_data(base_commit=base_commit).run_tests(runtime)
        report = await run_grader(
            self.data.grader_source,
            fail_to_pass=self.data.fail_to_pass,
            pass_to_pass=self.data.pass_to_pass,
            test_log=strip_script_markers(test_output),
        )
        return {
            "resolved": float(report["resolved"]),
            "fail_to_pass_fraction": pass_fraction(report["fail_to_pass"]),
            "pass_to_pass_fraction": pass_fraction(report["pass_to_pass"]),
            "parsed_tests": float(sum(report["parsed_counts"].values())),
        }


@dataclass(frozen=True, slots=True)
class TaskDirectory:
    """One V2.1 task directory: the task data plus the repository, language and
    quality flags the taskset filters on."""

    repo: str
    language: str
    quality_flags: list[str]
    data: SWERebenchV21Data


def read_task_directory(
    task_dir: Path, *, resources: vf.TaskResources
) -> TaskDirectory:
    """Read one V2.1 Harbor task directory."""
    task_toml = tomllib.loads((task_dir / "task.toml").read_text())
    metadata = task_toml["metadata"]
    environment = task_toml["environment"]
    match = _TEST_COMMAND.search((task_dir / "tests" / "test.sh").read_text())
    if match is None:
        raise ValueError(
            f"{task_dir.name}: tests/test.sh has no HARBOR_TEST_CMD heredoc"
        )
    test_env_exports, test_command = match.group(1).splitlines(), match.group(2)
    tests = json.loads((task_dir / "tests" / "config.json").read_text())
    return TaskDirectory(
        repo=metadata["repo"],
        language=metadata["language"],
        quality_flags=list(metadata.get("quality_flags", [])),
        data=SWERebenchV21Data(
            name=task_dir.name,
            prompt=(task_dir / "instruction.md").read_text(),
            system_prompt=SOLVER_SYSTEM_PROMPT,
            image=environment["docker_image"],
            workdir=environment["workdir"],
            resources=resources,
            # prime-envs runs each line of test_cmd in its test script.
            install_config={
                "test_cmd": [
                    *test_env_exports,
                    f"export HARBOR_TEST_CMD={shlex.quote(test_command)}",
                    _RUN_TEST_COMMAND,
                ],
                "log_parser": metadata["log_parser"],
            },
            # Read the patches as bytes: read_text() turns "\r\n" into "\n", and a
            # patch to a file with CRLF line endings then no longer applies.
            test_patch=(task_dir / "tests" / "test.patch").read_bytes().decode(),
            gold_patch=(task_dir / "solution" / "gold.patch").read_bytes().decode(),
            fail_to_pass=tests["FAIL_TO_PASS"],
            pass_to_pass=tests["PASS_TO_PASS"],
            grader_source=(task_dir / "tests" / "grade.py").read_text(),
            container_env=dict(environment.get("env", {})),
        ),
    )


def project_name(repo: str) -> str:
    """The repository name without its owner, lowercased. Some V2.1 projects appear
    under two owners after a rename (e.g. pypa/pipx and pipxproject/pipx); the
    train/validation split keys on this name so both land on the same side.
    Unrelated projects that share a name land together too."""
    return repo.split("/", 1)[1].lower()


def validation_projects(projects: set[str], *, fraction: float) -> set[str]:
    """Pick about `fraction` of the projects, at least one, ranked by the hash of
    their names so every run picks the same ones."""
    ranked = sorted(
        projects, key=lambda project: hashlib.sha256(project.encode()).digest()
    )
    return set(ranked[: max(1, round(fraction * len(ranked)))])


class SWERebenchV21TasksetConfig(vf.TasksetConfig):
    task: SWERebenchV21TaskConfig = SWERebenchV21TaskConfig()

    dataset: str = DATASET
    """Harbor Hub id of the task directories; Verifiers downloads it into
    ``~/.cache/harbor`` on first use."""

    dataset_dir: Path | None = None
    """A local copy of the task directories, e.g. on a shared filesystem; when set,
    `dataset` is not downloaded."""

    partition: Literal["train", "validation"] = "train"
    """V2.1 has no split, so this taskset splits the projects: no project is in both
    partitions, even under a renamed owner."""

    validation_project_fraction: float = Field(0.1, gt=0, lt=1)
    """Fraction of the projects held out for `partition="validation"`, rounded, at
    least one."""

    languages: list[str] | None = None
    """Keep only tasks in these languages (task.toml's `language`, e.g. "python",
    "go", "rust"); None keeps every language."""

    exclude_quality_flags: list[str] = Field(default_factory=lambda: ["weak_f2p"])
    """Skip tasks that carry any of these V2.1 `quality_flags`. `weak_f2p` marks
    tasks whose FAIL_TO_PASS tests barely check the fix."""

    resources: vf.TaskResources = vf.TaskResources(cpu=4, memory=7, disk=10)
    """CPU cores and GB of memory and disk per sandbox, below the 6 CPUs, 24 GB
    memory and 32 GB disk in every V2.1 task.toml. Docker applies CPU and memory as
    per-container caps, not reservations, and ignores disk; a runtime that reserves
    resources fits more sandboxes per host with these values. Raise memory if tests
    run out of it."""


class SWERebenchV21Taskset(vf.Taskset[SWERebenchV21Task, SWERebenchV21TasksetConfig]):
    """SWE-rebench V2.1's task directories as prime-envs SWE-rebench V2 tasks."""

    def load(self) -> Iterator[SWERebenchV21Task]:
        config = self.config
        root = config.dataset_dir or harbor_dataset_dir(
            HarborConfig(dataset=config.dataset)
        )
        task_dirs = [
            read_task_directory(toml_path.parent, resources=config.resources)
            for toml_path in sorted(root.rglob("task.toml"))
        ]
        if not task_dirs:
            raise ValueError(f"no SWE-rebench V2.1 tasks found in {root}")
        # Split all projects before filtering, so changing `languages` or
        # `exclude_quality_flags` does not move a project between partitions.
        validation = validation_projects(
            {project_name(task_dir.repo) for task_dir in task_dirs},
            fraction=config.validation_project_fraction,
        )
        for task_dir in task_dirs:
            if (
                config.languages is not None
                and task_dir.language not in config.languages
            ):
                continue
            if set(config.exclude_quality_flags) & set(task_dir.quality_flags):
                continue
            in_validation = project_name(task_dir.repo) in validation
            if in_validation != (config.partition == "validation"):
                continue
            yield SWERebenchV21Task(task_dir.data, config.task)


# Verifiers loads the one Taskset subclass this module exports.
__all__ = ["SWERebenchV21Taskset"]
