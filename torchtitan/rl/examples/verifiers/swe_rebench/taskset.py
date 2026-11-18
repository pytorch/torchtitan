# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SWE-rebench V2.1 tasks, built on prime-envs' SWE-rebench V2 task.

Each task directory of the V2.1 Harbor dataset holds one GitHub issue: the
task's Docker image, the issue text, the test command and log parser, a
held-out test patch, the FAIL_TO_PASS and PASS_TO_PASS test names and the gold
patch. This module turns each directory into a subclass of prime-envs'
``SWERebenchV2Task``. At scoring time that task restores the files the test
patch touches, applies it and runs the test command in the task container; the
env-server worker then parses the log outside the container, so the agent
cannot alter the parser.

Verifiers' env-server workers are spawned processes that import the taskset by
module name and must not import torch, so the taskset lives in this torch-free
module rather than in the recipe.
"""

import hashlib
import json
import re
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf
from pydantic import Field
from swerebench_v2 import log_parsers
from swerebench_v2.taskset import (
    ENV,
    is_resolved,
    normalize_test_name,
    SOLVER_SYSTEM_PROMPT,
    SWERebenchV2Data,
    SWERebenchV2Task,
)
from verifiers.v1 import capture_patch
from verifiers.v1.tasksets.harbor import HarborConfig
from verifiers.v1.tasksets.harbor.taskset import dataset_dir as harbor_dataset_dir

DATASET = (
    "ibragim-badertdinov/swe-rebench-v2.1-rl"
    "@sha256:0dcba0d99e8c2bc7f1a8aefdd2e3cf8f5aae2387659d6390a06446ed5f3612b3"
)
"""The SWE-rebench V2.1 RL Harbor dataset, pinned to one revision."""

ANSWER_KEY_DIR = "/swebench_matterhorn"
"""Some task images ship the reference solution here; setup deletes it."""

BASE_COMMIT_KEY = "swe_rebench_base_commit"
"""`trace.info` key for the repository commit the agent started from."""

# V2.1's tests/test.sh carries the test command in this heredoc.
_TEST_COMMAND = re.compile(
    r"<<'HARBOR_TEST_CMD_EOF'\n(.*?)\nHARBOR_TEST_CMD_EOF\n", re.DOTALL
)


class SWERebenchV21TaskConfig(vf.TaskConfig):
    reward_mode: Literal["f2p_regression", "resolved"] = "f2p_regression"
    """`resolved`: 1.0 when every FAIL_TO_PASS and PASS_TO_PASS test passes, else
    0.0 (V2.1's criterion). `f2p_regression` gives partial credit:
    f2p * (0.5 + 0.5 * p2p), where f2p and p2p are the pass fractions of the
    FAIL_TO_PASS and PASS_TO_PASS tests. Each broken PASS_TO_PASS test lowers the
    reward; breaking all of them halves it."""


def parse_test_statuses(test_output: str, log_parser: str) -> dict[str, str]:
    """Map each test name in the log to its status, e.g. ``"PASSED"``. An empty log
    (the test patch did not apply) yields no statuses."""
    parser = log_parsers.NAME_TO_PARSER[log_parser]
    return {
        normalize_test_name(name): status
        for name, status in parser(test_output).items()
    }


def pass_fraction(statuses: dict[str, str], test_names: list[str]) -> float:
    """Fraction of `test_names` that passed; 1.0 when there are none."""
    unique = {normalize_test_name(name) for name in test_names}
    if not unique:
        return 1.0
    passed = log_parsers.TestStatus.PASSED.value
    return sum(statuses.get(name) == passed for name in unique) / len(unique)


class SWERebenchV21Task(
    SWERebenchV2Task,
    vf.Task[SWERebenchV2Data, vf.State, SWERebenchV21TaskConfig],
):
    """prime-envs' SWE-rebench V2 task, adapted to V2.1 task directories.

    V2.1 tasks carry no base commit, which prime-envs needs to restore the test
    patch's files before applying it and to diff the agent's changes. The image's
    HEAD is that commit, so setup records it before the agent can move HEAD.
    Setup also deletes the answer key some images ship, and scoring adds the
    `f2p_regression` reward mode and per-test metrics.
    """

    @property
    def key(self) -> str:
        # The default key hashes all task data, including idx (the position after
        # this taskset's filters) and the resources; the instance id stays the
        # same when either changes.
        return self.data.name

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
        return metrics["f2p_fraction"] * (0.5 + 0.5 * metrics["p2p_fraction"])

    async def validate(self, runtime: vf.Runtime) -> bool:
        # Verifiers calls this with the runtime only, right after setup, so HEAD is
        # still the base commit.
        base_commit = await self._read_head(runtime)
        await self.apply_gold_patch(runtime)
        metrics = await self._grade(runtime, base_commit=base_commit)
        return metrics["resolved"] == 1.0

    async def _read_head(self, runtime: vf.Runtime) -> str:
        head = await runtime.run(["git", "rev-parse", "HEAD"], ENV)
        if head.exit_code != 0 or not head.stdout.strip():
            raise RuntimeError(
                f"cannot read the base commit of {self.data.name}: {head.stderr[-500:]}"
            )
        return head.stdout.strip()

    async def _grade(
        self, runtime: vf.Runtime, *, base_commit: str
    ) -> dict[str, float]:
        """Grade the current tree: restore the test patch's files to `base_commit`,
        apply the test patch, run the tests, and return `resolved`, the FAIL_TO_PASS
        and PASS_TO_PASS pass fractions and the number of parsed tests."""
        # prime-envs reads the base commit from the task data.
        test_output = await self.with_data(base_commit=base_commit).run_tests(runtime)
        statuses = parse_test_statuses(
            test_output, self.data.install_config["log_parser"]
        )
        resolved = bool(statuses) and is_resolved(
            statuses, self.data.fail_to_pass, self.data.pass_to_pass
        )
        return {
            "resolved": float(resolved),
            "f2p_fraction": pass_fraction(statuses, self.data.fail_to_pass),
            "p2p_fraction": pass_fraction(statuses, self.data.pass_to_pass),
            "parsed_tests": float(len(statuses)),
        }


@dataclass(frozen=True, slots=True)
class TaskDirectory:
    """One V2.1 task directory: the task data plus the repository and quality flags
    the taskset filters on."""

    repo: str
    quality_flags: list[str]
    data: SWERebenchV2Data


def read_task_directory(
    task_dir: Path, *, resources: vf.TaskResources
) -> TaskDirectory:
    """Read one V2.1 Harbor task directory."""
    task_toml = tomllib.loads((task_dir / "task.toml").read_text())
    metadata = task_toml["metadata"]
    environment = task_toml["environment"]
    log_parser = metadata["log_parser"]
    if log_parser not in log_parsers.NAME_TO_PARSER:
        raise ValueError(f"{task_dir.name}: unknown log parser {log_parser!r}")
    match = _TEST_COMMAND.search((task_dir / "tests" / "test.sh").read_text())
    if match is None:
        raise ValueError(f"{task_dir.name}: tests/test.sh carries no test command")
    tests = json.loads((task_dir / "tests" / "config.json").read_text())
    return TaskDirectory(
        repo=metadata["repo"],
        quality_flags=list(metadata.get("quality_flags", [])),
        data=SWERebenchV2Data(
            name=task_dir.name,
            prompt=(task_dir / "instruction.md").read_text(),
            system_prompt=SOLVER_SYSTEM_PROMPT,
            image=environment["docker_image"],
            workdir=environment["workdir"],
            resources=resources,
            install_config={"test_cmd": [match.group(1)], "log_parser": log_parser},
            test_patch=(task_dir / "tests" / "test.patch").read_text(),
            gold_patch=(task_dir / "solution" / "gold.patch").read_text(),
            fail_to_pass=tests["FAIL_TO_PASS"],
            pass_to_pass=tests["PASS_TO_PASS"],
        ),
    )


def project_name(repo: str) -> str:
    """The repository name without its owner. 20 V2.1 projects appear under two
    owners after a rename (e.g. pypa/pipx and pipxproject/pipx); the split keys on
    this name so both land on the same side."""
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

    exclude_quality_flags: list[str] = Field(default_factory=lambda: ["weak_f2p"])
    """Skip tasks that carry any of these V2.1 `quality_flags`. `weak_f2p` marks
    tasks whose FAIL_TO_PASS tests barely check the fix."""

    resources: vf.TaskResources = vf.TaskResources(cpu=4, memory=7, disk=10)
    """CPU cores and GB of memory and disk per sandbox. Each V2.1 task.toml asks for
    6 CPUs, 24 GB memory and 32 GB disk; these smaller defaults fit more sandboxes
    on one host. Raise memory if tests run out of it."""


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
        validation = validation_projects(
            {project_name(task_dir.repo) for task_dir in task_dirs},
            fraction=config.validation_project_fraction,
        )
        index = 0
        for task_dir in task_dirs:
            if set(config.exclude_quality_flags) & set(task_dir.quality_flags):
                continue
            in_validation = project_name(task_dir.repo) in validation
            if in_validation != (config.partition == "validation"):
                continue
            yield SWERebenchV21Task(
                task_dir.data.model_copy(update={"idx": index}), config.task
            )
            index += 1


# Verifiers loads the one Taskset subclass this module exports.
__all__ = ["SWERebenchV21Taskset"]
