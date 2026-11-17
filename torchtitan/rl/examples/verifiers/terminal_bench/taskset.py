# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verifiers' Harbor taskset, declared locally so each rollout worker imports us.

A rollout worker is a spawned process, and it only imports the module behind a
locally registered taskset alias (see `_local_taskset_module` in
`torchtitan/rl/examples/verifiers/rollouter.py`). Importing this module
registers the harness alias in that process too, which Verifiers needs to
resolve the harness id.
"""

import socket
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import (
    HarborConfig,
    HarborData,
    HarborTask,
    HarborTaskset,
)

# TODO: import `HarborTaskConfig` from a public Verifiers module once one exports it. A Verifiers bump may move it.
from verifiers.v1.tasksets.harbor.taskset import HarborTaskConfig

from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    register_harness_alias,
)

register_harness_alias()


class TerminalTasksetConfig(HarborConfig):
    """Harbor taskset config; a local class so its module is the worker's entry."""


class TerminalTask(HarborTask):
    """A Harbor task that refuses to build on a host without the `tests/` its grading reads."""

    def __init__(
        self, data: HarborData, config: HarborTaskConfig | None = None
    ) -> None:
        # A rollout worker builds the task on its own host. Grading copies `tests/` from
        # `data.task_dir` on that host into the sandbox. A missing dir would end as a TaskError,
        # which trains as the policy's failure. So fail the group here, before any sandbox starts.
        # A task whose separate verifier image carries its own /tests needs no local `tests/`. In
        # Verifiers 0.4.1, `parse_task` sets `verifier_image` only for such a task, and
        # `HarborTask.stage_tests` then skips the upload.
        tests_dir = Path(data.task_dir) / "tests"
        if data.verifier_image is None and not tests_dir.is_dir():
            raise RuntimeError(
                f"{tests_dir} is missing on host {socket.gethostname()}. Please make the "
                "Harbor cache available at the same path on every host, or set "
                "rollouter.worker_roles.generator=None and rollouter.worker_roles.controller=True."
            )
        super().__init__(data, config)


class TerminalTaskset(HarborTaskset, vf.Taskset[TerminalTask, TerminalTasksetConfig]):
    """Verifiers' Harbor taskset. A rollout worker builds its tasks as `TerminalTask`.

    The worker gets that class from `task_type()`, which reads the type parameter above. `load()`
    runs on the controller and builds plain `HarborTask`s, so only rollout workers check `tests/`.
    """

    config: TerminalTasksetConfig


__all__ = ["TerminalTaskset"]
