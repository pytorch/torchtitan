# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verifiers' Harbor taskset for Terminal-Bench and TMax."""

from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf
from verifiers.v1.tasksets.harbor import HarborConfig, HarborTask, HarborTaskset


class TerminalTasksetConfig(HarborConfig):
    """Config of ``TerminalTaskset``: Harbor's fields, unchanged."""


class TerminalTaskset(HarborTaskset, vf.Taskset[HarborTask, TerminalTasksetConfig]):
    """Verifiers' Harbor taskset, with each task's image WORKDIR and Harbor's verifier timeout.

    Example, a task.toml with no workdir and no verifier timeout, and ignore_timeouts=False:
        Verifiers' taskset:  workdir None (the runtime starts in /app), no grading timeout
        this taskset:        workdir "/app/dclm" (its Dockerfile), grading timeout 600 s
    """

    config: TerminalTasksetConfig

    # TODO: delete this once Verifiers falls back to the image's WORKDIR and Harbor's 600 s verifier
    # timeout for a task.toml without them. Today it reads only task.toml:
    # https://github.com/PrimeIntellect-ai/verifiers/blob/v0.4.0/verifiers/v1/tasksets/harbor/taskset.py#L700-L712
    def load(self) -> Iterator[HarborTask]:
        # Harbor is an optional Verifiers extra: import it only when tasks load.
        from harbor.environments.definition import parse_dockerfile_workdir
        from harbor.models.task.config import VerifierConfig

        default_scoring = VerifierConfig.model_fields["timeout_sec"].default
        for task in super().load():
            dockerfile = Path(task.data.task_dir) / "environment" / "Dockerfile"
            image_workdir = parse_dockerfile_workdir(dockerfile)
            # Harbor's parser keeps the quotes and $VARs that Docker strips or expands: skip those.
            if image_workdir and any(char in image_workdir for char in "\"'$"):
                image_workdir = None
            workdir = task.data.workdir or image_workdir
            timeout = task.data.timeout
            if not self.config.ignore_timeouts and timeout.scoring is None:
                # Scaled like a declared timeout.
                scoring = default_scoring * self.config.timeout_multiplier
                timeout = timeout.model_copy(update={"scoring": scoring})
            yield HarborTask(
                task.data.model_copy(update={"workdir": workdir, "timeout": timeout}),
                self.config.task,
            )
