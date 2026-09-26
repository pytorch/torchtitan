# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Load a frozen Harbor task tree without copying its tests into training data."""

import json
import re
from collections.abc import Iterator
from pathlib import Path

import verifiers.v1 as vf

from pydantic import Field
from verifiers.v1.errors import TaskError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.tasksets.harbor import (
    HarborConfig,
    HarborEnv,
    HarborTask,
    HarborTaskset,
)
from verifiers.v1.tasksets.harbor.taskset import make_tar, parse_task

from torchtitan.rl.experiments.verifiers.terminal_bench.harness import (
    register_harness_alias,
)

register_harness_alias()


def _dockerfile_workdir(task_dir: Path) -> str | None:
    """Recover the published image's final WORKDIR when task.toml omits it."""
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.is_file():
        return None
    workdir = None
    for line in dockerfile.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*WORKDIR\s+(\S+)", line)
        if match:
            workdir = match.group(1)
    return workdir


class TerminalTasksetConfig(HarborConfig):
    """Point Verifiers at a local, versioned Harbor task tree."""

    tasks_root: Path
    expected_num_tasks: int | None = Field(default=None, gt=0)
    image_overrides_path: Path | None = None
    dataset: str = "local/terminal-tasks"
    require_image: bool = True
    ignore_timeouts: bool = True


class TerminalTask(HarborTask):
    """Stage Harbor tests without restoring an unmapped host UID in Docker."""

    async def setup(self, runtime: Runtime) -> None:
        await super().setup(runtime)
        probe = await runtime.run(["sh", "-c", "command -v tmux >/dev/null 2>&1"], {})
        if probe.exit_code:
            raise TaskError(
                f"task {self.data.name!r} image {self.data.image!r} lacks tmux; "
                "build a pullable image with tmux and provide an image override"
            )

    async def _stage_tests(self, runtime: Runtime, wipe: bool = False) -> None:
        await runtime.write(
            "/tmp/tests.tgz", make_tar(Path(self.data.task_dir) / "tests")
        )
        stage = (
            f"{'rm -rf /tests && ' if wipe else ''}"
            "rm -f /logs/verifier/reward.json /logs/verifier/reward.txt && "
            "mkdir -p /logs/verifier /tests && "
            "tar --no-same-owner -xzf /tmp/tests.tgz -C /tests"
        )
        result = await runtime.run(["sh", "-c", stage], {})
        if result.exit_code:
            raise TaskError(
                f"staging tests failed (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )


class TerminalTaskset(HarborTaskset, vf.Taskset[TerminalTask, TerminalTasksetConfig]):
    """Reuse Verifiers' Harbor task setup, in-place verifier, and reward parser."""

    config: TerminalTasksetConfig

    def load(self) -> Iterator[TerminalTask]:
        root = self.config.tasks_root.resolve()
        if not root.is_dir():
            raise ValueError(f"Harbor tasks_root is not a directory: {root}")
        task_dirs = [
            path.parent
            for path in sorted(root.rglob("task.toml"))
            if (path.parent / "instruction.md").is_file()
            and (self.config.tasks is None or path.parent.name in self.config.tasks)
        ]
        if not task_dirs:
            raise ValueError(f"No Harbor tasks found under {root}")
        if (
            self.config.expected_num_tasks is not None
            and len(task_dirs) != self.config.expected_num_tasks
        ):
            raise ValueError(
                f"Expected {self.config.expected_num_tasks} Harbor tasks under {root}, "
                f"found {len(task_dirs)}"
            )
        images: dict[str, str] = {}
        if self.config.image_overrides_path is not None:
            images = json.loads(self.config.image_overrides_path.read_text())
            if not isinstance(images, dict) or any(
                not isinstance(name, str) or not isinstance(image, str) or not image
                for name, image in images.items()
            ):
                raise ValueError("image overrides must map task names to image refs")
        parse_config = (
            self.config.model_copy(
                update={"require_image": False, "ignore_dockerfile": True}
            )
            if self.config.image_overrides_path is not None
            else self.config
        )
        for idx, task_dir in enumerate(task_dirs):
            if self.config.image_overrides_path is not None:
                if task_dir.name not in images:
                    raise ValueError(f"No image override for task {task_dir.name!r}")
            task_data = parse_task(task_dir, idx, parse_config)
            if self.config.image_overrides_path is not None:
                task_data = task_data.model_copy(
                    update={"image": images[task_dir.name]}
                )
            if task_data.workdir is None and (workdir := _dockerfile_workdir(task_dir)):
                task_data = task_data.model_copy(update={"workdir": workdir})
            yield TerminalTask(task_data, self.config.task)


# Verifiers discovers the Harbor environment from this taskset plugin. The
# environment handles tasks with a separate verifier as well as shared grading.
__all__ = ["TerminalTaskset", "HarborEnv"]
