# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from torchtitan.experiments.rl.examples.terminal_bench.data import _image_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build work and verifier images for Terminal-Bench v1 tasks."
    )
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--image-prefix", default="torchtitan-terminal-bench")
    parser.add_argument("tasks", nargs="+")
    args = parser.parse_args()

    for task in args.tasks:
        task_dir = args.tasks_dir / task
        image_name = _image_name(args.image_prefix, task)
        _build(
            context=task_dir / "environment",
            tag=f"{image_name}-work:latest",
        )
        _build(
            context=task_dir / "tests",
            tag=f"{image_name}-verifier:latest",
        )


def _build(*, context: Path, tag: str) -> None:
    if not (context / "Dockerfile").is_file():
        raise FileNotFoundError(context / "Dockerfile")
    subprocess.run(
        ["docker", "build", "--tag", tag, str(context)],
        check=True,
    )


if __name__ == "__main__":
    main()
