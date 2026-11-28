# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for the AI2 TMax -> Harbor converter, on a synthetic AI2-shaped dataset."""

import os
from pathlib import Path

import pytest

pytest.importorskip("verifiers")
pq = pytest.importorskip("pyarrow.parquet")

import pyarrow as pa

from torchtitan.rl.examples.verifiers.terminal_bench.prepare_tmax import convert_dataset
from verifiers.v1.tasksets.harbor import HarborConfig
from verifiers.v1.tasksets.harbor.taskset import parse_task

FIRST = "task_000001_aaaaaaaa"
NESTED = "task_000002_bbbbbbbb"
NO_TESTS = "task_000003_cccccccc"
NO_DIR = "task_000004_dddddddd"


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _make_dataset(root) -> tuple[str, str]:
    """Four parquet rows; two usable tasks (one nested a level down)."""
    rows = [
        (FIRST, "hamishi740/swerl-tmax-v3:0123456789ab"),
        (NESTED, "hamishi740/swerl-tmax-v3:ba9876543210"),
        (NO_TESTS, "hamishi740/swerl-tmax-v3:111111111111"),
        (NO_DIR, "hamishi740/swerl-tmax-v3:222222222222"),
    ]
    table = pa.table(
        {
            "ground_truth": [task_id for task_id, _ in rows],
            "env_config": [
                {"env_name": "tmax", "image": image, "task_id": task_id}
                for task_id, image in rows
            ],
        }
    )
    parquet = os.path.join(root, "data")
    os.makedirs(parquet)
    pq.write_table(table, os.path.join(parquet, "train-00000-of-00001.parquet"))

    tasks = os.path.join(root, "task-data")
    first = os.path.join(tasks, FIRST)
    _write(os.path.join(first, "instruction.md"), "Fix the parser in /app.\n")
    _write(os.path.join(first, "setup.sh"), "echo baked into the image\n")
    _write(
        os.path.join(first, "tests", "test.sh"), "echo 1 > /logs/verifier/reward.txt\n"
    )
    nested = os.path.join(tasks, "shard0", NESTED)
    _write(os.path.join(nested, "instruction.md"), "Write a script.\n")
    _write(
        os.path.join(nested, "tests", "test.sh"), "echo 0 > /logs/verifier/reward.txt\n"
    )
    _write(os.path.join(tasks, NO_TESTS, "instruction.md"), "No verifier.\n")
    return parquet, tasks


def test_conversion_keeps_ai2_files_and_skips_incomplete_tasks(tmp_path) -> None:
    parquet, tasks = _make_dataset(str(tmp_path))
    out = str(tmp_path / "out")

    summary = convert_dataset(parquet, tasks, out)

    assert summary.written == [FIRST, NESTED]
    assert summary.skipped == {
        NO_TESTS: "no tests/test.sh",
        NO_DIR: "task directory not in task-data",
    }
    first = os.path.join(out, FIRST)
    assert (
        open(os.path.join(first, "instruction.md")).read()
        == "Fix the parser in /app.\n"
    )
    assert (
        open(os.path.join(first, "tests", "test.sh")).read()
        == "echo 1 > /logs/verifier/reward.txt\n"
    )
    # setup.sh is baked into AI2's image, so it is not carried over; nothing is built.
    assert not os.path.exists(os.path.join(first, "setup.sh"))
    assert not os.path.exists(os.path.join(first, "environment"))
    assert not os.path.exists(os.path.join(out, NO_TESTS))


def test_converted_tasks_load_through_verifiers_harbor_taskset(tmp_path) -> None:
    # Verifiers' Harbor taskset imports the harbor package when it parses a task.
    pytest.importorskip("harbor")
    parquet, tasks = _make_dataset(str(tmp_path))
    out = str(tmp_path / "out")
    convert_dataset(parquet, tasks, out)
    config = HarborConfig(dataset="local/tmax")

    first = parse_task(Path(out) / FIRST, 0, config)
    nested = parse_task(Path(out) / NESTED, 1, config)

    # The image is AI2's and the prompt is AI2's text.
    assert first.image == "hamishi740/swerl-tmax-v3:0123456789ab"
    assert nested.image == "hamishi740/swerl-tmax-v3:ba9876543210"
    assert first.name == f"allenai-tmax/{FIRST}"
    # The sandbox's default workdir (/testbed) is absent from AI2's images.
    assert first.workdir == "/home/user"
    assert first.prompt == "Fix the parser in /app."
    # Nothing is staged into the sandbox before the agent: no environment directory.
    assert first.upload_environment is False
    assert nested.upload_environment is False


def test_limit_include_and_exclude_select_tasks(tmp_path) -> None:
    parquet, tasks = _make_dataset(str(tmp_path))

    limited = convert_dataset(parquet, tasks, str(tmp_path / "a"), limit=1)
    included = convert_dataset(parquet, tasks, str(tmp_path / "b"), include={NESTED})
    excluded = convert_dataset(parquet, tasks, str(tmp_path / "c"), exclude={FIRST})

    assert limited.written == [FIRST]
    assert included.written == [NESTED]
    assert excluded.written == [NESTED]


def test_rerun_does_not_overwrite_converted_tasks(tmp_path) -> None:
    parquet, tasks = _make_dataset(str(tmp_path))
    out = str(tmp_path / "out")
    convert_dataset(parquet, tasks, out)

    again = convert_dataset(parquet, tasks, out)

    assert again.written == []
    assert again.skipped[FIRST] == "already converted"
