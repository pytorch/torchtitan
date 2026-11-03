# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers math example."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers")


from torchtitan.rl.examples.dapo_math import DapoMathSample
from torchtitan.rl.examples.verifiers.dapo_math import data
from torchtitan.rl.examples.verifiers.data import VerifiersTaskDataset


def test_verifiers_task_scores_math_response() -> None:
    math_task = data.VerifiersMathTask(
        data.VerifiersMathData(
            idx=0,
            prompt="problem",
            ground_truth=r"336^\circ",
        )
    )
    assert (
        asyncio.run(
            math_task.math_verify(SimpleNamespace(last_reply=r"Answer: \boxed{336}"))
        )
        == 1.0
    )
    assert (
        asyncio.run(
            math_task.math_verify(SimpleNamespace(last_reply=r"Answer: \boxed{335}"))
        )
        == 0.0
    )


def test_verifiers_task_dataset_is_resumable(monkeypatch) -> None:
    samples = [
        DapoMathSample(prompt="problem 1", ground_truth="34"),
        DapoMathSample(prompt="problem 2", ground_truth="113"),
        DapoMathSample(prompt="problem 3", ground_truth="7"),
    ]
    monkeypatch.setattr(data, "_load_math_dataset", lambda name: (iter(samples), 3))
    config = VerifiersTaskDataset.Config(
        verifiers_taskset=data.VerifiersMathTasksetConfig(
            id="torchtitan.rl.examples.verifiers.dapo_math.data",
            dataset="dapo_math",
        ),
        seed=7,
    )
    first = config.build()

    second = config.build()
    assert [next(first) for _ in range(3)] == [next(second) for _ in range(3)]

    checkpoint = first.state_dict()
    expected = [next(first) for _ in range(3)]
    resumed = config.build()
    resumed.load_state_dict(checkpoint)
    assert [next(resumed) for _ in range(3)] == expected


def test_verifiers_task_dataset_keeps_the_first_num_tasks(monkeypatch) -> None:
    samples = [
        DapoMathSample(prompt=f"problem {i}", ground_truth="1") for i in range(3)
    ]
    monkeypatch.setattr(data, "_load_math_dataset", lambda name: (iter(samples), 3))
    dataset = VerifiersTaskDataset.Config(
        verifiers_taskset=data.VerifiersMathTasksetConfig(
            id="torchtitan.rl.examples.verifiers.dapo_math.data",
            dataset="dapo_math",
        ),
        num_tasks=2,
        shuffle=False,
    ).build()

    task_indices = [next(dataset).verifiers_task_data["idx"] for _ in range(4)]
    assert task_indices == [0, 1, 0, 1]
