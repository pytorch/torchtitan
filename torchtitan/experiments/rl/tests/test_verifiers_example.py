# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers math example."""

from __future__ import annotations

import asyncio
import tomllib
from types import SimpleNamespace

import pytest

pytest.importorskip("verifiers")

from torchtitan.experiments.rl.examples.dapo_math import DapoMathSample
from torchtitan.experiments.rl.examples.verifiers import taskset
from torchtitan.experiments.rl.examples.verifiers.config_registry import (
    rl_dapo_qwen3_4b_verifiers_8k,
)
from torchtitan.experiments.rl.examples.verifiers.rollouter import (
    VerifiersMathRollouter,
)
from torchtitan.experiments.rl.rollout.verifiers import VerifiersTaskDataset


def test_verifiers_task_scores_math_response() -> None:
    math_task = taskset.VerifiersMathTask(
        taskset.VerifiersMathData(
            idx=0,
            prompt="problem",
            ground_truth=r"336^\circ",
        )
    )
    assert (
        asyncio.run(math_task.math_verify(SimpleNamespace(last_reply="Answer: $336$")))
        == 1.0
    )
    assert (
        asyncio.run(math_task.math_verify(SimpleNamespace(last_reply="Answer: $335$")))
        == 0.0
    )


def test_verifiers_task_dataset_is_resumable(monkeypatch) -> None:
    samples = [
        DapoMathSample(prompt="problem 1", ground_truth="34"),
        DapoMathSample(prompt="problem 2", ground_truth="113"),
        DapoMathSample(prompt="problem 3", ground_truth="7"),
    ]
    monkeypatch.setattr(taskset, "_dataset", lambda name: (iter(samples), 3))
    config = VerifiersTaskDataset.Config(
        taskset_id="torchtitan.experiments.rl.examples.verifiers.taskset",
        taskset_args={"dataset": "dapo_math"},
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


def test_verifiers_environment_uses_no_sandbox() -> None:
    config_path = VerifiersMathRollouter.Config().env_server.config_path
    with open(config_path, "rb") as file:
        config = tomllib.load(file)

    assert config["env"]["agent"]["runtime"]["type"] == "subprocess"
    assert config["env"]["agent"]["harness"]["id"] == "null"


def test_verifiers_config_keeps_dapo_training_recipe() -> None:
    config = rl_dapo_qwen3_4b_verifiers_8k()

    assert isinstance(config.rollouter, VerifiersMathRollouter.Config)
    assert config.generator.sampling.max_tokens == 8192
    assert config.dump_folder == "outputs/rl/qwen3_4b_verifiers_8k"
