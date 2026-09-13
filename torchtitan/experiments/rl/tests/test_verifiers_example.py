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

import verifiers.v1 as vf
from verifiers.v1.harnesses.null import NullHarnessConfig as VerifiersNullHarnessConfig

from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.examples.dapo_math import DapoMathSample
from torchtitan.experiments.rl.examples.verifiers.components import VerifiersTaskDataset
from torchtitan.experiments.rl.examples.verifiers.dapo_math import data
from torchtitan.experiments.rl.examples.verifiers.dapo_math.rollouter import (
    VerifiersMathRollouter,
)
from torchtitan.experiments.rl.renderer import RenderersLibraryConfig


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
            id="torchtitan.experiments.rl.examples.verifiers.dapo_math.data",
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


def test_verifiers_environment_uses_no_sandbox() -> None:
    rollouter_config = VerifiersMathRollouter.Config()
    config = rollouter_config.verifiers_env_server

    assert isinstance(config.environment, vf.SingleAgentEnvConfig)
    assert isinstance(config.environment.agent.runtime, vf.SubprocessConfig)
    assert isinstance(config.environment.agent.harness, VerifiersNullHarnessConfig)
    assert isinstance(config.serve.pool, vf.StaticPoolConfig)
    assert config.serve.pool.num_workers == 1
    assert (
        config.environment.taskset == rollouter_config.train_dataset.verifiers_taskset
    )
    assert config.local_taskset_module == data.__name__


def test_verifiers_config_keeps_dapo_training_recipe() -> None:
    config = ConfigManager().parse_args(
        [
            "--module",
            "verifiers.dapo_math",
            "--config",
            "rl_dapo_qwen3_4b_verifiers_8k",
        ]
    )
    assert isinstance(config.rollouter, VerifiersMathRollouter.Config)
    assert config.generator.sampling.max_tokens == 8192
    assert config.dump_folder == "outputs/rl/qwen3_4b_verifiers_8k"
    assert isinstance(config.renderer, RenderersLibraryConfig)
    assert config.renderer.renderers_config.name == "qwen3"
    assert config.renderer.renderers_config.enable_thinking
