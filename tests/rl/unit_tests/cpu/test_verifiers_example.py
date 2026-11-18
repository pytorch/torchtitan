# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the optional Verifiers math example."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

pytest.importorskip("verifiers")

import verifiers.v1 as vf

from torchtitan.components.renderer import RenderersConfigAdapter

from torchtitan.config import ConfigLoader
from torchtitan.rl.examples.dapo_math import DapoMathSample
from torchtitan.rl.examples.verifiers.dapo_math import data
from torchtitan.rl.examples.verifiers.data import VerifiersTaskDataset
from torchtitan.rl.examples.verifiers.rollouter import VerifiersRollouter
from torchtitan_recipes.rl.verifiers_dapo_math import _verifiers_math_rollouter_config
from verifiers.v1.harnesses.null import NullHarnessConfig as VerifiersNullHarnessConfig


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


def test_verifiers_environment_uses_no_sandbox() -> None:
    rollouter_config = _verifiers_math_rollouter_config(max_rollout_tokens=10240)
    config = rollouter_config.verifiers_env_server

    assert isinstance(config.environment, vf.SingleAgentEnvConfig)
    assert isinstance(config.environment.agent.runtime, vf.SubprocessConfig)
    assert isinstance(config.environment.agent.harness, VerifiersNullHarnessConfig)
    assert isinstance(config.serve.pool, vf.StaticPoolConfig)
    assert config.serve.pool.num_workers == 1
    assert (
        config.environment.taskset
        == rollouter_config.training_dataloader.dataset.verifiers_taskset
    )
    assert config.local_taskset_module == data.__name__


@pytest.mark.parametrize(
    ("agent_update", "expectation"),
    [
        (
            {"sampling": vf.SamplingConfig(temperature=0.7)},
            pytest.raises(ValueError, match=r"sets sampling \['temperature'\]"),
        ),
        (
            {"sampling": vf.SamplingConfig(seed=1)},
            pytest.raises(ValueError, match=r"sets sampling \['seed'\]"),
        ),
        ({"sampling": vf.SamplingConfig(max_tokens=256)}, contextlib.nullcontext()),
        (
            {
                "sampling": vf.SamplingConfig(temperature=0.7),
                "client": vf.EvalClientConfig(),
            },
            contextlib.nullcontext(),
        ),
    ],
    ids=["temperature", "seed", "max_tokens", "own_client"],
)
def test_agent_sampling_allows_only_max_tokens_and_chat_template_kwargs(
    agent_update, expectation
) -> None:
    rollouter_config = _verifiers_math_rollouter_config(max_rollout_tokens=10240)
    env_server_config = rollouter_config.verifiers_env_server
    agent = env_server_config.environment.agent.model_copy(update=agent_update)
    environment = env_server_config.environment.model_copy(update={"agent": agent})

    with expectation:
        replace(
            rollouter_config,
            verifiers_env_server=replace(env_server_config, environment=environment),
        )


@pytest.mark.parametrize(
    "recipe_env_class_is_the_tasksets",
    [False, True],
    ids=["single_agent_env", "taskset_env"],
)
def test_agent_sampling_check_sees_the_taskset_env_class_defaults(
    monkeypatch, recipe_env_class_is_the_tasksets
) -> None:
    """The env server resolves the dumped env config into the taskset's own env class. That
    class's default temperature then merges into the agent's sampling, which set only max_tokens.
    The merge also happens when the recipe already passes that class."""

    class DefaultTemperatureEnvConfig(vf.EnvConfig):
        agent: vf.AgentConfig = vf.AgentConfig(
            sampling=vf.SamplingConfig(temperature=0.6)
        )

    class DefaultTemperatureEnv(vf.Env[DefaultTemperatureEnvConfig]):
        async def run(self, task, agents) -> None:
            await agents.agent.run(task)

    taskset_module = ModuleType("taskset_with_its_own_env")
    taskset_module.VerifiersMathTaskset = data.VerifiersMathTaskset
    taskset_module.DefaultTemperatureEnv = DefaultTemperatureEnv
    taskset_module.__all__ = ["VerifiersMathTaskset", "DefaultTemperatureEnv"]
    monkeypatch.setitem(sys.modules, taskset_module.__name__, taskset_module)
    taskset = data.VerifiersMathTasksetConfig(
        id=taskset_module.__name__, dataset="dapo_math"
    )

    rollouter_config = _verifiers_math_rollouter_config(max_rollout_tokens=10240)
    dataloader_config = rollouter_config.training_dataloader
    env_server_config = rollouter_config.verifiers_env_server
    agent = env_server_config.environment.agent.model_copy(
        update={"sampling": vf.SamplingConfig(max_tokens=256)}
    )
    environment_class = (
        DefaultTemperatureEnvConfig
        if recipe_env_class_is_the_tasksets
        else vf.SingleAgentEnvConfig
    )
    environment = environment_class(agent=agent, taskset=taskset)

    with pytest.raises(ValueError, match=r"sets sampling \['temperature'\]"):
        replace(
            rollouter_config,
            training_dataloader=replace(
                dataloader_config,
                dataset=replace(dataloader_config.dataset, verifiers_taskset=taskset),
            ),
            verifiers_env_server=replace(env_server_config, environment=environment),
        )


def test_verifiers_config_keeps_dapo_training_recipe() -> None:
    config = ConfigLoader().load(
        [
            "--module",
            "torchtitan_recipes.rl.verifiers_dapo_math",
            "--config",
            "rl_dapo_qwen3_4b_verifiers_8k",
        ]
    )
    assert isinstance(config.rollouter, VerifiersRollouter.Config)
    assert config.generator.sampling.max_tokens == 8192
    assert config.rollouter.worker.generation_server.max_rollout_tokens == 10240
    assert config.dump_folder == "outputs/rl/qwen3_4b_verifiers_8k"
    assert isinstance(config.renderer, RenderersConfigAdapter)
    assert config.renderer.renderers_config.name == "qwen3"
    assert config.renderer.renderers_config.enable_thinking
