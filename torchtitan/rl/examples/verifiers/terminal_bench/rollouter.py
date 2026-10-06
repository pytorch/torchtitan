# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run Terminus-2 and Harbor grading inside one sandbox per rollout."""

import math
from dataclasses import dataclass

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig
from verifiers.v1.harnesses.terminus_2 import Terminus2HarnessConfig
from verifiers.v1.tasksets.harbor import HarborEnvConfig

from torchtitan.rl.examples.verifiers import (
    GenerationServer,
    RewardFromVerifiers,
    VerifiersEnvServer,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from torchtitan.rl.examples.verifiers.terminal_bench.taskset import (
    TerminalTasksetConfig,
)
from torchtitan.rl.rubric import Rubric

_ENV_SERVER_WORKERS = 16


class TerminalBenchRollouter(VerifiersRollouter):
    """Use Verifiers for agent execution and TitanRL for training orchestration."""

    @dataclass(kw_only=True, slots=True)
    class Config(VerifiersRollouter.Config):
        connection_timeout_sec: float = 1800.0
        """How long to wait for the Verifiers env server to report healthy at startup."""


def terminal_bench_rollouter_config(
    train_dataset: str,
    validation_dataset: str,
    *,
    max_context_length: int,
    max_turns: int,
    max_concurrent_rollouts: int,
) -> TerminalBenchRollouter.Config:
    """Select Harbor datasets by id.

    ``max_context_length`` is the generator's sequence length; the generation
    server caps each rollout at it. ``max_turns`` is the agent turn limit, which
    Verifiers enforces. ``max_concurrent_rollouts`` sizes the env server; set it
    to the number of rollouts the controller keeps in flight, or the excess
    queues in the env server and the generators idle.
    """
    if train_dataset == validation_dataset:
        raise ValueError(
            "Training and Terminal-Bench evaluation must use different datasets"
        )

    taskset_id = register_local_taskset_alias(TerminalTasksetConfig.__module__)
    return TerminalBenchRollouter.Config(
        train_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=TerminalTasksetConfig(
                id=taskset_id, dataset=train_dataset
            ),
            seed=42,
            shuffle=True,
        ),
        validation_dataset=VerifiersTaskDataset.Config(
            verifiers_taskset=TerminalTasksetConfig(
                id=taskset_id, dataset=validation_dataset
            ),
            seed=99,
            shuffle=False,
        ),
        verifiers_env_server=VerifiersEnvServer.Config(
            environment=HarborEnvConfig(
                agent=vf.AgentConfig(
                    harness=Terminus2HarnessConfig(id="terminus_2", version="0.22.0"),
                    runtime=vf.DockerConfig(),
                    max_turns=max_turns,
                    timeout=AgentTimeoutConfig(
                        setup=600,
                        rollout=7200,
                        scoring=12000,
                    ),
                ),
            ),
            serve=vf.ServeConfig(
                pool=vf.StaticPoolConfig(num_workers=_ENV_SERVER_WORKERS),
                max_concurrent=math.ceil(max_concurrent_rollouts / _ENV_SERVER_WORKERS),
                address="tcp://127.0.0.1:0",
            ),
        ),
        rubric=Rubric.Config(
            reward_fns=[RewardFromVerifiers.Config(weight=1.0)],
            error_reward=0.0,
        ),
        generation_server=GenerationServer.Config(
            max_rollout_tokens=max_context_length
        ),
    )
