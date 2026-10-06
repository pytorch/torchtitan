# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run Terminus-2 and Harbor grading inside one sandbox per rollout."""

from dataclasses import dataclass

import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig as AgentTimeoutConfig
from verifiers.v1.tasksets.harbor import HarborEnvConfig

from torchtitan.rl.examples.verifiers import (
    GenerationServer,
    RewardFromVerifiers,
    VerifiersEnvServer,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.rl.examples.verifiers.data import register_local_taskset_alias
from torchtitan.rl.examples.verifiers.terminal_bench.harness import (
    NUM_AGENT_TURNS,
    register_harness_alias,
    TerminalBenchTerminusHarnessConfig,
)
from torchtitan.rl.examples.verifiers.terminal_bench.taskset import (
    TerminalTasksetConfig,
)
from torchtitan.rl.rubric import Rubric


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
    max_tokens: int,
) -> TerminalBenchRollouter.Config:
    """Select Harbor datasets by id.

    ``max_context_length`` and ``max_tokens`` are the generator's sequence length
    and per-turn sampling cap; the generation server and the agent's context
    budget both follow them.
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
                    harness=TerminalBenchTerminusHarnessConfig(
                        id=register_harness_alias(),
                        version="0.22.0",
                        max_input_tokens=max_context_length,
                        max_output_tokens=max_tokens,
                    ),
                    runtime=vf.DockerConfig(),
                    max_turns=NUM_AGENT_TURNS,
                    timeout=AgentTimeoutConfig(
                        setup=600,
                        rollout=7200,
                        scoring=12000,
                    ),
                ),
            ),
            serve=vf.ServeConfig(
                pool=vf.StaticPoolConfig(num_workers=4),
                max_concurrent=4,
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
