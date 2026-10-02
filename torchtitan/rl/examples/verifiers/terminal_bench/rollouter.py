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
        # Per-request timeout on the model call. The base default is 120 s,
        # which cannot be met here: a turn is allowed max_tokens=16384, and
        # finishing that inside 120 s needs a sustained 137 tok/s for one
        # sequence while its 31 group siblings share the same engine. A turn
        # that crosses the deadline raises APITimeoutError, which the harness
        # surfaces as a rollout with no turns and reward 0.0 -- identical to a
        # task the agent genuinely failed. Kept below the 7200 s rollout
        # timeout so a stuck request still loses to the rollout deadline.
        connection_timeout_sec: float = 1800.0


def terminal_bench_rollouter_config(
    train_dataset: str, validation_dataset: str
) -> TerminalBenchRollouter.Config:
    """Select Harbor datasets by id; never mix benchmark tasks into training."""
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
                        id=register_harness_alias(), version="0.22.0"
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
        generation_server=GenerationServer.Config(max_rollout_tokens=65536),
    )
