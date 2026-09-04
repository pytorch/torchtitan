# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

import verifiers.v1 as vf
from verifiers.v1.harnesses.null import NullHarnessConfig

from torchtitan.experiments.rl.examples.verifiers.components import (
    VerifiersEnvServer,
    VerifiersRewardFn,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.experiments.rl.examples.verifiers.components.dataset import (
    register_local_taskset_alias,
)
from torchtitan.experiments.rl.examples.verifiers.dapo_math.taskset import (
    VerifiersMathTasksetConfig,
)
from torchtitan.experiments.rl.rubrics import Rubric


# Register the local module under the single-segment plugin ID expected by
# Verifiers' taskset loader. The spawned server repeats this registration.
_TASKSET_MODULE = "torchtitan.experiments.rl.examples.verifiers.dapo_math.taskset"
_TASKSET_ID = register_local_taskset_alias(_TASKSET_MODULE)


class VerifiersMathRollouter(VerifiersRollouter):
    """Run DAPO-Math and AIME rollouts through Verifiers."""

    @dataclass(kw_only=True, slots=True)
    class Config(VerifiersRollouter.Config):
        train_dataset: VerifiersTaskDataset.Config = field(
            default_factory=lambda: VerifiersTaskDataset.Config(
                taskset=VerifiersMathTasksetConfig(
                    id=_TASKSET_ID,
                    dataset="dapo_math",
                ),
                seed=42,
            )
        )
        validation_dataset: VerifiersTaskDataset.Config = field(
            default_factory=lambda: VerifiersTaskDataset.Config(
                taskset=VerifiersMathTasksetConfig(
                    id=_TASKSET_ID,
                    dataset="aime2025",
                ),
                seed=99,
                shuffle=False,
            )
        )
        env_server: VerifiersEnvServer.Config = field(
            default_factory=lambda: VerifiersEnvServer.Config(
                environment=vf.SingleAgentEnvConfig(
                    taskset=VerifiersMathTasksetConfig(
                        id=_TASKSET_ID,
                        dataset="dapo_math",
                    ),
                    agent=vf.AgentConfig(
                        runtime=vf.SubprocessConfig(),
                        max_turns=1,
                        harness=NullHarnessConfig(id="null"),
                    ),
                ),
                serve=vf.ServeConfig(
                    # One worker is sufficient for this lightweight single-turn
                    # task. CPU-heavy or multi-turn environments should use
                    # multiple workers, or an elastic pool, for process-level
                    # parallelism instead of sharing one EnvServer GIL.
                    pool=vf.StaticPoolConfig(num_workers=1),
                    address="tcp://127.0.0.1:0",
                ),
                local_taskset_module=_TASKSET_MODULE,
            )
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[VerifiersRewardFn.Config(weight=1.0)],
                error_reward=0.0,
            )
        )
        max_model_len: int = 10240
