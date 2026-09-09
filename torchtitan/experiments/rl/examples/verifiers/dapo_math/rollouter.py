# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import verifiers.v1 as vf
from verifiers.v1.harnesses.null import NullHarnessConfig as VerifiersNullHarnessConfig

from torchtitan.experiments.rl.examples.verifiers.components import (
    RewardFromVerifiers,
    VerifiersEnvServer,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.experiments.rl.examples.verifiers.components.data import (
    register_local_taskset_alias,
)
from torchtitan.experiments.rl.examples.verifiers.dapo_math.data import (
    VerifiersMathTasksetConfig,
)
from torchtitan.experiments.rl.rubrics import Rubric


def _math_taskset_config(
    dataset: Literal["dapo_math", "aime2025"],
) -> VerifiersMathTasksetConfig:
    taskset_id = register_local_taskset_alias(VerifiersMathTasksetConfig.__module__)
    return VerifiersMathTasksetConfig(id=taskset_id, dataset=dataset)


class VerifiersMathRollouter(VerifiersRollouter):
    """Run DAPO-Math and AIME through a local Verifiers environment server.

    The default uses one static environment worker, the null harness, and a
    subprocess runtime. It performs single-turn generation without tools or a
    sandbox.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(VerifiersRollouter.Config):
        train_dataset: VerifiersTaskDataset.Config = field(
            default_factory=lambda: VerifiersTaskDataset.Config(
                verifiers_taskset=_math_taskset_config("dapo_math"),
                seed=42,
            )
        )
        validation_dataset: VerifiersTaskDataset.Config = field(
            default_factory=lambda: VerifiersTaskDataset.Config(
                verifiers_taskset=_math_taskset_config("aime2025"),
                seed=99,
                shuffle=False,
            )
        )
        verifiers_env_server: VerifiersEnvServer.Config = field(
            default_factory=lambda: VerifiersEnvServer.Config(
                environment=vf.SingleAgentEnvConfig(
                    agent=vf.AgentConfig(
                        runtime=vf.SubprocessConfig(),
                        max_turns=1,
                        harness=VerifiersNullHarnessConfig(id="null"),
                    ),
                ),
                serve=vf.ServeConfig(
                    # This lightweight single-turn recipe starts with one worker.
                    # Increase num_workers for CPU-heavy or multi-turn environments.
                    pool=vf.StaticPoolConfig(num_workers=1),
                    address="tcp://127.0.0.1:0",
                ),
            )
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardFromVerifiers.Config(weight=1.0)],
                error_reward=0.0,
            )
        )
