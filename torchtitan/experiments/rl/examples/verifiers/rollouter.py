# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from torchtitan.experiments.rl.rollout.verifiers import (
    VerifiersEnvServer,
    VerifiersRewardFn,
    VerifiersRollouter,
    VerifiersTaskDataset,
)
from torchtitan.experiments.rl.rubrics import Rubric


_TASKSET_ID = "torchtitan.experiments.rl.examples.verifiers.taskset"


class VerifiersMathRollouter(VerifiersRollouter):
    """Run DAPO-Math and AIME rollouts through Verifiers."""

    @dataclass(kw_only=True, slots=True)
    class Config(VerifiersRollouter.Config):
        train_dataset: VerifiersTaskDataset.Config = field(
            default_factory=lambda: VerifiersTaskDataset.Config(
                taskset_id=_TASKSET_ID,
                taskset_args={"dataset": "dapo_math"},
                seed=42,
            )
        )
        validation_dataset: VerifiersTaskDataset.Config = field(
            default_factory=lambda: VerifiersTaskDataset.Config(
                taskset_id=_TASKSET_ID,
                taskset_args={"dataset": "aime2025"},
                seed=99,
                shuffle=False,
            )
        )
        env_server: VerifiersEnvServer.Config = field(
            default_factory=lambda: VerifiersEnvServer.Config(
                config_path=str(Path(__file__).with_name("verifiers_env.toml")),
            )
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[VerifiersRewardFn.Config(weight=1.0)],
                error_reward=0.0,
            )
        )
        max_model_len: int = 10240
