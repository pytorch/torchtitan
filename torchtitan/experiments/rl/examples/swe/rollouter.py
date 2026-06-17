# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import dataclass, field

from renderers import Renderer

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.swe.data import R2EGymDataset
from torchtitan.experiments.rl.examples.swe.env import SweEnv
from torchtitan.experiments.rl.examples.swe.rubric import RewardR2EGym
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rubrics import Rubric
from torchtitan.experiments.rl.sandbox import SandboxFactory

# Bundled smoke set: the two orange3 R2E-Gym instances whose images are small and
# commonly cached. Point ``train_dataset.data_path`` at a larger jsonl for real runs.
_SMOKE_DATA = os.path.join(os.path.dirname(__file__), "data", "r2e_smoke.jsonl")


class SweRollouter(Rollouter):
    """Wires the R2E-Gym dataset, the SWE coding-agent env, and the R2E rubric.

    Overrides ``make_env_group`` to inject ONE shared ``SandboxFactory`` into every
    env, so the factory's provisioning semaphore caps concurrency across the whole
    run (a per-env factory would make the cap meaningless).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: R2EGymDataset.Config = field(
            default_factory=lambda: R2EGymDataset.Config(data_path=_SMOKE_DATA, seed=42)
        )
        validation_dataset: R2EGymDataset.Config = field(
            default_factory=lambda: R2EGymDataset.Config(
                data_path=_SMOKE_DATA, seed=99, shuffle=False
            )
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardR2EGym.Config(weight=1.0)],
                # A truncated rollout never reached grading -> no learning signal.
                truncation_reward=0.0,
                # An errored rollout (e.g. provision failure) -> no signal.
                error_reward=0.0,
            )
        )
        message_env: SweEnv.Config = field(default_factory=SweEnv.Config)
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(
                # Must stay below the trainer's seq_len (set per config). A rollout
                # whose prompt would exceed this truncates WITHOUT grading.
                max_rollout_tokens=14000,
                # The env self-caps turns (so it grades on the last turn); leave
                # the TokenEnv turn cap off and let SweEnv.max_turns govern.
                max_num_turns=None,
                # Wraps the whole MessageEnv.step, including the terminal grading
                # run; must exceed SweEnv.eval_timeout_s with headroom.
                step_timeout_s=900.0,
            )
        )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # One factory (one semaphore) shared by every env this run builds.
        self._sandbox_factory: SandboxFactory = config.message_env.sandbox.build()

    def make_env_group(
        self, *, sample: object, group_size: int, renderer: Renderer
    ) -> list[TokenEnv]:
        return [
            self._token_env_config.build(
                message_env=self._message_env_config.build(
                    env_input=sample, sandbox_factory=self._sandbox_factory
                ),
                renderer=renderer,
            )
            for _ in range(group_size)
        ]
