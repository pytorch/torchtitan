# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the standalone RL trainer."""

from __future__ import annotations

from monarch.actor import Actor, concurrent_endpoint

from torchtitan.rl.trainer import Trainer
from torchtitan.rl.types import OptimizerStepOutput, TrainingMicrobatch


class _TrainerActorEndpoints:
    @concurrent_endpoint
    async def get_policy_version(self) -> int:
        return await super().get_policy_version()

    @concurrent_endpoint
    async def close(self) -> None:
        await super().close()

    @concurrent_endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        await super().sync_log_step(step, relative_step)

    @concurrent_endpoint
    async def forward_backward_steps(
        self,
        training_data: list[list[TrainingMicrobatch]],
        num_global_valid_tokens: int,
    ) -> dict[str, float]:
        return await super().forward_backward_steps(
            training_data,
            num_global_valid_tokens,
        )

    @concurrent_endpoint
    async def optimizer_step(self, *, last_step: bool = False) -> OptimizerStepOutput:
        return await super().optimizer_step(last_step=last_step)

    @concurrent_endpoint
    async def push_model_state_dict(self) -> None:
        await super().push_model_state_dict()


class TrainerActor(Actor, _TrainerActorEndpoints, Trainer):
    """Expose a standalone :class:`Trainer` through Monarch endpoints."""
