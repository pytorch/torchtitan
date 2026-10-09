# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the standalone RL trainer."""

from __future__ import annotations

from typing import Any

import torch
from monarch.actor import Actor, concurrent_endpoint

from torchtitan.rl.trainer import Trainer
from torchtitan.rl.types import OptimizerStepOutput, TrainingMicrobatch


class _TrainerActorEndpoints:
    @concurrent_endpoint
    async def initialize_torchstore_client(self) -> None:
        await super().initialize_torchstore_client()

    @concurrent_endpoint
    async def get_policy_version(self) -> int:
        return super().policy_version

    @concurrent_endpoint
    async def get_controller_state(self) -> dict[str, Any] | None:
        return await super().get_controller_state()

    @concurrent_endpoint
    async def close(self) -> None:
        await super().close()

    @concurrent_endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        await super().sync_log_step(step, relative_step)

    @concurrent_endpoint
    async def forward_backward(
        self,
        training_data: list[list[TrainingMicrobatch]],
        global_loss_token_counts: torch.Tensor,
        global_routing_token_counts: torch.Tensor,
    ) -> dict[str, float]:
        return await super().forward_backward(
            training_data,
            global_loss_token_counts,
            global_routing_token_counts,
        )

    @concurrent_endpoint
    async def optim_step(
        self, *, controller_state: dict[str, Any], last_step: bool = False
    ) -> OptimizerStepOutput:
        return await super().optim_step(
            controller_state=controller_state, last_step=last_step
        )

    @concurrent_endpoint
    async def push_model_state_dict(self) -> None:
        await super().push_model_state_dict()


class TrainerActor(Actor, _TrainerActorEndpoints, Trainer):
    """Expose a standalone :class:`Trainer` through Monarch endpoints."""
