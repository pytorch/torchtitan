# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the standalone RL trainer."""

from __future__ import annotations

import time
from typing import Any

import torch
from monarch.actor import Actor, concurrent_endpoint

from torchtitan.rl.model_state_dict_transport import (
    ModelStateDictTransport,
    TensorShard,
    TrainerSource,
)
from torchtitan.rl.trainer.microbatch_group_reader import ReplicaColumn
from torchtitan.rl.trainer.trainer import OptimizerStepOutput, Trainer


class _TrainerActorEndpoints:
    @concurrent_endpoint
    async def initialize_torchstore_client(self) -> None:
        await super().initialize_torchstore_client()

    @concurrent_endpoint
    async def register_model_state_dict_source(
        self,
        config: ModelStateDictTransport.Config,
        *,
        generator_rank_layouts: list[dict[str, TensorShard]],
    ) -> TrainerSource:
        return super().register_model_state_dict_source(
            config, generator_rank_layouts=generator_rank_layouts
        )

    @concurrent_endpoint
    async def get_policy_version(self) -> int:
        return super().policy_version

    @concurrent_endpoint
    async def get_dp_rank(self) -> int:
        return self.dp_rank

    @concurrent_endpoint
    async def get_resume_state(self) -> dict[str, Any] | None:
        return await super().get_resume_state()

    @concurrent_endpoint
    async def close(self) -> None:
        await super().close()

    @concurrent_endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        await super().sync_log_step(step, relative_step)

    @concurrent_endpoint
    async def receive_replica_column(self, column: ReplicaColumn) -> float:
        """`Trainer.receive_replica_column`, plus the seconds this handler took (see
        `forward_backward`)."""
        start = time.perf_counter()
        await super().receive_replica_column(column)
        return time.perf_counter() - start

    @concurrent_endpoint
    async def forward_backward(
        self,
        *,
        num_microbatch_groups: int,
        global_loss_token_counts: torch.Tensor,
        global_routing_token_counts: torch.Tensor,
    ) -> tuple[dict[str, float] | None, float]:
        """`Trainer.forward_backward`'s metrics plus the seconds this handler took; the controller
        subtracts them from its wall time to get the hop's transfer time (see `compute_hop_metrics`)."""
        start = time.perf_counter()
        metrics = await super().forward_backward(
            num_microbatch_groups=num_microbatch_groups,
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_routing_token_counts,
        )
        return metrics, time.perf_counter() - start

    @concurrent_endpoint
    async def optim_step(
        self, *, resume_state: dict[str, Any], last_step: bool = False
    ) -> OptimizerStepOutput:
        return await super().optim_step(resume_state=resume_state, last_step=last_step)

    @concurrent_endpoint
    async def push_model_state_dict(self) -> None:
        await super().push_model_state_dict()


class TrainerActor(Actor, _TrainerActorEndpoints, Trainer):
    """Expose a standalone :class:`Trainer` through Monarch endpoints."""
