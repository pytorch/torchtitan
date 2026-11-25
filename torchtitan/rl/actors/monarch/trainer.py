# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the standalone RL trainer."""

from __future__ import annotations

from typing import Any

import torch
from monarch.actor import concurrent_endpoint

from torchtitan.rl.actors.monarch.actor import Actor
from torchtitan.rl.model_state_dict_transport import (
    ModelStateDictTransport,
    TensorShard,
    TrainerSource,
)
from torchtitan.rl.torch_transport_rdma import RDMAProbe
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
    async def check_rdma_probes(self, probes: list[RDMAProbe]) -> None:
        await super().check_rdma_probes(probes)

    @concurrent_endpoint
    async def receive_replica_column(self, column: ReplicaColumn) -> None:
        await super().receive_replica_column(column)

    @concurrent_endpoint
    async def start_reading(self, *, num_microbatch_groups: int) -> None:
        await super().start_reading(num_microbatch_groups=num_microbatch_groups)

    @concurrent_endpoint
    async def forward_backward(
        self,
        *,
        num_microbatch_groups: int,
        global_loss_token_counts: torch.Tensor,
        global_routing_token_counts: torch.Tensor,
    ) -> dict[str, float] | None:
        return await super().forward_backward(
            num_microbatch_groups=num_microbatch_groups,
            global_loss_token_counts=global_loss_token_counts,
            global_routing_token_counts=global_routing_token_counts,
        )

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
