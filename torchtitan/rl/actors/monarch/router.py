# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the inter-generator router."""

from __future__ import annotations

import math
from typing import Any

from monarch.actor import concurrent_endpoint, current_size

from torchtitan.rl.actors.monarch.actor import TitanActor
from torchtitan.rl.actors.monarch.generator import GeneratorActor
from torchtitan.rl.actors.monarch.provisioning import (
    generator_role_names,
    RolePlacement,
)
from torchtitan.rl.actors.monarch.torch_elastic import setup_torch_elastic_env
from torchtitan.rl.actors.system import ServiceHandle
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.model_state_dict_transport import (
    ModelStateDictTransport,
    TensorShard,
    TrainerSource,
)
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter


class _InterGeneratorRouterActorEndpoints:
    def __init__(
        self,
        config: InterGeneratorRouter.Config,
        *,
        placements: list[RolePlacement],
        generator_config: Generator.Config,
        generator_kwargs: dict[str, Any],
    ) -> None:
        num_actors = math.prod(current_size().values())
        assert (
            num_actors == 1
        ), f"InterGeneratorRouter must be a singleton, but its mesh holds {num_actors} actors"
        super().__init__(config, num_generators=len(placements))
        self._placements = placements
        self._generator_config = generator_config
        self._generator_kwargs = generator_kwargs

    @concurrent_endpoint
    async def spawn_generators(self) -> None:
        await super().spawn_generators()

    @concurrent_endpoint
    async def reserve(self, request_id: str, *, routing_session_id: str | None) -> Any:
        return super().reserve(request_id, routing_session_id=routing_session_id)

    @concurrent_endpoint
    async def release_reservation(self, request_id: str) -> None:
        super().release_reservation(request_id)

    @concurrent_endpoint
    async def start_engine_loop(self) -> None:
        await super().start_engine_loop()

    @concurrent_endpoint
    async def wait_engine_loop(self) -> None:
        await super().wait_engine_loop()

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        await super().sync_log_step(step)

    @concurrent_endpoint
    async def release_session(self, routing_session_id: str) -> None:
        await super().release_session(routing_session_id)

    @concurrent_endpoint
    async def release_groups(self, group_ids: list[int]) -> None:
        await super().release_groups(group_ids)

    @concurrent_endpoint
    async def pull_model_state_dict(self, policy_version: int) -> None:
        await super().pull_model_state_dict(policy_version)

    @concurrent_endpoint
    async def model_state_dict_layouts(
        self,
    ) -> list[list[tuple[int, dict[str, TensorShard]]]]:
        return await super().model_state_dict_layouts()

    @concurrent_endpoint
    async def connect_model_state_dict_transport(
        self,
        config: ModelStateDictTransport.Config,
        *,
        trainer_sources: list[TrainerSource],
    ) -> None:
        await super().connect_model_state_dict_transport(
            config, trainer_sources=trainer_sources
        )

    @concurrent_endpoint
    async def initialize_torchstore_clients(self) -> None:
        await super().initialize_torchstore_clients()

    @concurrent_endpoint
    async def close_generators(self) -> list[Any | BaseException]:
        return await super().close_generators()


class InterGeneratorRouterActor(
    TitanActor, _InterGeneratorRouterActorEndpoints, InterGeneratorRouter
):
    """Expose the singleton :class:`InterGeneratorRouter` through Monarch endpoints;
    spawns each generator's proc mesh on its ``RolePlacement``."""

    async def _spawn_generator(self, index: int) -> ServiceHandle:
        procs = self._placements[index].spawn_procs()
        await setup_torch_elastic_env(procs)
        generator = procs.spawn(
            generator_role_names(len(self._placements))[index],
            GeneratorActor,
            self._generator_config,
            **self._generator_kwargs,
        )
        return ServiceHandle(
            all_ranks=generator, rank0=generator.flatten("rank").slice(rank=0)
        )
