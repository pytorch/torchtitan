# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch actor adapter for the inter-generator router."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from monarch.actor import Actor, concurrent_endpoint, current_size

from torchtitan.rl.actors.system import ServiceHandle
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter


class _InterGeneratorRouterActorEndpoints:
    def __init__(
        self,
        config: InterGeneratorRouter.Config,
        *,
        generators: Sequence[ServiceHandle],
    ) -> None:
        num_actors = math.prod(current_size().values())
        assert (
            num_actors == 1
        ), f"InterGeneratorRouter must be a singleton, but its mesh holds {num_actors} actors"
        super().__init__(config, generators=generators)

    @concurrent_endpoint
    async def generate(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        group_id: int,
        routing_session_id: str | None,
        sampling_config: Any | None,
        metrics_prefix: str,
    ) -> Any:
        return await super().generate(
            prompt_token_ids,
            request_id=request_id,
            group_id=group_id,
            routing_session_id=routing_session_id,
            sampling_config=sampling_config,
            metrics_prefix=metrics_prefix,
        )

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
    async def close_generators(self) -> list[Any | BaseException]:
        return await super().close_generators()


class InterGeneratorRouterActor(
    Actor, _InterGeneratorRouterActorEndpoints, InterGeneratorRouter
):
    """Expose the singleton :class:`InterGeneratorRouter` through Monarch endpoints."""
