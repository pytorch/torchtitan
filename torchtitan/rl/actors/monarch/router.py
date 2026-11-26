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

from monarch.actor import concurrent_endpoint, current_size

from torchtitan.rl.actors.monarch.actor import Actor
from torchtitan.rl.actors.system import ServiceHandle
from torchtitan.rl.observability import metrics as m
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
    async def reserve(
        self,
        request_id: str,
        *,
        routing_session_id: str | None,
        group_id: int,
        num_prompt_tokens: int,
    ) -> Any:
        return super().reserve(
            request_id,
            routing_session_id=routing_session_id,
            group_id=group_id,
            num_prompt_tokens=num_prompt_tokens,
        )

    @concurrent_endpoint
    async def release_reservation(
        self, request_id: str, *, num_reply_tokens: int
    ) -> None:
        super().release_reservation(request_id, num_reply_tokens=num_reply_tokens)

    # Every endpoint stays a `@concurrent_endpoint`: Monarch starts those in send order, so a
    # release can't overtake the `acquire_group` it follows.
    @concurrent_endpoint
    async def acquire_group(
        self, group_id: int, *, num_sessions: int, validation: bool
    ) -> None:
        await super().acquire_group(
            group_id, num_sessions=num_sessions, validation=validation
        )

    @concurrent_endpoint
    async def admission_metrics(self) -> list[m.Metric]:
        return super().admission_metrics()

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
