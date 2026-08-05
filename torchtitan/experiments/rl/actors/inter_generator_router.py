# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Actor wrapper for inter-generator routing."""

from collections.abc import Sequence
from typing import Any

from monarch.actor import Actor, endpoint

from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.experiments.rl.routing.types import RoutingContext
from torchtitan.observability import structured_logger as sl


class InterGeneratorRouterActor(Actor):
    """Give mutable inter-generator routing state a single process owner.

    ``InterGeneratorRouter`` tracks mutable generator load and lifecycle state
    and is not safe to share across threads or processes. Hosting it in a
    singleton Monarch actor confines that state to one actor process and makes
    other actors access it through endpoints instead of sharing the router.
    """

    def __init__(
        self,
        config: InterGeneratorRouter.Config,
        *,
        generators: Sequence[Any],
    ) -> None:
        self._router = InterGeneratorRouter(config, generators=generators)

    @endpoint
    async def generate(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        routing_session_id: str | None,
        sampling_config: Any | None,
        metrics_prefix: str,
    ) -> Any:
        """Route one generation call while holding global routing state."""
        # Dispatches to the chosen generator's rank-0 intake via call_one, so
        # it returns the Completion directly (no ValueMesh unwrap).
        return await self._router.route(
            "generate",
            prompt_token_ids,
            request_id=request_id,
            # VLLMGenerator.generate also requires this field for its
            # intra-mesh DP routing.
            routing_session_id=routing_session_id,
            sampling_config=sampling_config,
            metrics_prefix=metrics_prefix,
            # Load is measured as in-flight request count (one unit per call).
            routing_ctx=RoutingContext(
                estimated_cost=1,
                session_id=routing_session_id,
            ),
        )

    @endpoint
    async def start_engine_loop(self) -> None:
        await self._router.fanout("start_engine_loop")

    @endpoint
    async def sync_log_step(self, step: int) -> None:
        sl.set_step(step)
        await self._router.fanout("sync_log_step", step)

    @endpoint
    async def pull_model_state_dict(self, policy_version: int) -> None:
        await self._router.pull_model_state_dict(policy_version=policy_version)

    @endpoint
    async def close_generators(self) -> list[Any | BaseException]:
        return await self._router.fanout("close", return_exceptions=True)
