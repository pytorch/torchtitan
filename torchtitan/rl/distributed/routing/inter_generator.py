# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator routing."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any

from monarch.actor import Actor, concurrent_endpoint, current_size

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoutingStrategy,
)
from torchtitan.rl.distributed.routing.types import RoutingCandidate, RoutingContext


class _GeneratorState(Enum):
    """Lifecycle state controlling routability; ``SYNCING`` is only entered when draining (i.e. hot-swap is off)."""

    SERVING = auto()
    SYNCING = auto()


@dataclass(kw_only=True, slots=True)
class _GeneratorHandle(RoutingCandidate):
    """Router-side metadata for one generator mesh."""

    actor: Any
    """Monarch actor handle for the full generator mesh. Used for fan-out calls
    that every rank must run."""

    rank0_actor: Any
    """Cached rank-0 slice of ``actor``. Used for calls that only rank 0 needs
    to run."""

    reserved_load: int = 0
    """Router-side estimate of in-flight routed generation work."""

    state: _GeneratorState = _GeneratorState.SERVING
    """Current routing lifecycle state for this generator."""

    policy_version: int | None = None
    """Version installed by the last completed pull on this generator."""

    idle: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when this generator has no reserved routed calls."""


@dataclass(kw_only=True, slots=True)
class _RoutingSession:
    generator: _GeneratorHandle
    cache_policy_version: int
    max_policy_version: int


@dataclass(kw_only=True, slots=True)
class _GroupRoute:
    generator: _GeneratorHandle
    cache_policy_version: int


class InterGeneratorRouter(Actor, Configurable):
    """Routes generation calls across generator meshes and pulls model's state dict.

    This is layer 1 of the two-layer routing design: it routes each call across
    generator *meshes* (replicas). Within the chosen mesh, ``IntraGeneratorRouter``
    then routes the request across that mesh's data-parallel ranks.

    Singleton:
        Routing decisions read and write mutable states such as ``_serving``,
        ``_GeneratorHandle.state``, and so on. These states are not backed by
        shared storage, so if there are multiple router instances, they cannot
        know each others' routing decisions. Instead of using shared storage,
        we solve the problem by enforcing the singleton pattern:
          * there should be only 1 router mesh in a training job;
          * this mesh should consists of only 1 actor.

        This pattern is simpler to implement, and should be good enough to handle
        the RL job's scale because the router is just a proxy, and the number of
        concurrent requests should be reasonable for a singleton to handle.

    Monarch Actor:
       The router singleton needs to be access from different processes or even
       different hosts. If we instantiate the router as an instance of a normal
       Python class, that instance's cannot be accessed from other processes or
       hosts. To solve this problem, we model the router as Monarch Actor, and
       pass the actor reference around.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        strategy: RoutingStrategy.Config = field(
            default_factory=LeastLoadedRoutingStrategy.Config
        )
        """Strategy for choosing a generator for a new routing session, e.g.
        ``RoundRobinRoutingStrategy.Config()`` or
        ``LeastLoadedRoutingStrategy.Config()``."""

        hot_swap: bool = True
        """When True, pulls model's state dict concurrently with in-flight
        generation (no draining). When False, each generator is drained before
        its pull.

        Draining only waits for a generator's in-flight ``_route`` call (one
        turn) to finish; between turns of a multi-turn rollout the generator is
        idle, so a weight sync may land mid-rollout and successive turns can run
        under different policy versions."""

    def __init__(
        self,
        config: Config,
        *,
        generators: Sequence[Any],
    ):
        num_actors = math.prod(current_size().values())
        assert (
            num_actors == 1
        ), f"InterGeneratorRouter must be a singleton, but its mesh holds {num_actors} actors"

        self._config = config
        self._generators = [
            _GeneratorHandle(
                actor=generator,
                rank0_actor=generator.flatten("rank").slice(rank=0),
            )
            for generator in generators
        ]
        if not self._generators:
            raise ValueError("InterGeneratorRouter requires at least one generator")
        for h in self._generators:
            h.idle.set()

        self._strategy = config.strategy.build()
        self._sessions: dict[str, _RoutingSession] = {}
        self._group_routes: dict[int, _GroupRoute] = {}
        self._sessions_by_group: dict[int, set[str]] = {}
        self._routing_state_changed = asyncio.Event()
        self._serving = asyncio.Event()
        self._refresh_serving_status()

    def _candidates(self) -> list[_GeneratorHandle]:
        """Return generator handles that are currently routable."""

        return [h for h in self._generators if h.state is _GeneratorState.SERVING]

    def _refresh_serving_status(self) -> None:
        """Update whether any generator can serve; only changes while draining (i.e. hot-swap is off)."""

        if self._candidates():
            self._serving.set()
        else:
            self._serving.clear()

    def _set_state(self, h: _GeneratorHandle, state: _GeneratorState) -> None:
        """Move a generator between serving and syncing states."""

        h.state = state
        self._refresh_serving_status()
        self._routing_state_changed.set()

    def _reserve(self, h: _GeneratorHandle, cost: int) -> None:
        """Reserve estimated generation work on a handle before dispatch."""

        if cost < 0:
            raise ValueError(f"route estimated_cost must be non-negative, got {cost}")
        if h.reserved_load == 0:
            h.idle.clear()
        h.reserved_load += cost

    def _release(self, h: _GeneratorHandle, cost: int) -> None:
        """Release estimated generation work after a routed call finishes."""

        h.reserved_load -= cost
        assert (
            h.reserved_load >= 0
        ), f"generator reserved_load went negative: {h.reserved_load}"
        if h.reserved_load == 0:
            h.idle.set()

    async def _route(
        self,
        method: str,
        *args,
        routing_ctx: RoutingContext,
        pin_session: bool = False,
        routing_group_id: int | None = None,
        **kwargs,
    ) -> Any:
        """Route a call, sharing cache salt within a group and pinning each rollout."""
        session_id = routing_ctx.session_id if pin_session else None
        session = self._sessions.get(session_id) if session_id is not None else None
        group_route = (
            self._group_routes.get(routing_group_id)
            if pin_session and routing_group_id is not None
            else None
        )
        while True:
            await self._serving.wait()
            candidates = self._candidates()
            if session is not None and any(h is session.generator for h in candidates):
                h = session.generator
                break
            if (
                session is None
                and group_route is not None
                and any(h is group_route.generator for h in candidates)
            ):
                h = group_route.generator
                break
            if session is not None:
                min_version = session.max_policy_version
            elif group_route is not None:
                min_version = group_route.cache_policy_version
            else:
                min_version = None
            eligible = [
                h
                for h in candidates
                if min_version is None
                or (h.policy_version is not None and h.policy_version >= min_version)
            ]
            if eligible:
                h = self._strategy.choose(routing_ctx, eligible)
                break
            # A replacement generator must not roll back a rollout or group.
            self._routing_state_changed.clear()
            await self._routing_state_changed.wait()
        selected_cache_policy_version = None
        if pin_session:
            if group_route is None:
                # Siblings enter concurrently, so pin the group's namespace at
                # routing without waiting for the first generation to complete.
                assert routing_group_id is not None
                assert (
                    h.policy_version is not None
                ), "generation requires an initial weight pull"
                group_route = _GroupRoute(
                    generator=h, cache_policy_version=h.policy_version
                )
                self._group_routes[routing_group_id] = group_route
            selected_cache_policy_version = (
                session.cache_policy_version
                if session is not None and h is session.generator
                else (None if session is not None else group_route.cache_policy_version)
            )
            kwargs["cache_policy_version"] = selected_cache_policy_version
            if session_id is not None and routing_group_id is not None:
                self._sessions_by_group.setdefault(routing_group_id, set()).add(
                    session_id
                )
        self._reserve(h, routing_ctx.estimated_cost)
        try:
            result = await getattr(h.rank0_actor, method).call_one(*args, **kwargs)
            if session_id is not None and (
                routing_group_id is None
                or session_id in self._sessions_by_group.get(routing_group_id, ())
            ):
                self._sessions[session_id] = _RoutingSession(
                    generator=h,
                    cache_policy_version=(
                        selected_cache_policy_version
                        if selected_cache_policy_version is not None
                        else result.min_policy_version
                    ),
                    max_policy_version=max(
                        session.max_policy_version
                        if session is not None
                        else result.max_policy_version,
                        result.max_policy_version,
                    ),
                )
            return result
        finally:
            self._release(h, routing_ctx.estimated_cost)

    async def _fanout(
        self,
        method: str,
        *args,
        return_exceptions: bool = False,
        **kwargs,
    ) -> list[Any | BaseException]:
        """Call ``method`` on every generator concurrently and gather results.

        Args:
            method: Actor endpoint name to call on every generator.
            *args: Positional arguments forwarded to each call.
            return_exceptions: If False (default), the first exception
                propagates immediately; if True, each call's exception is
                returned in the list instead of raised. Either way, a failure
                never cancels the other calls.
            **kwargs: Keyword arguments forwarded to each call.

        Returns:
            One entry per generator, in order: its result, or its exception when
            ``return_exceptions`` is True.
        """
        return await asyncio.gather(
            *[getattr(h.actor, method).call(*args, **kwargs) for h in self._generators],
            return_exceptions=return_exceptions,
        )

    async def _pull_model_state_dict(self, *, policy_version: int) -> None:
        """Pull the given policy version's state dict into every generator.

        Args:
            policy_version: Trainer policy version whose state dict to pull.
        """

        async def _pull_one(h: _GeneratorHandle) -> None:
            if self._config.hot_swap:
                # Hot swap: pull concurrently with in-flight generation, without
                # draining. Whether the pull is genuinely concurrent and safe is
                # up to the generator's implementation.
                await h.rank0_actor.pull_model_state_dict.call_one(policy_version)
            else:
                # Drain: stop routing to this generator and wait for in-flight
                # work to finish before pulling, then re-admit it.
                self._set_state(h, _GeneratorState.SYNCING)
                try:
                    with sl.log_trace_span("router_drain_wait"):
                        await h.idle.wait()
                    await h.rank0_actor.pull_model_state_dict.call_one(policy_version)
                finally:
                    self._set_state(h, _GeneratorState.SERVING)
            h.policy_version = policy_version
            self._routing_state_changed.set()

        # Start the pulls in parallel. Technically we could do rolling sync to
        # maintain availability during weight sync, but that's not a priority
        # for now.
        # TODO(perf): stagger the per-generator fetches when num_generators is large so they don't
        #   all read the trainer's CPU-staged weights at once -- bounds trainer host RAM. Matters for
        #   big models / many generators, not at small scale.
        await asyncio.gather(*[_pull_one(h) for h in self._generators])

    @concurrent_endpoint
    async def generate(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        routing_session_id: str | None,
        routing_group_id: int,
        sampling_config: Any | None,
        metrics_prefix: str,
    ) -> Any:
        """Route one generation call to a generator and return its completion."""
        return await self._route(
            "generate",
            prompt_token_ids,
            request_id=request_id,
            # VLLMGenerator.generate also requires this field for its
            # intra-mesh DP routing.
            routing_session_id=routing_session_id,
            routing_group_id=routing_group_id,
            sampling_config=sampling_config,
            metrics_prefix=metrics_prefix,
            pin_session=True,
            # Load is measured as in-flight request count (one unit per call).
            routing_ctx=RoutingContext(
                estimated_cost=1,
                session_id=routing_session_id,
            ),
        )

    def _finish_group(self, group_id: int) -> None:
        self._group_routes.pop(group_id, None)
        for session_id in self._sessions_by_group.pop(group_id, set()):
            self._sessions.pop(session_id, None)

    @concurrent_endpoint
    async def finish_group(self, group_id: int) -> None:
        """Release all routing sessions from a completed rollout group."""
        self._finish_group(group_id)

    @concurrent_endpoint
    async def start_engine_loop(self) -> None:
        """Start the engine loop on every rank of every generator."""
        await self._fanout("start_engine_loop")

    @concurrent_endpoint
    async def sync_log_step(self, step: int) -> None:
        """Set the step counter in this process and in every generator rank."""
        sl.set_step(step)
        await self._fanout("sync_log_step", step)

    @concurrent_endpoint
    async def pull_model_state_dict(self, policy_version: int) -> None:
        """Pull the given policy version's state dict into every generator."""
        # Wrapper the logic in a private method so we can test it independently
        # without the need to spawn the Monarch actor mesh.
        await self._pull_model_state_dict(policy_version=policy_version)

    @concurrent_endpoint
    async def close_generators(self) -> list[Any | BaseException]:
        """Close every generator, returning each one's result or exception."""
        return await self._fanout("close", return_exceptions=True)
