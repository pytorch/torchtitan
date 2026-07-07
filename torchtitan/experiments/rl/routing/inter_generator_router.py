# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator routing."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any

from torchtitan.config import Configurable
from torchtitan.experiments.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoutingStrategy,
)
from torchtitan.experiments.rl.routing.types import RoutingCandidate, RoutingContext
from torchtitan.observability import structured_logger as sl


class _GeneratorState(Enum):
    """Lifecycle state controlling routability; ``SYNCING`` is only entered when draining (i.e. hot-swap is off)."""

    SERVING = auto()
    SYNCING = auto()


@dataclass(kw_only=True, slots=True)
class _GeneratorHandle(RoutingCandidate):
    """Controller-side metadata for one generator mesh."""

    actor: Any
    """Monarch actor handle for the full generator mesh. Used for fan-out calls
    that every rank must run."""

    rank0_actor: Any
    """Cached rank-0 slice of ``actor``. Used for calls that only rank 0 needs
    to run."""

    reserved_load: int = 0
    """Controller-side estimate of in-flight routed generation work."""

    state: _GeneratorState = _GeneratorState.SERVING
    """Current routing lifecycle state for this generator."""

    idle: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when this generator has no reserved routed calls."""


class InterGeneratorRouter(Configurable):
    """Routes generation calls across generator meshes and pulls model's state dict.

    This is layer 1 of the two-layer routing design: it routes each call across
    generator *meshes* (replicas). Within the chosen mesh, ``IntraGeneratorRouter``
    then routes the request across that mesh's data-parallel ranks.

    Thread safety:
        This class is not thread-safe. Its mutable state and ``asyncio.Event``
        objects are intended to be accessed from one event loop. Do not call a
        single router instance from multiple OS threads or event loops.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        strategy: RoutingStrategy.Config = field(
            default_factory=LeastLoadedRoutingStrategy.Config
        )
        """Routing strategy, selected by its config type, e.g.
        ``RoundRobinRoutingStrategy.Config()`` or
        ``LeastLoadedRoutingStrategy.Config()``."""

        hot_swap: bool = True
        """When True, pulls model's state dict concurrently with in-flight
        generation (no draining). When False, each generator is drained before
        its pull.

        Draining only waits for a generator's in-flight ``route`` call (one
        turn) to finish; between turns of a multi-turn rollout the generator is
        idle, so a weight sync may land mid-rollout and successive turns can run
        under different policy versions."""

    def __init__(
        self,
        config: Config,
        *,
        generators: Sequence[Any],
    ):
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

    async def route(
        self,
        method: str,
        *args,
        routing_ctx: RoutingContext,
        **kwargs,
    ) -> Any:
        """Dispatch one call to a strategy-chosen serving generator's rank 0;
        return its result.
        """
        await self._serving.wait()
        candidates = self._candidates()
        assert candidates, "serving event was set with no serving generators"
        h = self._strategy.choose(routing_ctx, candidates)
        self._reserve(h, routing_ctx.estimated_cost)
        try:
            return await getattr(h.rank0_actor, method).call_one(*args, **kwargs)
        finally:
            self._release(h, routing_ctx.estimated_cost)

    async def fanout(
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

    async def pull_model_state_dict(self, *, policy_version: int) -> None:
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

        # Start the pulls in parallel. Technically we could do rolling sync to
        # maintain availability during weight sync, but that's not a priority
        # for now.
        # TODO(perf): stagger the per-generator fetches when num_generators is large so they don't
        #   all read the trainer's CPU-staged weights at once -- bounds trainer host RAM. Matters for
        #   big models / many generators, not at small scale.
        await asyncio.gather(*[_pull_one(h) for h in self._generators])
