# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator routing."""

from __future__ import annotations

import asyncio
import itertools
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl


class _GeneratorState(Enum):
    """Lifecycle state controlling routability; ``SYNCING`` is only entered when draining (i.e. hot-swap is off)."""

    SERVING = auto()
    SYNCING = auto()


@dataclass(kw_only=True, slots=True)
class _GeneratorHandle:
    """Controller-side metadata for one generator mesh."""

    actor: Any
    """Monarch actor handle for the generator mesh."""

    reserved_load: int = 0
    """Controller-side estimate of in-flight routed generation work."""

    state: _GeneratorState = _GeneratorState.SERVING
    """Current routing lifecycle state for this generator."""

    idle: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when this generator has no reserved routed calls."""


@dataclass(frozen=True, kw_only=True, slots=True)
class RoutingContext:
    """Routing metadata for one generation request."""

    estimated_cost: int = 1
    """Estimated request cost used by load-aware routing strategies."""

    session_key: str | None = None
    """Sticky-routing key: requests sharing a key route to the same generator so its KV/prefix cache
    is reused (e.g. a multi-turn rollout's turns). `None` opts out (the request is routed by load)."""


class RoutingStrategy(Configurable, ABC):
    """Policy object that chooses one generator for a request.

    Add a new strategy by subclassing this, defining a nested ``Config``, and
    selecting it explicitly in config, e.g.
    ``GeneratorRouter.Config(strategy=MyRoutingStrategy.Config())``.
    """

    def __init__(self, config: Configurable.Config):
        # Stateless by default; stateful strategies override __init__.
        del config

    @abstractmethod
    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[_GeneratorHandle],
    ) -> _GeneratorHandle:
        """Choose one generator from the (non-empty) serving candidates."""


class RoundRobinRoutingStrategy(RoutingStrategy):
    """Cycle over the serving generators in order, ignoring load."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config):
        del config
        self._counter = itertools.count()

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[_GeneratorHandle],
    ) -> _GeneratorHandle:
        """Return the next serving generator in round-robin order."""

        del routing_ctx
        return candidates[next(self._counter) % len(candidates)]


class LeastLoadedRoutingStrategy(RoutingStrategy):
    """Pick the serving generator with the least controller-reserved load."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[_GeneratorHandle],
    ) -> _GeneratorHandle:
        """Return the serving generator with the lowest reserved load."""

        del routing_ctx
        return min(candidates, key=lambda handle: handle.reserved_load)


class StickySessionRoutingStrategy(RoutingStrategy):
    """Pin each session (e.g. one multi-turn rollout) to the generator it first landed on, so its turns
    reuse that generator's KV/prefix cache. Least-loaded scatters a session's turns across generators
    and misses the prefix cache; sticky keeps them together.

    Falls back to least-loaded for a session with no key, an unseen key, or when the pinned generator
    is no longer serving (e.g. draining for a non-hotswap weight sync).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    # Cap on remembered pins. Only sessions in flight need a pin (a finished session's key never
    # recurs), and at most ~num_rollout_workers are in flight; this LRU bound keeps the map
    # from growing without limit over a long run while never evicting an active session.
    _MAX_PINS = 1024

    def __init__(self, config: Config):
        del config
        # session_key -> pinned generator handle (by identity), LRU-ordered (oldest first).
        self._pinned: OrderedDict[str, _GeneratorHandle] = OrderedDict()

    def choose(
        self,
        routing_ctx: RoutingContext,
        candidates: Sequence[_GeneratorHandle],
    ) -> _GeneratorHandle:
        key = routing_ctx.session_key
        if key is not None:
            pinned = self._pinned.get(key)
            if pinned is not None and any(
                pinned is candidate for candidate in candidates
            ):
                self._pinned.move_to_end(key)  # mark recently used
                return pinned  # still serving: keep the session on its generator
        chosen = min(candidates, key=lambda handle: handle.reserved_load)
        if key is not None:
            self._pinned[
                key
            ] = chosen  # pin a new (or displaced) session to a least-loaded generator
            self._pinned.move_to_end(key)
            if len(self._pinned) > self._MAX_PINS:
                self._pinned.popitem(
                    last=False
                )  # evict the oldest (a long-finished session)
        return chosen


class GeneratorRouter(Configurable):
    """Routes generation calls across generator meshes and pulls model's state dict.

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
        """Default True: pull the state dict concurrently with in-flight generation (no draining).

        False drains each generator before its pull, which blocks on the in-flight ``route`` (a full
        turn) — so under long generations the drain dominates weight-sync time. Either way a sync can
        land between turns of a multi-turn rollout, so successive turns may run under different policy
        versions (tracked per-turn via ``version_intervals``)."""

    def __init__(
        self,
        config: Config,
        *,
        generators: Sequence[Any],
    ):
        self._config = config
        self._generators = [
            _GeneratorHandle(actor=generator) for generator in generators
        ]
        if not self._generators:
            raise ValueError("GeneratorRouter requires at least one generator")
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
        """Dispatch one call to a strategy-chosen serving generator; return its result."""

        await self._serving.wait()
        candidates = self._candidates()
        assert candidates, "serving event was set with no serving generators"
        h = self._strategy.choose(routing_ctx, candidates)
        self._reserve(h, routing_ctx.estimated_cost)
        try:
            return await getattr(h.actor, method).call(*args, **kwargs)
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
                await h.actor.pull_model_state_dict.call(policy_version)
            else:
                # Drain: stop routing to this generator and wait for in-flight
                # work to finish before pulling, then re-admit it.
                self._set_state(h, _GeneratorState.SYNCING)
                try:
                    # The drain wait IS the weight-sync cost under load: it blocks until the
                    # generator's in-flight `route` (one full turn) finishes — ~6s for a long
                    # thinking=True turn, ~ms for a short one. Spanned so the gantt shows it.
                    with sl.log_trace_span("router_drain_wait"):
                        await h.idle.wait()
                    await h.actor.pull_model_state_dict.call(policy_version)
                finally:
                    self._set_state(h, _GeneratorState.SERVING)

        # Start the pulls in parallel. Technically we could do rolling sync to
        # maintain availability during weight sync, but that's not a priority
        # for now.
        await asyncio.gather(*[_pull_one(h) for h in self._generators])
