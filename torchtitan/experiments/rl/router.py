# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator routing."""

from __future__ import annotations

import asyncio
import importlib
import itertools
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from typing import Any

from torchtitan.config import Configurable


class _GenState(Enum):
    """Lifecycle state that controls whether a generator can receive routes."""

    SERVING = auto()
    SYNCING = auto()


@dataclass(kw_only=True, slots=True)
class GeneratorHandle:
    """Controller-side metadata for one generator mesh."""

    actor: Any
    """Monarch actor handle for the generator mesh."""

    reserved_load: int = 0
    """Controller-side estimate of in-flight routed generation work."""

    state: _GenState = _GenState.SERVING
    """Current routing lifecycle state for this generator."""

    idle: asyncio.Event = field(default_factory=asyncio.Event)
    """Set when this generator has no reserved routed calls."""


@dataclass(frozen=True, kw_only=True, slots=True)
class RouteContext:
    """Routing metadata for one generation request."""

    est_cost: int = 1
    """Estimated request cost used by load-aware routing strategies."""


class RoutingStrategy(ABC):
    """Policy object that chooses one serving generator for a request."""

    @abstractmethod
    def choose(
        self,
        ctx: RouteContext,
        candidates: Sequence[GeneratorHandle],
    ) -> GeneratorHandle:
        """Choose one generator from the currently-serving candidates."""

        ...


_ROUTING_STRATEGIES: dict[str, type[RoutingStrategy]] = {}


def register_strategy(name: str):
    """Register a RoutingStrategy subclass under a config-visible name."""

    def _decorator(cls: type[RoutingStrategy]) -> type[RoutingStrategy]:
        """Add cls to the strategy registry and return it unchanged."""

        if name in _ROUTING_STRATEGIES:
            raise ValueError(f"routing strategy {name!r} is already registered")
        _ROUTING_STRATEGIES[name] = cls
        return cls

    return _decorator


def _load_strategy_cls(name: str) -> type[RoutingStrategy]:
    """Resolve a registered strategy name or dotted strategy class path."""

    if name in _ROUTING_STRATEGIES:
        return _ROUTING_STRATEGIES[name]
    if "." not in name:
        known = ", ".join(sorted(_ROUTING_STRATEGIES))
        raise ValueError(
            f"unknown routing strategy {name!r}; known strategies: {known}"
        )
    module_name, cls_name = name.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    if not issubclass(cls, RoutingStrategy):
        raise ValueError(f"{name!r} is not a routing strategy")
    return cls


def _build_strategy(name: str) -> RoutingStrategy:
    """Instantiate the routing strategy configured by name."""

    return _load_strategy_cls(name)()


@register_strategy("round_robin")
class _RoundRobin(RoutingStrategy):
    """Cycle over the currently-serving generators."""

    def __init__(self):
        self._counter = itertools.count()

    def choose(
        self,
        ctx: RouteContext,
        candidates: Sequence[GeneratorHandle],
    ) -> GeneratorHandle:
        """Return the next serving generator in round-robin order."""

        del ctx
        return candidates[next(self._counter) % len(candidates)]


@register_strategy("least_loaded")
class _LeastLoaded(RoutingStrategy):
    """Pick the generator with the least controller-reserved load."""

    def choose(
        self,
        ctx: RouteContext,
        candidates: Sequence[GeneratorHandle],
    ) -> GeneratorHandle:
        """Return the serving generator with the lowest reserved load."""

        del ctx
        return min(candidates, key=lambda h: h.reserved_load)


class GeneratorRouter(Configurable):
    """Routes generation calls and coordinates generator weight sync.

    Thread safety:
        This class is not thread-safe. Its mutable state and ``asyncio.Event``
        objects are intended to be accessed from one event loop. Do not call a
        single router instance from multiple OS threads or event loops.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        strategy: str = "least_loaded"
        """Routing strategy name, or a dotted strategy class path."""

        hot_swap: bool = False
        """
        Whether weight sync should update generators without draining in-flight
        generations first.
        """

    def __init__(
        self,
        config: Config,
        *,
        generators: Sequence[Any],
    ):
        self._config = config
        self._generators = [
            GeneratorHandle(actor=generator) for generator in generators
        ]
        if not self._generators:
            raise ValueError("GeneratorRouter requires at least one generator")
        for h in self._generators:
            h.idle.set()

        self._strategy = _build_strategy(config.strategy)
        self._serving = asyncio.Event()
        self._refresh_serving()

    def _candidates(self) -> list[GeneratorHandle]:
        """Return generator handles that are currently routable."""

        return [h for h in self._generators if h.state is _GenState.SERVING]

    def _refresh_serving(self) -> None:
        """Update the event that tracks whether any generator can serve work."""

        if self._candidates():
            self._serving.set()
        else:
            self._serving.clear()

    def _set_state(self, h: GeneratorHandle, state: _GenState) -> None:
        """Move a generator between serving and syncing states."""

        h.state = state
        self._refresh_serving()

    def _reserve(self, h: GeneratorHandle, cost: int) -> None:
        """Reserve estimated generation work on a handle before dispatch."""

        if cost < 0:
            raise ValueError(f"route est_cost must be non-negative, got {cost}")
        if h.reserved_load == 0:
            h.idle.clear()
        h.reserved_load += cost

    def _release(self, h: GeneratorHandle, cost: int) -> None:
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
        ctx: RouteContext | None = None,
        **kwargs,
    ):
        """Dispatch one actor call to a strategy-selected serving generator."""

        ctx = ctx or RouteContext()
        await self._serving.wait()
        candidates = self._candidates()
        assert candidates, "serving event was set with no serving generators"
        h = self._strategy.choose(ctx, candidates)

        self._reserve(h, ctx.est_cost)
        try:
            return await getattr(h.actor, method).call(*args, **kwargs)
        finally:
            self._release(h, ctx.est_cost)

    async def fanout(
        self,
        method: str,
        *args,
        return_exceptions: bool = False,
        **kwargs,
    ):
        """Call one actor method on every generator and gather the results."""

        return await asyncio.gather(
            *[getattr(h.actor, method).call(*args, **kwargs) for h in self._generators],
            return_exceptions=return_exceptions,
        )

    async def sync_weights(self, policy_version: int) -> None:
        """Sync trainer weights to all generators."""

        async def _sync_one(h: GeneratorHandle) -> None:
            if self._config.hot_swap:
                # For hot swap, router allows syncing the weights concurrently
                # with in-flight generations. Note that the actual concurrency
                # and correctness is ultimately up to the generator
                # implementation.
                await h.actor.pull_model_state_dict.call(policy_version)
            else:
                # Set the generator to syncing state to block it from serving
                # new work. But wait for any in-flight work to finish before
                # syncing the weights.
                self._set_state(h, _GenState.SYNCING)
                try:
                    await h.idle.wait()
                    await h.actor.pull_model_state_dict.call(policy_version)
                finally:
                    self._set_state(h, _GenState.SERVING)

        # Start the syncs in parallel. Technically we could do rolling sync to
        # maintain availability during weight sync, but that's not a priority
        # for now.
        results = await asyncio.gather(
            *[_sync_one(h) for h in self._generators],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result
