# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest

from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.distributed.routing.types import RoutingContext


class _Endpoint:
    def __init__(self, value=None, *, wait: bool = False, raises: bool = False):
        self.value = value
        self.raises = raises
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not wait:
            self.release.set()

    async def call_one(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.started.set()
        await self.release.wait()
        if self.raises:
            raise RuntimeError("endpoint failed")
        return self.value

    async def call(self, *args, **kwargs):
        return await self.call_one(*args, **kwargs)


class _Actor:
    """A single-rank generator mesh fake."""

    def __init__(
        self,
        name: str,
        *,
        wait_generate: bool = False,
        wait_prefetch: bool = False,
        wait_pull: bool = False,
        raises_pull: bool = False,
    ):
        self.generate = _Endpoint(name, wait=wait_generate)
        self.prefetch_model_state_dict = _Endpoint(wait=wait_prefetch)
        self.pull_model_state_dict = _Endpoint(None, wait=wait_pull, raises=raises_pull)

    def flatten(self, *args, **kwargs):
        return self

    def slice(self, **kwargs):
        return self

    def __len__(self):
        return 1


def _router(actors, *, strategy=None) -> InterGeneratorRouter:
    return InterGeneratorRouter(
        InterGeneratorRouter.Config(
            strategy=strategy or LeastLoadedRoutingStrategy.Config(),
        ),
        generators=actors,
    )


def test_least_loaded_routes_to_lowest_reserved_load():
    async def _run():
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_generate=True),
        ]
        router = _router(actors)

        first = asyncio.create_task(
            router._route("generate", routing_ctx=RoutingContext(estimated_cost=3))
        )
        await actors[0].generate.started.wait()

        # gen0 now has reserved load 3, so the next route prefers gen1.
        second = asyncio.create_task(
            router._route("generate", routing_ctx=RoutingContext(estimated_cost=1))
        )
        await actors[1].generate.started.wait()

        actors[0].generate.release.set()
        actors[1].generate.release.set()

        assert await first == "gen0"
        assert await second == "gen1"
        assert [h.reserved_load for h in router._generators] == [0, 0]

    asyncio.run(_run())


def test_route_releases_reserved_load_on_failure():
    async def _run():
        actor = _Actor("gen0")
        actor.generate.raises = True
        router = _router([actor])

        with pytest.raises(RuntimeError, match="endpoint failed"):
            await router._route(
                "generate", routing_ctx=RoutingContext(estimated_cost=5)
            )

        assert router._generators[0].reserved_load == 0

    asyncio.run(_run())


def test_round_robin_cycles_through_generators():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1"), _Actor("gen2")]
        router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())

        results = [
            await router._route("generate", routing_ctx=RoutingContext())
            for _ in range(4)
        ]
        # Cycles through all three in order, then wraps back to the first.
        assert results == ["gen0", "gen1", "gen2", "gen0"]

    asyncio.run(_run())


def test_sticky_session_reuses_generator_for_same_session():
    async def _run():
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_generate=True),
        ]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())

        first = asyncio.create_task(
            router._route(
                "generate",
                routing_ctx=RoutingContext(estimated_cost=3, session_id="s0"),
            )
        )
        await actors[0].generate.started.wait()

        second = asyncio.create_task(
            router._route(
                "generate",
                routing_ctx=RoutingContext(estimated_cost=1, session_id="s0"),
            )
        )
        await asyncio.sleep(0)

        assert len(actors[0].generate.calls) == 2
        assert actors[1].generate.calls == []

        actors[0].generate.release.set()
        assert await first == "gen0"
        assert await second == "gen0"

    asyncio.run(_run())


def test_sticky_session_spreads_new_sessions_started_on_idle_generators():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())

        # Each route completes before the next one starts, so the least-loaded
        # fallback sees both generators idle when it places either session. The
        # pin is permanent, so the two sessions must not land on one generator.
        first = await router._route(
            "generate", routing_ctx=RoutingContext(session_id="s0")
        )
        second = await router._route(
            "generate", routing_ctx=RoutingContext(session_id="s1")
        )
        assert {first, second} == {"gen0", "gen1"}

    asyncio.run(_run())


def test_sticky_session_can_use_round_robin_for_new_sessions():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(
            actors,
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=RoundRobinRoutingStrategy.Config()
            ),
        )

        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s1"))
            == "gen1"
        )
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s2"))
            == "gen0"
        )

    asyncio.run(_run())


def test_sticky_session_without_session_id_uses_fallback_without_affinity():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(
            actors,
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=RoundRobinRoutingStrategy.Config()
            ),
        )

        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen0"
        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen1"
        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen0"

    asyncio.run(_run())


def test_sticky_session_respects_max_sessions():
    async def _run():
        actors = [_Actor("gen0", wait_generate=True), _Actor("gen1")]
        router = _router(
            actors,
            strategy=StickySessionRoutingStrategy.Config(max_sessions=1),
        )

        first = asyncio.create_task(
            router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
        )
        await actors[0].generate.started.wait()

        # s0 is pinned to gen0 and still in flight, so the least-loaded fallback
        # assigns the new s1 session to gen1. Since max_sessions=1, s1 evicts s0.
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s1"))
            == "gen1"
        )
        # s0 was evicted from the sticky map, so this route is a new-session
        # fallback. gen0 still has reserved_load from the first request, while
        # gen1 is idle, so least-loaded picks gen1.
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen1"
        )

        actors[0].generate.release.set()
        assert await first == "gen0"

    asyncio.run(_run())


def test_sticky_session_rejects_non_positive_max_sessions():
    with pytest.raises(ValueError, match="max_sessions must be positive"):
        StickySessionRoutingStrategy.Config(max_sessions=0).build()


def test_pull_model_state_dict_pulls_every_generator():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors)

        await router._pull_model_state_dict(policy_version=7)

        assert all(len(actor.prefetch_model_state_dict.calls) == 1 for actor in actors)
        assert [actor.pull_model_state_dict.calls for actor in actors] == [
            [((7,), {})],
            [((7,), {})],
        ]

    asyncio.run(_run())


def test_sticky_session_stays_on_its_generator_during_a_pull():
    async def _run():
        # The router never takes a generator out of rotation: a generator that drains
        # before its pull holds the turn itself, so the session keeps its generator.
        actors = [_Actor("gen0", wait_pull=True), _Actor("gen1")]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )

        pull_task = asyncio.create_task(router._pull_model_state_dict(policy_version=3))
        await actors[0].pull_model_state_dict.started.wait()
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )

        actors[0].pull_model_state_dict.release.set()
        await pull_task
        assert [actor.prefetch_model_state_dict.calls for actor in actors] == [
            [((), {})],
            [((), {})],
        ]

    asyncio.run(_run())


def test_pull_failure_propagates():
    async def _run():
        router = _router([_Actor("gen0", raises_pull=True)])
        with pytest.raises(RuntimeError, match="endpoint failed"):
            await router._pull_model_state_dict(policy_version=1)
        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen0"

    asyncio.run(_run())
