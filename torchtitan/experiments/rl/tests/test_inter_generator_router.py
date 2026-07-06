# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest

from torchtitan.experiments.rl.routing.inter_generator_router import (
    _GeneratorState,
    InterGeneratorRouter,
)
from torchtitan.experiments.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.experiments.rl.routing.types import RoutingContext


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


class _Actor:
    """A single-rank generator mesh fake."""

    def __init__(
        self,
        name: str,
        *,
        wait_generate: bool = False,
        wait_pull: bool = False,
        raises_pull: bool = False,
    ):
        self.generate = _Endpoint(name, wait=wait_generate)
        self.pull_model_state_dict = _Endpoint(None, wait=wait_pull, raises=raises_pull)

    def slice(self, **kwargs):
        return self

    def __len__(self):
        return 1


def _router(actors, *, strategy=None, hot_swap=False) -> InterGeneratorRouter:
    return InterGeneratorRouter(
        InterGeneratorRouter.Config(
            strategy=strategy or LeastLoadedRoutingStrategy.Config(),
            hot_swap=hot_swap,
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
            router.route_rank0("generate", routing_ctx=RoutingContext(estimated_cost=3))
        )
        await actors[0].generate.started.wait()

        # gen0 now has reserved load 3, so the next route prefers gen1.
        second = asyncio.create_task(
            router.route_rank0("generate", routing_ctx=RoutingContext(estimated_cost=1))
        )
        await actors[1].generate.started.wait()

        actors[0].generate.release.set()
        actors[1].generate.release.set()

        assert await first == "gen0"
        assert await second == "gen1"
        assert [h.reserved_load for h in router._generators] == [0, 0]
        assert all(h.idle.is_set() for h in router._generators)

    asyncio.run(_run())


def test_route_releases_reserved_load_on_failure():
    async def _run():
        actor = _Actor("gen0")
        actor.generate.raises = True
        router = _router([actor])

        with pytest.raises(RuntimeError, match="endpoint failed"):
            await router.route_rank0(
                "generate", routing_ctx=RoutingContext(estimated_cost=5)
            )

        assert router._generators[0].reserved_load == 0
        assert router._generators[0].idle.is_set()

    asyncio.run(_run())


def test_round_robin_cycles_through_generators():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1"), _Actor("gen2")]
        router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())

        results = [
            await router.route_rank0("generate", routing_ctx=RoutingContext())
            for _ in range(4)
        ]
        # Cycles through all three in order, then wraps back to the first.
        assert results == ["gen0", "gen1", "gen2", "gen0"]

    asyncio.run(_run())


def test_round_robin_skips_syncing_generators():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())
        router._set_state(router._generators[0], _GeneratorState.SYNCING)

        assert (
            await router.route_rank0("generate", routing_ctx=RoutingContext()) == "gen1"
        )
        assert actors[0].generate.calls == []
        assert len(actors[1].generate.calls) == 1

    asyncio.run(_run())


def test_sticky_session_reuses_generator_for_same_session():
    async def _run():
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_generate=True),
        ]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())

        first = asyncio.create_task(
            router.route_rank0(
                "generate",
                routing_ctx=RoutingContext(estimated_cost=3, session_id="s0"),
            )
        )
        await actors[0].generate.started.wait()

        second = asyncio.create_task(
            router.route_rank0(
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


def test_sticky_session_assigns_new_generator_when_sticky_target_is_syncing():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())

        assert (
            await router.route_rank0(
                "generate", routing_ctx=RoutingContext(session_id="s0")
            )
            == "gen0"
        )

        router._set_state(router._generators[0], _GeneratorState.SYNCING)
        assert (
            await router.route_rank0(
                "generate", routing_ctx=RoutingContext(session_id="s0")
            )
            == "gen1"
        )

        router._set_state(router._generators[0], _GeneratorState.SERVING)
        assert (
            await router.route_rank0(
                "generate", routing_ctx=RoutingContext(session_id="s0")
            )
            == "gen1"
        )

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
            await router.route_rank0(
                "generate", routing_ctx=RoutingContext(session_id="s0")
            )
            == "gen0"
        )
        assert (
            await router.route_rank0(
                "generate", routing_ctx=RoutingContext(session_id="s1")
            )
            == "gen1"
        )
        assert (
            await router.route_rank0(
                "generate", routing_ctx=RoutingContext(session_id="s0")
            )
            == "gen0"
        )
        assert (
            await router.route_rank0(
                "generate", routing_ctx=RoutingContext(session_id="s2")
            )
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

        assert (
            await router.route_rank0("generate", routing_ctx=RoutingContext()) == "gen0"
        )
        assert (
            await router.route_rank0("generate", routing_ctx=RoutingContext()) == "gen1"
        )
        assert (
            await router.route_rank0("generate", routing_ctx=RoutingContext()) == "gen0"
        )

    asyncio.run(_run())


def test_sticky_session_respects_max_sessions():
    async def _run():
        actors = [_Actor("gen0", wait_generate=True), _Actor("gen1")]
        router = _router(
            actors,
            strategy=StickySessionRoutingStrategy.Config(max_sessions=1),
        )

        first = asyncio.create_task(
            router.route_rank0("generate", routing_ctx=RoutingContext(session_id="s0"))
        )
        await actors[0].generate.started.wait()

        # s0 is pinned to gen0 and still in flight, so the least-loaded fallback
        # assigns the new s1 session to gen1. Since max_sessions=1, s1 evicts s0.
        assert (
            await router.route_rank0(
                "generate", routing_ctx=RoutingContext(session_id="s1")
            )
            == "gen1"
        )
        # s0 was evicted from the sticky map, so this route is a new-session
        # fallback. gen0 still has reserved_load from the first request, while
        # gen1 is idle, so least-loaded picks gen1.
        assert (
            await router.route_rank0(
                "generate", routing_ctx=RoutingContext(session_id="s0")
            )
            == "gen1"
        )

        actors[0].generate.release.set()
        assert await first == "gen0"

    asyncio.run(_run())


def test_sticky_session_rejects_non_positive_max_sessions():
    with pytest.raises(ValueError, match="max_sessions must be positive"):
        StickySessionRoutingStrategy.Config(max_sessions=0).build()


def test_drain_excludes_syncing_generator_from_routes():
    async def _run():
        actors = [
            _Actor("gen0", wait_pull=True),
            _Actor("gen1"),
        ]
        router = _router(actors)

        pull_task = asyncio.create_task(router.pull_model_state_dict(policy_version=1))
        await actors[0].pull_model_state_dict.started.wait()

        assert router._generators[0].state is _GeneratorState.SYNCING
        assert (
            await router.route_rank0("generate", routing_ctx=RoutingContext()) == "gen1"
        )

        actors[0].pull_model_state_dict.release.set()
        await pull_task
        assert [h.state for h in router._generators] == [
            _GeneratorState.SERVING,
            _GeneratorState.SERVING,
        ]
        assert [actor.pull_model_state_dict.calls for actor in actors] == [
            [((1,), {})],
            [((1,), {})],
        ]

    asyncio.run(_run())


def test_drain_pulls_idle_generators_while_busy_generator_drains():
    async def _run():
        # gen0 is busy generating, so its pull must wait for the in-flight route
        # to drain; gen1 is idle and starts pulling right away.
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_pull=True),
        ]
        router = _router(actors)

        route_task = asyncio.create_task(
            router.route_rank0("generate", routing_ctx=RoutingContext())
        )
        await actors[0].generate.started.wait()

        pull_task = asyncio.create_task(router.pull_model_state_dict(policy_version=2))
        await asyncio.wait_for(
            actors[1].pull_model_state_dict.started.wait(), timeout=1.0
        )

        # Both are SYNCING, but gen0's pull is still waiting for its in-flight
        # route to drain, so only gen1 (idle) has started pulling.
        assert [h.state for h in router._generators] == [
            _GeneratorState.SYNCING,
            _GeneratorState.SYNCING,
        ]
        assert actors[0].pull_model_state_dict.calls == []

        # gen1 finishes pulling and is routable again while gen0 still drains.
        actors[1].pull_model_state_dict.release.set()
        assert (
            await asyncio.wait_for(
                router.route_rank0("generate", routing_ctx=RoutingContext()),
                timeout=1.0,
            )
            == "gen1"
        )

        # Draining gen0's route lets its pull proceed; both end up pulled.
        actors[0].generate.release.set()
        assert await route_task == "gen0"
        await pull_task
        assert [actor.pull_model_state_dict.calls for actor in actors] == [
            [((2,), {})],
            [((2,), {})],
        ]

    asyncio.run(_run())


def test_hot_swap_keeps_generators_serving_during_pull():
    async def _run():
        actor = _Actor("gen0", wait_pull=True)
        router = _router([actor], hot_swap=True)

        pull_task = asyncio.create_task(router.pull_model_state_dict(policy_version=3))
        await actor.pull_model_state_dict.started.wait()

        # Hot swap does not quiesce the generator, so it keeps serving.
        assert router._generators[0].state is _GeneratorState.SERVING
        assert (
            await router.route_rank0("generate", routing_ctx=RoutingContext()) == "gen0"
        )

        actor.pull_model_state_dict.release.set()
        await pull_task
        assert actor.pull_model_state_dict.calls == [((3,), {})]

    asyncio.run(_run())


def test_single_generator_blocks_routes_while_draining():
    async def _run():
        actor = _Actor("gen0", wait_pull=True)
        router = _router([actor])

        pull_task = asyncio.create_task(router.pull_model_state_dict(policy_version=1))
        await actor.pull_model_state_dict.started.wait()

        route_task = asyncio.create_task(
            router.route_rank0("generate", routing_ctx=RoutingContext())
        )
        await asyncio.sleep(0)
        assert not route_task.done()
        assert actor.generate.calls == []

        actor.pull_model_state_dict.release.set()
        await pull_task

        assert await route_task == "gen0"

    asyncio.run(_run())


def test_drain_restores_serving_on_pull_failure():
    async def _run():
        actor = _Actor("gen0", raises_pull=True)
        router = _router([actor])

        with pytest.raises(RuntimeError, match="endpoint failed"):
            await router.pull_model_state_dict(policy_version=1)

        assert router._generators[0].state is _GeneratorState.SERVING
        assert (
            await router.route_rank0("generate", routing_ctx=RoutingContext()) == "gen0"
        )

    asyncio.run(_run())


def test_pull_model_state_dict_pulls_every_generator():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors)

        await router.pull_model_state_dict(policy_version=7)

        assert [actor.pull_model_state_dict.calls for actor in actors] == [
            [((7,), {})],
            [((7,), {})],
        ]

    asyncio.run(_run())
