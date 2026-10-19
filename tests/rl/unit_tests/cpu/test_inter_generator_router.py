# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import inspect

import pytest

from torchtitan.rl.actors.monarch.router import (
    _InterGeneratorRouterActorEndpoints,
    InterGeneratorRouterActor,
)
from torchtitan.rl.actors.system import ServiceHandle
from torchtitan.rl.generator.generation_server import GenerationServer
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
    StickySessionRoutingStrategy,
)


class _Endpoint:
    def __init__(self, value=None, *, wait: bool = False):
        self.value = value
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not wait:
            self.release.set()

    async def call_one(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.started.set()
        await self.release.wait()
        return self.value

    async def call(self, *args, **kwargs):
        return await self.call_one(*args, **kwargs)


class _Actor:
    """A single-rank generator mesh fake."""

    def __init__(
        self,
        name: str,
        *,
        wait_prefetch: bool = False,
        url: str | None = None,
    ):
        self.start_engine_loop = _Endpoint()
        self.http_url = _Endpoint(url)
        self.prefetch_model_state_dict = _Endpoint(wait=wait_prefetch)
        self.load_model_state_dict = _Endpoint(None)
        self.release_sessions = _Endpoint()
        self.release_groups = _Endpoint()

    def __len__(self):
        return 1


def _router(actors, *, strategy=None, hot_swap=False) -> InterGeneratorRouter:
    return InterGeneratorRouter(
        InterGeneratorRouter.Config(
            strategy=strategy or LeastLoadedRoutingStrategy.Config(),
            hot_swap=hot_swap,
        ),
        generators=[ServiceHandle(all_ranks=actor, rank0=actor) for actor in actors],
    )


async def _acquire(router, session_id=None, *, group_id: int = 0) -> int:
    """Take a lease and return the chosen generator's index."""
    return (await router.acquire(session_id, group_id=group_id)).index


async def _call(router, session_id=None, *, group_id: int = 0) -> int:
    """One generate call's lease: acquire, then release. Returns the chosen generator's index."""
    index = await _acquire(router, session_id, group_id=group_id)
    await router.release(index, group_id=group_id)
    return index


def test_least_loaded_acquires_the_lowest_reserved_load():
    async def _run():
        router = _router([_Actor("gen0"), _Actor("gen1")])
        router._generators[1].url = "http://localhost:41233"

        # gen0 holds a lease, so the next acquire gets gen1 and its URL.
        assert await _acquire(router) == 0
        endpoint = await router.acquire(None, group_id=0)
        assert (endpoint.index, endpoint.url) == (1, "http://localhost:41233")
        assert [h.reserved_load for h in router._generators] == [1, 1]

        await router.release(0, group_id=0)
        await router.release(1, group_id=0)
        assert [h.reserved_load for h in router._generators] == [0, 0]

    asyncio.run(_run())


def test_release_rollout_returns_the_folded_metadata_and_always_gives_the_lease_back():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        actors[1].release_sessions = _Endpoint({"trace-a": "metadata"})
        router = _router(actors)
        router._generators[0].reserved_load = 1
        assert await _acquire(router, group_id=5) == 1

        metadata = await router.release_rollout(1, group_id=5, session_ids=["trace-a"])
        assert metadata == {"trace-a": "metadata"}
        assert actors[1].release_sessions.calls == [((["trace-a"],), {})]
        assert router._generators[1].reserved_load == 0

        # A generator that fails the call (e.g. closing) still gets its lease back.
        assert await _acquire(router, group_id=5) == 1

        async def fail(session_ids):
            raise RuntimeError("generator closing")

        actors[1].release_sessions.call_one = fail
        with pytest.raises(RuntimeError, match="generator closing"):
            await router.release_rollout(1, group_id=5, session_ids=["trace-b"])
        assert router._generators[1].reserved_load == 0

    asyncio.run(_run())


def test_release_rollout_gives_the_lease_back_when_an_adapter_overrides_release():
    """On the actor, `release` is the Monarch endpoint object, which is not callable."""

    class _AdapterRouter(InterGeneratorRouter):
        release = None

    async def _run():
        actor = _Actor("gen0")
        router = _AdapterRouter(
            InterGeneratorRouter.Config(strategy=LeastLoadedRoutingStrategy.Config()),
            generators=[ServiceHandle(all_ranks=actor, rank0=actor)],
        )
        await _acquire(router, group_id=5)
        await router.release_rollout(0, group_id=5, session_ids=["trace-a"])
        assert router._generators[0].reserved_load == 0

    asyncio.run(_run())


def test_release_groups_gives_back_the_leases_its_callers_left():
    async def _run():
        router = _router([_Actor("gen0"), _Actor("gen1")])
        # Group 3's caller died holding two leases; group 4 still holds one.
        await _acquire(router, group_id=3)
        await _acquire(router, group_id=3)
        await _acquire(router, group_id=4)
        assert sum(h.reserved_load for h in router._generators) == 3

        await router.release_groups([3])
        assert sum(h.reserved_load for h in router._generators) == 1

        # A late release of a group already given back is a no-op, not a negative load.
        await router.release(0, group_id=3)
        await router.release(1, group_id=3)
        assert sum(h.reserved_load for h in router._generators) == 1

    asyncio.run(_run())


def test_start_engine_loop_checks_each_generator_url_from_the_router():
    async def _run():
        server = GenerationServer(generate=None, max_context_length=1)
        await server.start()
        try:
            router = _router([_Actor("gen0", url=server.url)])
            await router.start_engine_loop()
            assert router._generators[0].url == server.url

            unreachable = _router([_Actor("gen0", url="http://127.0.0.1:1")])
            with pytest.raises(RuntimeError, match="the router can't reach"):
                await unreachable.start_engine_loop()
        finally:
            await server.close()

    asyncio.run(_run())


def test_round_robin_cycles_through_generators():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1"), _Actor("gen2")]
        router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())

        # Cycles through all three in order, then wraps back to the first.
        assert [await _call(router) for _ in range(4)] == [0, 1, 2, 0]

    asyncio.run(_run())


def test_sticky_session_reuses_generator_for_same_session():
    async def _run():
        router = _router(
            [_Actor("gen0"), _Actor("gen1")],
            strategy=StickySessionRoutingStrategy.Config(repin_load_ratio=None),
        )

        # s0's first call is still in flight when its second arrives: both go to gen0.
        assert await _acquire(router, "s0") == 0
        assert await _acquire(router, "s0") == 0

    asyncio.run(_run())


def test_sticky_session_spreads_new_sessions_started_on_idle_generators():
    async def _run():
        router = _router(
            [_Actor("gen0"), _Actor("gen1")],
            strategy=StickySessionRoutingStrategy.Config(),
        )

        # Each call releases before the next one starts, so the least-loaded fallback
        # sees both generators idle when it places either session. The pin is
        # permanent, so the two sessions must not land on one generator.
        assert {await _call(router, "s0"), await _call(router, "s1")} == {0, 1}

    asyncio.run(_run())


def test_sticky_session_repins_off_an_overloaded_generator():
    async def _run():
        router = _router(
            [_Actor("gen0"), _Actor("gen1")],
            strategy=StickySessionRoutingStrategy.Config(),
        )

        assert await _call(router, "s0") == 0

        # gen0 piles up 3 in-flight calls vs 0 on gen1: 3 > default ratio 1.5 * mean load 1.5.
        router._generators[0].reserved_load = 3
        assert await _call(router, "s0") == 1

        # The move is a re-pin: s0 stays on gen1 once gen0 recovers.
        router._generators[0].reserved_load = 0
        assert await _call(router, "s0") == 1
        assert router._strategy.num_repins == 1

    asyncio.run(_run())


def test_sticky_session_can_use_round_robin_for_new_sessions():
    async def _run():
        router = _router(
            [_Actor("gen0"), _Actor("gen1")],
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=RoundRobinRoutingStrategy.Config()
            ),
        )

        assert [
            await _call(router, session_id) for session_id in ("s0", "s1", "s0", "s2")
        ] == [0, 1, 0, 0]

    asyncio.run(_run())


def test_sticky_session_without_session_id_uses_fallback_without_affinity():
    async def _run():
        router = _router(
            [_Actor("gen0"), _Actor("gen1")],
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=RoundRobinRoutingStrategy.Config()
            ),
        )

        assert [await _call(router) for _ in range(3)] == [0, 1, 0]

    asyncio.run(_run())


def test_sticky_session_respects_max_sessions():
    async def _run():
        router = _router(
            [_Actor("gen0"), _Actor("gen1")],
            strategy=StickySessionRoutingStrategy.Config(max_sessions=1),
        )

        # s0 is pinned to gen0 and still holds its lease, so the least-loaded
        # fallback assigns the new s1 session to gen1. Since max_sessions=1, s1 evicts s0.
        assert await _acquire(router, "s0") == 0
        assert await _call(router, "s1") == 1
        # s0 was evicted from the sticky map, so this acquire is a new-session
        # fallback. gen0 still holds s0's first lease while gen1 is idle, so
        # least-loaded picks gen1.
        assert await _call(router, "s0") == 1

    asyncio.run(_run())


def test_sticky_session_release_drops_its_assignment():
    async def _run():
        router = _router(
            [_Actor("gen0"), _Actor("gen1")],
            strategy=StickySessionRoutingStrategy.Config(),
        )
        assert await _acquire(router, "s0") == 0

        await router.release_session("s0")
        # Released, s0 is a new session: least-loaded picks idle gen1 over busy gen0.
        assert await _call(router, "s0") == 1

    asyncio.run(_run())


def test_session_release_reaches_only_its_pinned_generator():
    """The generator a session is pinned to releases its held KV."""

    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())
        assert await _call(router, "s0") == 0

        await router.release_session("s0")

        assert actors[0].release_sessions.calls == [((["s0"],), {})]
        assert actors[1].release_sessions.calls == []

    asyncio.run(_run())


def test_sticky_session_rejects_non_positive_max_sessions():
    with pytest.raises(ValueError, match="max_sessions must be positive"):
        StickySessionRoutingStrategy.Config(max_sessions=0).build()


def test_sticky_session_rejects_repin_load_ratio_at_most_one():
    with pytest.raises(ValueError, match="repin_load_ratio must be > 1"):
        StickySessionRoutingStrategy.Config(repin_load_ratio=1.0)


def test_prefetch_runs_before_the_load_and_routing_continues():
    async def _run():
        actor = _Actor("gen0", wait_prefetch=True)
        router = _router([actor])

        pull_task = asyncio.create_task(router.pull_model_state_dict(3))
        await actor.prefetch_model_state_dict.started.wait()

        assert await _call(router) == 0
        assert actor.load_model_state_dict.calls == []

        actor.prefetch_model_state_dict.release.set()
        await pull_task
        assert actor.prefetch_model_state_dict.calls == [((), {})]

    asyncio.run(_run())


@pytest.mark.parametrize("hot_swap", [False, True])
def test_pull_model_state_dict_pulls_every_generator(hot_swap):
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, hot_swap=hot_swap)

        await router.pull_model_state_dict(7)

        assert all(len(actor.prefetch_model_state_dict.calls) == 1 for actor in actors)
        # Without hot swap, each generator drains itself before it loads.
        assert [actor.load_model_state_dict.calls for actor in actors] == [
            [((7,), {"drain": not hot_swap})],
            [((7,), {"drain": not hot_swap})],
        ]

    asyncio.run(_run())


class _StreamEndpoint:
    """An endpoint whose ``stream()`` yields one reply per rank, in arrival order."""

    def __init__(self, *replies):
        self.replies = replies

    def stream(self):
        return (reply() for reply in self.replies)


def test_wait_engine_loop_raises_when_one_rank_crashes_while_its_peer_hangs():
    async def _run():
        peer_blocked = asyncio.Event()

        async def crashed():
            raise RuntimeError("engine loop crashed")

        async def blocked():
            # A TP peer of the crashed rank stays in a collective and never replies.
            await peer_blocked.wait()

        crashing, healthy = _Actor("gen0"), _Actor("gen1")
        crashing.wait_engine_loop = _StreamEndpoint(crashed, blocked)
        healthy.wait_engine_loop = _StreamEndpoint(blocked)
        router = _router([crashing, healthy])

        with pytest.raises(RuntimeError, match="engine loop crashed"):
            await asyncio.wait_for(router.wait_engine_loop(), timeout=5)

    asyncio.run(_run())


def test_wait_engine_loop_returns_once_every_rank_exits():
    async def _run():
        async def exited():
            return None

        actors = [_Actor("gen0"), _Actor("gen1")]
        for actor in actors:
            actor.wait_engine_loop = _StreamEndpoint(exited, exited)
        router = _router(actors)

        await asyncio.wait_for(router.wait_engine_loop(), timeout=5)

    asyncio.run(_run())


def test_router_actor_exposes_every_public_router_method():
    public_methods = {
        name
        for name, method in vars(InterGeneratorRouter).items()
        if inspect.iscoroutinefunction(method) and not name.startswith("_")
    }
    assert issubclass(InterGeneratorRouterActor, InterGeneratorRouter)
    assert public_methods == {
        name
        for name in vars(_InterGeneratorRouterActorEndpoints)
        if not name.startswith("_")
    }


def test_session_kv_metrics_gather_every_replica_and_count_repins():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        # One value per rank, as a mesh call returns: gen1 has two DP replicas.
        for per_rank, actor in zip([{0: [3]}, {0: [5], 1: [7]}], actors):
            actor.get_session_kv_metrics = _Endpoint(
                {
                    rank: [
                        m.Metric("generator/session_kv/held_sessions", m.Mean(value))
                        for value in values
                    ]
                    for rank, values in per_rank.items()
                }
            )
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())
        router._strategy.num_repins = 2
        first = await router.get_session_kv_metrics()
        second = await router.get_session_kv_metrics()
        return first, second

    first, second = asyncio.run(_run())

    assert m.MetricsProcessor._aggregate_metrics(first) == {
        "generator/session_kv/held_sessions/mean": 5.0,
        "generator/session_kv/repinned_sessions/sum": 2.0,
    }
    assert (
        m.MetricsProcessor._aggregate_metrics(second)[
            "generator/session_kv/repinned_sessions/sum"
        ]
        == 0.0
    )
