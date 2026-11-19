# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from torchtitan.rl.actors.system import ServiceHandle
from torchtitan.rl.routing.inter_generator import (
    GeneratorFaultToleranceConfig,
    InterGeneratorRouter,
)
from torchtitan.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
    StickySessionRoutingStrategy,
)


class _Endpoint:
    def __init__(self, *, wait: bool = False, error: Exception | None = None):
        self.error = error
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not wait:
            self.release.set()

    async def call_one(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.started.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error

    async def call(self, *args, **kwargs):
        return await self.call_one(*args, **kwargs)


class _Actor:
    """A single-rank generator mesh fake."""

    def __init__(
        self,
        name: str,
        *,
        wait_prefetch: bool = False,
        wait_pull: bool = False,
        raises_pull: bool = False,
    ):
        self.name = name
        self.prefetch_model_state_dict = _Endpoint(wait=wait_prefetch)
        self.load_model_state_dict = _Endpoint(
            wait=wait_pull,
            error=RuntimeError("endpoint failed") if raises_pull else None,
        )
        self.release_sessions = _Endpoint()
        self.start_engine_loop = _Endpoint()
        self.release_groups = _Endpoint()
        self.close = _Endpoint()


class _Unavailable(Exception):
    """What a call to a gone generator raises in these tests."""


class _FakeRouter(InterGeneratorRouter):
    """A router whose generator ``i`` is ``actors[i]``."""

    def __init__(self, config, actors):
        super().__init__(config, num_generators=len(actors))
        self.actors = actors
        self.stopped: list[int] = []

    async def _spawn_generator(self, index):
        actor = self.actors[index]
        return ServiceHandle(all_ranks=actor, rank0=actor)

    async def _stop_generator(self, index):
        self.stopped.append(index)

    def _is_generator_unavailable(self, error):
        return isinstance(error, _Unavailable)


def _router(
    actors, *, strategy=None, hot_swap=False, fault_tolerance=None
) -> _FakeRouter:
    router = _FakeRouter(
        InterGeneratorRouter.Config(
            strategy=strategy or LeastLoadedRoutingStrategy.Config(),
            hot_swap=hot_swap,
            fault_tolerance=fault_tolerance,
        ),
        actors,
    )

    async def start():
        await router.spawn_generators()
        await router.start_engine_loop()

    # In a thread, since some tests build routers inside a running event loop.
    with ThreadPoolExecutor(1) as executor:
        executor.submit(asyncio.run, start()).result()
    return router


def _route(router, *, session_id: str | None = None) -> str:
    """Route one generate call that finishes at once; return its generator's name."""
    name = router.reserve("probe", routing_session_id=session_id).name
    router.release_reservation("probe")
    return name


def test_least_loaded_routes_to_lowest_reserved_load():
    router = _router([_Actor("gen0"), _Actor("gen1")])
    assert router.reserve("r0", routing_session_id=None).name == "gen0"
    assert router.reserve("r1", routing_session_id=None).name == "gen1"

    # Releasing r1 frees gen1, the generator it reserved.
    router.release_reservation("r1")
    assert [h.reserved_load for h in router._generators] == [1, 0]
    # The tie-break alone would pick gen0 (least recently chosen); load picks idle gen1.
    assert router.reserve("r2", routing_session_id=None).name == "gen1"

    router.release_reservation("r0")
    router.release_reservation("r2")
    assert [h.reserved_load for h in router._generators] == [0, 0]


def test_least_loaded_tie_break_spreads_over_a_changing_candidate_set():
    router = _router([_Actor(f"gen{i}") for i in range(4)])

    # gen1 drains for a model state dict load on every other request, so the candidate
    # set alternates between four and three generators. Each call finishes
    # before the next starts, so the survivors are always tied at zero load
    # and only the tie-break decides.
    chosen = []
    for i in range(12):
        router._generators[1].draining = i % 2 == 1
        chosen.append(_route(router))

    assert [chosen.count(f"gen{i}") for i in range(4)] == [3, 3, 3, 3]


def test_round_robin_cycles_through_generators():
    actors = [_Actor("gen0"), _Actor("gen1"), _Actor("gen2")]
    router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())

    # Cycles through all three in order, then wraps back to the first.
    assert [_route(router) for _ in range(4)] == ["gen0", "gen1", "gen2", "gen0"]


def test_round_robin_skips_draining_generators():
    actors = [_Actor("gen0"), _Actor("gen1")]
    router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())
    router._generators[0].draining = True

    assert [_route(router) for _ in range(2)] == ["gen1", "gen1"]


def test_sticky_session_reuses_generator_for_same_session():
    actors = [_Actor("gen0"), _Actor("gen1")]
    router = _router(
        actors, strategy=StickySessionRoutingStrategy.Config(repin_load_ratio=None)
    )

    assert router.reserve("r0", routing_session_id="s0").name == "gen0"
    # gen0 is busier than idle gen1, yet s0's next call stays on gen0.
    assert router.reserve("r1", routing_session_id="s0").name == "gen0"


def test_sticky_session_spreads_new_sessions_started_on_idle_generators():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")], strategy=StickySessionRoutingStrategy.Config()
    )

    # Each call finishes before the next one starts, so the least-loaded
    # fallback sees both generators idle when it places either session. The
    # pin is permanent, so the two sessions must not land on one generator.
    first = _route(router, session_id="s0")
    second = _route(router, session_id="s1")
    assert {first, second} == {"gen0", "gen1"}


def test_sticky_session_stays_on_its_draining_generator():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")], strategy=StickySessionRoutingStrategy.Config()
    )

    assert _route(router, session_id="s0") == "gen0"

    # gen0 drains for a model state dict load, with 3 turns held back vs 0 on gen1:
    # overloaded by the default 1.5x ratio, yet s0's next turn still goes to gen0.
    router._generators[0].draining = True
    router._generators[0].reserved_load = 3
    assert _route(router, session_id="s0") == "gen0"

    # s0 is still pinned to gen0 after the drain.
    router._generators[0].draining = False
    router._generators[0].reserved_load = 0
    assert _route(router, session_id="s0") == "gen0"


def test_new_sessions_spread_over_generators_that_all_drain():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")], strategy=StickySessionRoutingStrategy.Config()
    )
    for h in router._generators:
        h.draining = True

    # Every generator drains, so new sessions go to them anyway, spread by load.
    chosen = [
        router.reserve(f"r{i}", routing_session_id=f"s{i}").name for i in range(4)
    ]
    assert sorted(chosen) == ["gen0", "gen0", "gen1", "gen1"]


def test_sticky_session_repins_off_an_overloaded_generator():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")], strategy=StickySessionRoutingStrategy.Config()
    )

    assert _route(router, session_id="s0") == "gen0"

    # gen0 piles up 3 in-flight calls vs 0 on gen1: 3 > default ratio 1.5 * mean load 1.5.
    router._generators[0].reserved_load = 3
    assert _route(router, session_id="s0") == "gen1"

    # The move is a re-pin: s0 stays on gen1 once gen0 recovers.
    router._generators[0].reserved_load = 0
    assert _route(router, session_id="s0") == "gen1"


def test_sticky_session_can_use_round_robin_for_new_sessions():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")],
        strategy=StickySessionRoutingStrategy.Config(
            fallback_strategy=RoundRobinRoutingStrategy.Config()
        ),
    )

    assert _route(router, session_id="s0") == "gen0"
    assert _route(router, session_id="s1") == "gen1"
    assert _route(router, session_id="s0") == "gen0"
    assert _route(router, session_id="s2") == "gen0"


def test_sticky_session_without_session_id_uses_fallback_without_affinity():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")],
        strategy=StickySessionRoutingStrategy.Config(
            fallback_strategy=RoundRobinRoutingStrategy.Config()
        ),
    )

    assert [_route(router) for _ in range(3)] == ["gen0", "gen1", "gen0"]


def test_sticky_session_respects_max_sessions():
    router = _router(
        [_Actor("gen0"), _Actor("gen1")],
        # No repinning, which would also move s0 off its busy generator.
        strategy=StickySessionRoutingStrategy.Config(
            max_sessions=1, repin_load_ratio=None
        ),
    )

    assert router.reserve("r0", routing_session_id="s0").name == "gen0"
    # s0 is pinned to gen0 and still in flight, so the least-loaded fallback
    # assigns the new s1 session to gen1. Since max_sessions=1, s1 evicts s0.
    assert _route(router, session_id="s1") == "gen1"
    # Evicted, s0 is a new session: least-loaded picks idle gen1 over busy gen0.
    assert _route(router, session_id="s0") == "gen1"


def test_release_session_unpins_it_and_frees_its_kv_on_its_generator():
    actors = [_Actor("gen0"), _Actor("gen1")]
    # No repinning, which would also move s0 off its busy generator.
    router = _router(
        actors, strategy=StickySessionRoutingStrategy.Config(repin_load_ratio=None)
    )
    assert router.reserve("r0", routing_session_id="s0").name == "gen0"

    asyncio.run(router.release_session("s0"))

    assert actors[0].release_sessions.calls == [((["s0"],), {})]
    assert actors[1].release_sessions.calls == []
    # Unpinned, s0 is a new session: least-loaded picks idle gen1 over busy gen0.
    assert _route(router, session_id="s0") == "gen1"


def test_sticky_session_rejects_non_positive_max_sessions():
    with pytest.raises(ValueError, match="max_sessions must be positive"):
        StickySessionRoutingStrategy.Config(max_sessions=0).build()


def test_sticky_session_rejects_repin_load_ratio_at_most_one():
    with pytest.raises(ValueError, match="repin_load_ratio must be > 1"):
        StickySessionRoutingStrategy.Config(repin_load_ratio=1.0)


def test_drain_marks_generator_draining_for_its_load():
    async def _run():
        actors = [_Actor("gen0", wait_pull=True), _Actor("gen1")]
        router = _router(actors)

        pull_task = asyncio.create_task(router.pull_model_state_dict(1))
        await asyncio.wait_for(
            actors[0].load_model_state_dict.started.wait(), timeout=1.0
        )
        # gen1's pull doesn't wait for gen0's.
        await asyncio.wait_for(
            actors[1].load_model_state_dict.started.wait(), timeout=1.0
        )
        assert [h.draining for h in router._generators] == [True, False]
        # Even when gen1 is busier, an unpinned call skips the draining gen0.
        router._generators[1].reserved_load = 2
        assert _route(router) == "gen1"

        actors[0].load_model_state_dict.release.set()
        await pull_task
        assert [h.draining for h in router._generators] == [False, False]

    asyncio.run(_run())


def test_hot_swap_keeps_generators_serving_during_pull():
    async def _run():
        actor = _Actor("gen0", wait_pull=True)
        router = _router([actor], hot_swap=True)

        pull_task = asyncio.create_task(router.pull_model_state_dict(3))
        await actor.load_model_state_dict.started.wait()

        # Hot swap does not drain the generator, so it keeps serving.
        assert not router._generators[0].draining

        actor.load_model_state_dict.release.set()
        await pull_task
        assert actor.load_model_state_dict.calls == [((3,), {"drain": False})]

    asyncio.run(_run())


def test_prefetch_keeps_generator_serving_before_drain():
    async def _run():
        actor = _Actor("gen0", wait_prefetch=True)
        router = _router([actor])

        pull_task = asyncio.create_task(router.pull_model_state_dict(3))
        await actor.prefetch_model_state_dict.started.wait()

        assert not router._generators[0].draining
        assert actor.load_model_state_dict.calls == []

        actor.prefetch_model_state_dict.release.set()
        await pull_task
        assert actor.prefetch_model_state_dict.calls == [((), {})]

    asyncio.run(_run())


def test_drain_clears_draining_on_pull_failure():
    async def _run():
        actor = _Actor("gen0", raises_pull=True)
        router = _router([actor])

        with pytest.raises(RuntimeError, match="endpoint failed"):
            await router.pull_model_state_dict(1)

        assert not router._generators[0].draining

    asyncio.run(_run())


def test_pull_model_state_dict_pulls_every_generator():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors)

        await router.pull_model_state_dict(7)

        assert all(len(actor.prefetch_model_state_dict.calls) == 1 for actor in actors)
        assert [actor.load_model_state_dict.calls for actor in actors] == [
            [((7,), {"drain": True})],
            [((7,), {"drain": True})],
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


def test_failed_generator_leaves_routing_and_is_stopped():
    async def _run():
        router = _router(
            [_Actor("gen0"), _Actor("gen1")],
            fault_tolerance=GeneratorFaultToleranceConfig(),
        )
        old_gen0 = router._generators[0]
        assert router.reserve("r0", routing_session_id=None).name == "gen0"
        assert router.reserve("r1", routing_session_id=None).name == "gen1"
        assert router.reserve("r2", routing_session_id=None).name == "gen0"

        # E.g. the engine rejects r1's prompt: the caller raises it, nothing fails.
        assert not router.release_reservation("r1", error=ValueError("prompt too long"))
        # The calls find gen0 gone: the callers may retry, and gen0 leaves routing.
        assert router.release_reservation("r0", error=_Unavailable())
        assert router.release_reservation("r2", error=_Unavailable())
        assert [_route(router) for _ in range(2)] == ["gen1", "gen1"]

        await asyncio.gather(*router._background_tasks)
        # One stop for both reports; a later report of gen0 is a no-op.
        router._report_failure(old_gen0, "late report")
        assert router.stopped == [0]

    asyncio.run(_run())


def test_fault_tolerant_pull_fails_only_the_generator_whose_pull_fails():
    async def _run():
        router = _router(
            [_Actor("gen0", raises_pull=True), _Actor("gen1")],
            fault_tolerance=GeneratorFaultToleranceConfig(),
        )

        await router.pull_model_state_dict(1)

        assert [h.state for h in router._generators] == ["dead", "healthy"]

    asyncio.run(_run())


async def _exited():
    return None


async def _crashed():
    raise RuntimeError("engine loop crashed")


def test_wait_engine_loop_takes_a_crashed_generator_out_of_service():
    async def _run():
        gen0, gen1 = _Actor("gen0"), _Actor("gen1")
        gen0.wait_engine_loop = _StreamEndpoint(_crashed)
        gen1.wait_engine_loop = _StreamEndpoint(_exited)
        router = _router([gen0, gen1], fault_tolerance=GeneratorFaultToleranceConfig())

        # Returns once gen1 exits; gen0's crash only takes it out of service.
        await asyncio.wait_for(router.wait_engine_loop(), timeout=5)
        await asyncio.gather(*router._background_tasks)
        assert router.stopped == [0]

    asyncio.run(_run())


@pytest.mark.parametrize(
    "fault_tolerance, match",
    [
        (
            GeneratorFaultToleranceConfig(min_healthy_generators=2),
            "fewer than min_healthy_generators=2",
        ),
    ],
)
def test_wait_engine_loop_raises_when_fault_tolerance_gives_up(fault_tolerance, match):
    async def _run():
        never = asyncio.Event()

        async def blocked():
            await never.wait()

        gen0, gen1 = _Actor("gen0"), _Actor("gen1")
        gen0.wait_engine_loop = _StreamEndpoint(_crashed)
        gen1.wait_engine_loop = _StreamEndpoint(blocked)
        router = _router([gen0, gen1], fault_tolerance=fault_tolerance)

        with pytest.raises(RuntimeError, match=match):
            await asyncio.wait_for(router.wait_engine_loop(), timeout=5)
        # The failed generator is stopped, so calls in flight to it fail, and no new
        # call is routed.
        await asyncio.gather(*router._background_tasks)
        assert router.stopped == [0]
        with pytest.raises(RuntimeError, match="gave up"):
            router.reserve("r0", routing_session_id=None)

    asyncio.run(_run())


def test_fanout_reports_generators_found_gone_and_raises_other_errors():
    async def _run():
        gen0, gen1 = _Actor("gen0"), _Actor("gen1")
        gen0.release_groups = _Endpoint(error=_Unavailable())
        router = _router([gen0, gen1], fault_tolerance=GeneratorFaultToleranceConfig())

        await router.release_groups([1])
        assert [h.state for h in router._generators] == ["dead", "healthy"]

        gen1.release_groups = _Endpoint(error=ValueError("bug"))
        with pytest.raises(ValueError, match="bug"):
            await router.release_groups([2])

    asyncio.run(_run())


def test_close_generators_skips_dead_generators_and_retries_nothing():
    async def _run():
        gen0, gen1 = _Actor("gen0"), _Actor("gen1")
        router = _router([gen0, gen1], fault_tolerance=GeneratorFaultToleranceConfig())
        router._report_failure(router._generators[0], "test")
        router.reserve("r0", routing_session_id=None)

        results = await router.close_generators()

        assert results == [None, None]
        assert gen0.close.calls == [] and len(gen1.close.calls) == 1
        # Calls failing at shutdown aren't retried.
        assert not router.release_reservation("r0", error=_Unavailable())

    asyncio.run(_run())
