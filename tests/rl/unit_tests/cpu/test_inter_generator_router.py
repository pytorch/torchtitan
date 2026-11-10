# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from types import SimpleNamespace

import pytest

import torchtitan.rl.routing.inter_generator as inter_generator_module
from torchtitan.rl.actors.system import ServiceHandle
from torchtitan.rl.controller.controller import Controller
from torchtitan.rl.controller.work_buffer import RolloutGroupWork
from torchtitan.rl.generator.engine import KVCacheBudget
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.routing.admission import KVGrowthEstimateAdmission
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    RoundRobinRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.routing.types import RoutingContext


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
        self.load_model_state_dict = _Endpoint(None, wait=wait_pull, raises=raises_pull)
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


def test_least_loaded_tie_break_spreads_over_a_changing_candidate_set():
    async def _run():
        actors = [_Actor(f"gen{i}") for i in range(4)]
        router = _router(actors)

        # gen1 drains for a model state dict load on every other request, so the candidate
        # set alternates between four and three generators. Each route finishes
        # before the next starts, so the survivors are always tied at zero load
        # and only the tie-break decides.
        chosen = []
        for i in range(12):
            router._generators[1].draining = i % 2 == 1
            chosen.append(await router._route("generate", routing_ctx=RoutingContext()))

        assert [chosen.count(f"gen{i}") for i in range(4)] == [3, 3, 3, 3]

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


def test_round_robin_skips_draining_generators():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())
        router._generators[0].draining = True

        assert await router._route("generate", routing_ctx=RoutingContext()) == "gen1"
        assert actors[0].generate.calls == []
        assert len(actors[1].generate.calls) == 1

    asyncio.run(_run())


def test_sticky_session_reuses_generator_for_same_session():
    async def _run():
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_generate=True),
        ]
        router = _router(
            actors, strategy=StickySessionRoutingStrategy.Config(repin_load_ratio=None)
        )

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


def test_sticky_session_stays_on_its_draining_generator():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())

        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )

        # gen0 drains for a model state dict load, with 3 turns held back vs 0 on gen1:
        # overloaded by the default 1.5x ratio, yet s0's next turn still goes to gen0.
        router._generators[0].draining = True
        router._generators[0].reserved_load = 3
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )

        # s0 is still pinned to gen0 after the drain.
        router._generators[0].draining = False
        router._generators[0].reserved_load = 0
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )

    asyncio.run(_run())


def test_new_sessions_spread_over_generators_that_all_drain():
    async def _run():
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_generate=True),
        ]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())
        for h in router._generators:
            h.draining = True

        # Every generator drains, so new sessions go to them, spread by load, and wait
        # there; none waits for the first generator to finish its load.
        routes = [
            asyncio.create_task(
                router._route(
                    "generate", routing_ctx=RoutingContext(session_id=f"s{i}")
                )
            )
            for i in range(4)
        ]
        await asyncio.sleep(0)
        assert [len(actor.generate.calls) for actor in actors] == [2, 2]

        for actor in actors:
            actor.generate.release.set()
        assert sorted(await asyncio.gather(*routes)) == ["gen0", "gen0", "gen1", "gen1"]

    asyncio.run(_run())


def test_sticky_session_repins_off_an_overloaded_generator():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())

        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen0"
        )

        # gen0 piles up 3 in-flight calls vs 0 on gen1: 3 > default ratio 1.5 * mean load 1.5.
        router._generators[0].reserved_load = 3
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
            == "gen1"
        )

        # The move is a re-pin: s0 stays on gen1 once gen0 recovers.
        router._generators[0].reserved_load = 0
        assert (
            await router._route("generate", routing_ctx=RoutingContext(session_id="s0"))
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


async def _generate(router, *, group_id: int, session_id: str):
    return await router._route(
        "generate", group_id=group_id, routing_ctx=RoutingContext(session_id=session_id)
    )


def test_sticky_session_release_drops_its_assignment():
    async def _run():
        actors = [_Actor("gen0", wait_generate=True), _Actor("gen1")]
        router = _router(actors, strategy=StickySessionRoutingStrategy.Config())
        first = asyncio.create_task(_generate(router, group_id=0, session_id="s0"))
        await actors[0].generate.started.wait()

        await router.release_session("s0")
        # Released, s0 is a new session: least-loaded picks idle gen1 over busy gen0.
        assert await _generate(router, group_id=0, session_id="s0") == "gen1"

        actors[0].generate.release.set()
        assert await first == "gen0"

    asyncio.run(_run())


def test_session_release_reaches_only_the_generators_its_calls_went_to():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1"), _Actor("gen2")]
        router = _router(actors, strategy=RoundRobinRoutingStrategy.Config())
        # Round robin sends s0's two turns to gen0 and gen1 (like a re-pinned session), s1's to gen2.
        assert await _generate(router, group_id=0, session_id="s0") == "gen0"
        assert await _generate(router, group_id=0, session_id="s0") == "gen1"
        assert await _generate(router, group_id=0, session_id="s1") == "gen2"

        await router.release_session("s0")

        assert [len(actor.release_sessions.calls) for actor in actors] == [1, 1, 0]
        assert actors[0].release_sessions.calls == [((["s0"],), {})]

        # s2 never gets its own release (a retried attempt): its group's end forgets it.
        completion = await router.generate(
            [1],
            request_id="s2/t0",
            group_id=1,
            routing_session_id="s2",
            sampling_config=None,
        )
        assert completion == "gen0"
        await router.release_groups([1])
        assert router._group_session_ids == {}
        await router.release_session("s2")
        assert len(actors[0].release_sessions.calls) == 1

    asyncio.run(_run())


def test_sticky_session_rejects_non_positive_max_sessions():
    with pytest.raises(ValueError, match="max_sessions must be positive"):
        StickySessionRoutingStrategy.Config(max_sessions=0).build()


def test_sticky_session_rejects_repin_load_ratio_at_most_one():
    with pytest.raises(ValueError, match="repin_load_ratio must be > 1"):
        StickySessionRoutingStrategy.Config(repin_load_ratio=1.0)


def test_drain_marks_generator_draining_for_its_load():
    async def _run():
        # gen0 has a route in flight, which its load drains on the generator side.
        actors = [
            _Actor("gen0", wait_generate=True, wait_pull=True),
            _Actor("gen1"),
        ]
        router = _router(actors)

        route_task = asyncio.create_task(
            router._route("generate", routing_ctx=RoutingContext())
        )
        await actors[0].generate.started.wait()

        # The router starts gen0's load without waiting for the in-flight route.
        pull_task = asyncio.create_task(router.pull_model_state_dict(1))
        await asyncio.wait_for(
            actors[0].load_model_state_dict.started.wait(), timeout=1.0
        )
        # gen1's pull doesn't wait for gen0's.
        await asyncio.wait_for(
            actors[1].load_model_state_dict.started.wait(), timeout=1.0
        )
        assert [h.draining for h in router._generators] == [True, False]
        # Even when gen1 is busier, an unpinned route skips the draining gen0.
        router._generators[1].reserved_load = 2
        assert (
            await asyncio.wait_for(
                router._route("generate", routing_ctx=RoutingContext()), timeout=1.0
            )
            == "gen1"
        )

        actors[0].generate.release.set()
        actors[0].load_model_state_dict.release.set()
        await pull_task
        assert await route_task == "gen0"
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


# Admission. Each generator holds 10 blocks of 10 tokens, a session one block per 10 tokens, and
# a max-length context (50 tokens) 5 blocks. Groups may charge 9 blocks per generator, and until 50
# sessions have ended each session counts as a max-length context: one group of one fits per
# generator.
_BUDGET = KVCacheBudget(
    num_blocks=10,
    block_size=10,
    num_growing_kv_cache_groups=1,
    fixed_blocks_per_session=0,
    max_model_len=50,
)


class _KVActor(_Actor):
    """A generator fake whose completions have ``completion_tokens`` tokens."""

    def __init__(self, name: str, *, completion_tokens: int = 10, **kwargs):
        super().__init__(name, **kwargs)
        self.generate.value = SimpleNamespace(
            name=name, token_ids=[0] * completion_tokens
        )
        self.start_engine_loop = _Endpoint()
        self.kv_cache_budget = _Endpoint(_BUDGET)


async def _admission_router(actors) -> InterGeneratorRouter:
    router = InterGeneratorRouter(
        InterGeneratorRouter.Config(
            strategy=LeastLoadedRoutingStrategy.Config(),
            admission=KVGrowthEstimateAdmission.Config(limit=0.9),
        ),
        generators=[ServiceHandle(all_ranks=actor, rank0=actor) for actor in actors],
    )
    await router.start_engine_loop()
    return router


async def _kv_generate(
    router, *, group_id: int, session_id: str, prompt_tokens: int = 20
) -> str:
    completion = await router.generate(
        [0] * prompt_tokens,
        request_id=f"{session_id}/turn",
        group_id=group_id,
        routing_session_id=session_id,
        sampling_config=None,
    )
    return completion.name


async def _until(condition) -> None:
    async def wait() -> None:
        while not condition():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=5.0)


def test_admission_places_waiting_groups_oldest_first():
    async def _run():
        router = await _admission_router([_KVActor("gen0"), _KVActor("gen1")])
        await router.acquire_group(0, group_size=1)
        await router.acquire_group(1, group_size=1)
        # A placed group's calls stay on its generator, even while it drains.
        router._generators[1].draining = True
        assert await _kv_generate(router, group_id=1, session_id="g1/r0") == "gen1"
        router._generators[1].draining = False

        # Group 5 asks before group 3; both wait, and group 3 goes first.
        waiting_5 = asyncio.create_task(router.acquire_group(5, group_size=1))
        waiting_3 = asyncio.create_task(router.acquire_group(3, group_size=1))
        await _until(lambda: len(router._waiting_groups) == 2)
        # Calls of placed groups never wait for room.
        assert await _kv_generate(router, group_id=0, session_id="g0/r0") == "gen0"

        await router.release_groups([1])
        await waiting_3
        assert not waiting_5.done()
        assert await _kv_generate(router, group_id=3, session_id="g3/r0") == "gen1"
        await router.release_groups([0])
        await waiting_5
        assert await _kv_generate(router, group_id=5, session_id="g5/r0") == "gen0"

    asyncio.run(_run())


def test_admission_counts_a_session_by_its_calls_context_length():
    async def _run():
        actor = _KVActor("gen0", completion_tokens=10)
        router = await _admission_router([actor])
        await router.acquire_group(0, group_size=1)
        # 50 sessions of one call: a 20-token prompt (2 blocks) and a 10-token reply (3 blocks).
        for i in range(50):
            await _kv_generate(router, group_id=0, session_id=f"g0/r{i}")
            await router.release_session(f"g0/r{i}")
        router._admission.charges()  # recomputes R
        assert router._admission.growth_per_session == 1.0

        # A call with a 30-token reply: the session charges its 2-block prompt plus R while the
        # call runs, then its 5 blocks plus max(0, R - 3) once it returns.
        actor.generate.value = SimpleNamespace(name="gen0", token_ids=[0] * 30)
        actor.generate.release.clear()
        call = asyncio.create_task(
            _kv_generate(router, group_id=0, session_id="g0/long")
        )
        await _until(lambda: len(actor.generate.calls) == 51)
        assert router._admission.charges() == [2 + 1.0]
        actor.generate.release.set()
        await call
        assert router._admission.charges() == [5.0]

    asyncio.run(_run())


def test_admission_ended_session_frees_room_for_the_routers_own_retry(monkeypatch):
    monkeypatch.setattr(inter_generator_module, "_ADMISSION_RETRY_S", 0.01)

    async def _run():
        router = await _admission_router([_KVActor("gen0")])
        await router.acquire_group(0, group_size=1)
        await _kv_generate(router, group_id=0, session_id="g0/r0")
        waiting = asyncio.create_task(router.acquire_group(1, group_size=1))
        await asyncio.sleep(0.05)
        assert not waiting.done()

        # Group 0's only session ends, but the group is not released yet: the router's own
        # retry places group 1.
        await router.release_session("g0/r0")
        await asyncio.wait_for(waiting, timeout=5.0)

    asyncio.run(_run())


def test_admission_cancelled_caller_keeps_its_place():
    async def _run():
        router = await _admission_router([_KVActor("gen0")])
        await router.acquire_group(0, group_size=1)
        cancelled = asyncio.create_task(router.acquire_group(1, group_size=1))
        await _until(lambda: 1 in router._waiting_groups)

        # The group stays queued, and placing it later does not fail on the cancelled call.
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        await router.release_groups([0])
        assert list(router._placed_groups) == [1]

    asyncio.run(_run())


def test_admission_routes_calls_of_an_unplaced_group_by_strategy():
    async def _run():
        router = await _admission_router([_KVActor("gen0")])
        # Calls of a group not placed, e.g. a validation group, or a training sibling still
        # running after its group failed: least-loaded routing, never charged.
        assert await _kv_generate(router, group_id=7, session_id="g7/r0") == "gen0"
        # Had it been placed, gen0 would charge 5 of 9 blocks and group 1 would wait.
        await asyncio.wait_for(router.acquire_group(1, group_size=1), timeout=1.0)

    asyncio.run(_run())


def test_admission_session_release_reaches_its_groups_generator():
    async def _run():
        actors = [_KVActor("gen0"), _KVActor("gen1")]
        router = await _admission_router(actors)
        await router.acquire_group(0, group_size=1)
        await router.acquire_group(1, group_size=1)
        assert await _kv_generate(router, group_id=1, session_id="g1/r0") == "gen1"

        await router.release_session("g1/r0")

        assert actors[0].release_sessions.calls == []
        assert actors[1].release_sessions.calls == [((["g1/r0"],), {})]

    asyncio.run(_run())


def test_admission_call_outliving_its_group_returns_its_completion():
    async def _run():
        actor = _KVActor("gen0", wait_generate=True)
        router = await _admission_router([actor])
        await router.acquire_group(0, group_size=1)
        call = asyncio.create_task(_kv_generate(router, group_id=0, session_id="g0/r0"))
        await actor.generate.started.wait()

        await router.release_groups([0])
        actor.generate.release.set()
        assert await call == "gen0"
        await asyncio.wait_for(router.acquire_group(1, group_size=1), timeout=1.0)

    asyncio.run(_run())


class _RouterHandle:
    """Stands in for the router's actor handle: each endpoint's ``call_one`` calls the router."""

    def __init__(self, router: InterGeneratorRouter):
        self.acquire_group = SimpleNamespace(call_one=router.acquire_group)
        self.release_groups = SimpleNamespace(call_one=router.release_groups)


def test_rollout_loop_acquires_a_generator_before_building_envs():
    """A waiting group holds no env or sandbox; a failed group still frees its generator."""

    async def _run():
        router = await _admission_router([_KVActor("gen0")])
        envs_built: list[int] = []
        finish_group_0 = asyncio.Event()

        async def run_group_rollouts(*, group_id: int, **kwargs) -> RolloutGroup:
            envs_built.append(group_id)
            if group_id == 0:
                await finish_group_0.wait()
                raise RuntimeError("env crashed")
            return RolloutGroup(group_id=group_id, rollouts=[])

        works = [
            RolloutGroupWork(group_id=group_id, sample=None) for group_id in (0, 1)
        ]
        finalized: list[RolloutGroup] = []

        async def claim_next() -> RolloutGroupWork | None:
            return works.pop(0) if works else None

        async def finalize_work(group: RolloutGroup) -> None:
            finalized.append(group)

        controller = SimpleNamespace(
            config=SimpleNamespace(
                async_loop=SimpleNamespace(num_samples_per_prompt=1),
                generator_router=router._config,
            ),
            generator_router=_RouterHandle(router),
            _rollouter=SimpleNamespace(run_group_rollouts=run_group_rollouts),
            _sampling=SamplingConfig(),
            rollout_recorder=SimpleNamespace(record=lambda **kwargs: None),
        )
        group_buffer = SimpleNamespace(
            claim_next=claim_next, finalize_work=finalize_work
        )
        rollout_loops = [
            asyncio.create_task(
                Controller._rollout_loop(
                    controller,
                    group_buffer=group_buffer,
                    generate_fn=None,
                    release_session_fn=None,
                )
            )
            for _ in range(2)
        ]
        await _until(lambda: 1 in router._waiting_groups)
        assert envs_built == [0]

        finish_group_0.set()
        await asyncio.gather(*rollout_loops)
        assert envs_built == [0, 1]
        assert [len(group.rollouts) for group in finalized] == [0, 0]
        assert router._placed_groups == {}

    asyncio.run(_run())
