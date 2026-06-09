# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio

import pytest

from torchtitan.experiments.rl.generator_router import _GenState, GeneratorRouter, RouteContext


class _Endpoint:
    def __init__(self, value=None, *, wait: bool = False, raises: bool = False):
        self.value = value
        self.raises = raises
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        if not wait:
            self.release.set()

    async def call(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        self.started.set()
        await self.release.wait()
        if self.raises:
            raise RuntimeError("endpoint failed")
        return self.value


class _Actor:
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


def _router(
    actors,
    *,
    strategy: str = "least_loaded",
    hot_swap: bool = False,
) -> GeneratorRouter:
    return GeneratorRouter(
        GeneratorRouter.Config(strategy=strategy, hot_swap=hot_swap),
        generators=actors,
    )


def test_least_loaded_uses_reserved_load_for_concurrent_routes():
    async def _run():
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_generate=True),
        ]
        router = _router(actors)

        first = asyncio.create_task(
            router.route("generate", ctx=RouteContext(est_cost=3))
        )
        await actors[0].generate.started.wait()

        second = asyncio.create_task(
            router.route("generate", ctx=RouteContext(est_cost=1))
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
            await router.route("generate", ctx=RouteContext(est_cost=5))

        assert router._generators[0].reserved_load == 0
        assert router._generators[0].idle.is_set()

    asyncio.run(_run())


def test_round_robin_skips_weight_syncing_generators():
    async def _run():
        actors = [_Actor("gen0"), _Actor("gen1")]
        router = GeneratorRouter(
            GeneratorRouter.Config(strategy="round_robin"),
            generators=actors,
        )
        router._set_state(router._generators[0], _GenState.SYNCING)

        assert await router.route("generate") == "gen1"
        assert actors[0].generate.calls == []
        assert len(actors[1].generate.calls) == 1

    asyncio.run(_run())


def test_weight_sync_excludes_syncing_generator_from_routes():
    async def _run():
        actors = [
            _Actor("gen0", wait_pull=True),
            _Actor("gen1"),
        ]
        router = _router(actors)

        sync_task = asyncio.create_task(router.sync_weights(1))
        await actors[0].pull_model_state_dict.started.wait()

        assert router._generators[0].state is _GenState.SYNCING
        assert await router.route("generate") == "gen1"

        actors[0].pull_model_state_dict.release.set()
        await sync_task
        assert [h.state for h in router._generators] == [
            _GenState.SERVING,
            _GenState.SERVING,
        ]
        assert [actor.pull_model_state_dict.calls for actor in actors] == [
            [((1,), {})],
            [((1,), {})],
        ]

    asyncio.run(_run())


def test_weight_sync_pulls_idle_generators_while_busy_generator_drains():
    async def _run():
        actors = [
            _Actor("gen0", wait_generate=True),
            _Actor("gen1", wait_pull=True),
        ]
        router = _router(actors)

        route_task = asyncio.create_task(router.route("generate"))
        await actors[0].generate.started.wait()

        sync_task = asyncio.create_task(router.sync_weights(2))
        await asyncio.wait_for(
            actors[1].pull_model_state_dict.started.wait(),
            timeout=1.0,
        )

        assert [h.state for h in router._generators] == [
            _GenState.SYNCING,
            _GenState.SYNCING,
        ]
        assert actors[0].pull_model_state_dict.calls == []

        actors[1].pull_model_state_dict.release.set()
        assert await asyncio.wait_for(router.route("generate"), timeout=1.0) == "gen1"

        actors[0].generate.release.set()
        assert await route_task == "gen0"
        await sync_task
        assert [actor.pull_model_state_dict.calls for actor in actors] == [
            [((2,), {})],
            [((2,), {})],
        ]

    asyncio.run(_run())


def test_hot_swap_keeps_generators_serving_during_weight_sync():
    async def _run():
        actor = _Actor("gen0", wait_pull=True)
        router = _router([actor], hot_swap=True)

        weight_sync_task = asyncio.create_task(router.sync_weights(3))
        await actor.pull_model_state_dict.started.wait()

        assert router._generators[0].state is _GenState.SERVING
        assert await router.route("generate") == "gen0"

        actor.pull_model_state_dict.release.set()
        await weight_sync_task
        assert actor.pull_model_state_dict.calls == [((3,), {})]

    asyncio.run(_run())


def test_single_generator_blocks_routes_while_weight_syncing():
    async def _run():
        actor = _Actor("gen0", wait_pull=True)
        router = _router([actor])

        weight_sync_task = asyncio.create_task(router.sync_weights(1))
        await actor.pull_model_state_dict.started.wait()

        route_task = asyncio.create_task(router.route("generate"))
        await asyncio.sleep(0)
        assert not route_task.done()
        assert actor.generate.calls == []

        actor.pull_model_state_dict.release.set()
        await weight_sync_task

        assert await route_task == "gen0"

    asyncio.run(_run())


def test_weight_sync_restores_serving_on_pull_failure():
    async def _run():
        actor = _Actor("gen0", raises_pull=True)
        router = _router([actor])

        with pytest.raises(RuntimeError, match="endpoint failed"):
            await router.sync_weights(1)

        assert router._generators[0].state is _GenState.SERVING
        assert await router.route("generate") == "gen0"

    asyncio.run(_run())
