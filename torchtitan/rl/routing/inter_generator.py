# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator routing."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.system import ServiceHandle
from torchtitan.rl.generator.generation_server import VerifiersGenerationMetadata
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.routing.strategies import (
    RoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.routing.types import RoutingCandidate, RoutingContext


@dataclass(kw_only=True, slots=True)
class _GeneratorHandle(RoutingCandidate):
    """Router-side metadata for one generator mesh."""

    actor: Any
    """Actor handle for the full generator mesh. Used for fan-out calls that every
    rank must run."""

    rank0_actor: Any
    """Rank-0 handle of the same generator. Used for calls that only rank 0 needs
    to run."""

    index: int
    """Position in the router's generator list; callers pass it back to `release` or
    `release_rollout`."""

    url: str | None = None
    """Base URL of the generator's HTTP server; set by `start_engine_loop`."""

    reserved_load: int = 0
    """Router-side estimate of in-flight routed generation work."""


@dataclass(frozen=True, slots=True)
class GeneratorEndpoint:
    """Where a caller sends a generate call, from `InterGeneratorRouter.acquire`."""

    index: int
    """Pass it back to `InterGeneratorRouter.release` or `release_rollout`."""

    url: str
    """Base URL of the generator's HTTP server, e.g. `"http://host-3.example.com:41233"`."""


class InterGeneratorRouter(Configurable):
    """Picks the generator mesh for each generate call and pulls model's state dict.

    This is layer 1 of the two-layer routing design: it picks a generator *mesh*
    (replica) per call. The caller then sends the call to that generator's HTTP
    server (`GenerationServer`), and within the mesh, `IntraGeneratorRouter`
    routes the request across its data-parallel ranks. No request body passes
    through the router:

        caller --acquire(session)--> router --> GeneratorEndpoint(index, url)
        caller --HTTP POST url/inference/v1/generate--> generator
        caller --release(index)--> router

    Singleton:
        Routing decisions read and write mutable states such as
        `_GeneratorHandle.reserved_load` and the strategy's session pins. These
        states are not backed by shared storage, so if there are multiple router
        instances, they cannot know each others' routing decisions. Instead of
        using shared storage, we solve the problem by enforcing the singleton pattern:
          * there should be only 1 router mesh in a training job;
          * this mesh should consists of only 1 actor.

        This pattern is simpler to implement, and should be good enough to handle
        the RL job's scale because the router only places calls and never carries
        their payloads.

    Actor:
       The router singleton is reached from different processes and hosts, so it
       runs as an actor and callers pass its handle around; see
       ``actors/monarch/router.py`` for the Monarch adapter, which also enforces
       the singleton.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        strategy: RoutingStrategy.Config = field(
            default_factory=StickySessionRoutingStrategy.Config
        )
        """Routing strategy, selected by its config type. The default keeps a
        session's requests (one multi-turn rollout) on one generator, so each turn
        reuses that generator's prefix KV, unless that generator is overloaded
        (``repin_load_ratio``); new sessions go to the least-loaded generator.
        Other options: ``LeastLoadedRoutingStrategy.Config()``,
        ``RoundRobinRoutingStrategy.Config()``."""

        hot_swap: bool = True
        """When True, pulls model's state dict concurrently with in-flight
        generation (no draining). When False, each generator drains itself before
        its load: it finishes the requests it already admitted, and holds new ones
        until the load lands (`Generator.load_model_state_dict(drain=True)`).

        Draining only waits for the admitted requests (one turn each) to finish;
        between turns of a multi-turn rollout the generator is idle, so a model state
        dict load may land mid-rollout and successive turns can run under different
        policy versions. A turn routed to a draining generator waits there instead of
        moving to another one."""

    def __init__(
        self,
        config: Config,
        *,
        generators: Sequence[ServiceHandle],
    ):
        self._config = config
        self._generators = [
            _GeneratorHandle(
                actor=generator.all_ranks, rank0_actor=generator.rank0, index=index
            )
            for index, generator in enumerate(generators)
        ]
        if not self._generators:
            raise ValueError("InterGeneratorRouter requires at least one generator")

        self._strategy = config.strategy.build()
        # `_strategy.num_repins` at the last `get_session_kv_metrics`.
        self._num_repins_logged = 0
        # Group id -> {generator index: leases its callers hold}. Lets `release_groups` give back
        # leases a caller never released (e.g. its process died mid-call).
        self._leases: dict[int, Counter[int]] = {}

    def _reserve(self, h: _GeneratorHandle, cost: int) -> None:
        """Reserve estimated generation work on a handle before dispatch."""

        if cost < 0:
            raise ValueError(f"route estimated_cost must be non-negative, got {cost}")
        h.reserved_load += cost

    def _release(self, h: _GeneratorHandle, cost: int) -> None:
        """Release estimated generation work after a routed call finishes."""

        h.reserved_load -= cost
        assert (
            h.reserved_load >= 0
        ), f"generator reserved_load went negative: {h.reserved_load}"

    def _release_lease(self, index: int, *, group_id: int) -> None:
        """Give back one of `group_id`'s leases on generator `index`; a no-op once `release_groups`
        freed it. `release_rollout` calls this, not `release`: on the actor, `self.release` is
        the Monarch endpoint, not this method."""
        leases = self._leases.get(group_id)
        if leases is None or leases[index] == 0:
            return
        leases[index] -= 1
        self._release(self._generators[index], cost=1)

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

    async def wait_engine_loop(self) -> None:
        """Wait for every generator rank's engine loop; raise on the first one that crashes.

        ``call()`` returns only once every rank of a mesh has replied, but when one rank's
        engine loop crashes, its TP peers stay blocked in collectives and may not reply until
        those time out. ``stream()`` yields each rank's reply as it arrives, so the first
        crash raises at once.
        """

        async def wait(generator: _GeneratorHandle) -> None:
            for reply in generator.actor.wait_engine_loop.stream():
                await reply

        await asyncio.gather(*(wait(generator) for generator in self._generators))

    async def pull_model_state_dict(self, policy_version: int) -> None:
        """Pull the given policy version's state dict into every generator.

        Args:
            policy_version: Trainer policy version whose state dict to pull.
        """

        async def _pull_one(h: _GeneratorHandle) -> None:
            # Transfer over RDMA while the generator remains available
            await h.actor.prefetch_model_state_dict.call()
            # The generator drains itself (hot_swap=False) only for the local CPU-to-GPU
            # apply, not for the network transfer above.
            await h.rank0_actor.load_model_state_dict.call_one(
                policy_version, drain=not self._config.hot_swap
            )

        # Start the pulls in parallel. Technically we could do rolling sync to
        # maintain availability during model state dict loads, but that's not a priority
        # for now.
        # TODO(perf): stagger the per-generator fetches when num_generators is large so they don't
        #   all read the trainer's CPU-staged weights at once -- bounds trainer host RAM. Matters for
        #   big models / many generators, not at small scale.
        await asyncio.gather(*[_pull_one(h) for h in self._generators])

    async def acquire(
        self, session_id: str | None, *, group_id: int
    ) -> GeneratorEndpoint:
        """Take one unit of load on the generator chosen for `session_id`; `release` (or
        `release_rollout`) gives it back.

        Load is in-flight leases:

            native rollout     one lease per generate call; the sticky strategy can re-pin between turns
            Verifiers rollout  one lease, no session, for the whole episode; its agent posts to one URL

        Args:
            session_id: Routing session for sticky placement; None places by load alone.
            group_id: Rollout group the lease belongs to; `release_groups` frees what it still holds.
        """
        h = self._strategy.choose(
            RoutingContext(estimated_cost=1, session_id=session_id), self._generators
        )
        self._reserve(h, cost=1)
        self._leases.setdefault(group_id, Counter())[h.index] += 1
        return GeneratorEndpoint(index=h.index, url=h.url)

    async def release(self, index: int, *, group_id: int) -> None:
        """Give back one unit of load that `acquire` took; a no-op once `release_groups` freed it."""
        self._release_lease(index, group_id=group_id)

    async def release_rollout(
        self, index: int, *, group_id: int, session_ids: list[str]
    ) -> dict[str, VerifiersGenerationMetadata]:
        """End a whole-rollout lease: the generator drops the rollout's sessions and returns what
        they folded. The lease goes back even if that call fails."""
        try:
            return await self._generators[index].rank0_actor.release_sessions.call_one(
                session_ids
            )
        finally:
            self._release_lease(index, group_id=group_id)

    async def start_engine_loop(self) -> None:
        """Start the engine loop on every rank of every generator, then check that this host
        reaches each generator's HTTP server."""
        await self._fanout("start_engine_loop")
        urls = await asyncio.gather(
            *(h.rank0_actor.http_url.call_one() for h in self._generators)
        )
        # Callers (rollout workers, Verifiers env servers) run on this host too, so an address
        # the router can't reach fails here, at setup, instead of on the first rollout.
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            for h, url in zip(self._generators, urls, strict=True):
                try:
                    async with session.get(f"{url}/healthz") as response:
                        response.raise_for_status()
                except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                    raise RuntimeError(
                        f"generator {h.index} serves HTTP at {url}, which the router can't reach"
                    ) from error
                h.url = url

    async def sync_log_step(self, step: int) -> None:
        """Set the step counter in this process and in every generator rank."""
        sl.set_step(step)
        await self._fanout("sync_log_step", step)

    async def release_session(self, routing_session_id: str) -> None:
        """Forget a routing session after its rollout's last generation call, and tell its
        generator to release the session's held KV."""
        pinned = self._strategy.pinned_candidate(
            RoutingContext(session_id=routing_session_id)
        )
        self._strategy.release_session(routing_session_id)
        # Without a pin (non-sticky routing), `release_groups` frees the held KV at group end.
        if pinned is not None:
            await pinned.rank0_actor.release_sessions.call_one([routing_session_id])

    async def release_groups(self, group_ids: list[int]) -> None:
        """Tell every generator that these rollout groups are finished, and give back the load
        their callers still hold."""
        for group_id in group_ids:
            for index, num_leases in self._leases.pop(group_id, Counter()).items():
                self._release(self._generators[index], cost=num_leases)
        await self._fanout("release_groups", group_ids)

    async def get_session_kv_metrics(self) -> list[m.Metric]:
        """Every generator replica's session-KV metrics, and the sessions re-pinned to another
        generator since the last call."""
        per_generator = await self._fanout("get_session_kv_metrics")
        num_repins = self._strategy.num_repins
        repinned_sessions = num_repins - self._num_repins_logged
        self._num_repins_logged = num_repins
        return [
            *(
                metric
                for per_rank in per_generator
                for metrics in per_rank.values()
                for metric in metrics
            ),
            m.Metric(
                "generator/session_kv/repinned_sessions", m.Sum(repinned_sessions)
            ),
        ]

    async def close_generators(self) -> list[Any | BaseException]:
        """Close every generator, returning each one's result or exception."""
        return await self._fanout("close", return_exceptions=True)
