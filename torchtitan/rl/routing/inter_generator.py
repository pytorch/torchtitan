# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Generator routing."""

from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.system import ServiceHandle
from torchtitan.rl.routing.strategies import (
    RoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.routing.types import RoutingCandidate, RoutingContext

if TYPE_CHECKING:
    from torchtitan.rl.model_state_dict_transport import (
        ModelStateDictTransport,
        TensorShard,
        TrainerSource,
    )

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class GeneratorFaultToleranceConfig:
    """Take failed generators out of service instead of failing the run.

    A generator fails as a whole when any of its ranks fails: its process dies, its
    actor or engine loop crashes, a call to it finds it gone, or its model state dict
    pull fails. The router takes it out of routing at once and stops its processes;
    training goes on with the other generators.
    """

    min_healthy_generators: int = 1
    """The run fails when a failure leaves fewer healthy generators than this. At
    least 1: with none, no rollout is generated and the trainer can't step."""

    def __post_init__(self) -> None:
        if self.min_healthy_generators < 1:
            raise ValueError(
                "fault_tolerance.min_healthy_generators must be at least 1, got "
                f"{self.min_healthy_generators}"
            )


@dataclass(kw_only=True, slots=True)
class _GeneratorHandle(RoutingCandidate):
    """Router-side metadata for one generator mesh."""

    index: int
    """The generator's index."""

    actor: Any
    """Actor handle for the full generator mesh. Used for fan-out calls that every
    rank must run."""

    rank0_actor: Any
    """Rank-0 handle of the same generator. Used for calls that only rank 0 needs
    to run."""

    state: Literal["healthy", "dead"] = "healthy"
    """``healthy`` generators are routed to. A ``dead`` one is never called again."""

    reserved_load: int = 0
    """Number of in-flight generate calls reserved on this generator."""

    draining: bool = False
    """True while the generator drains for a model state dict load (hot_swap=False only)."""


class InterGeneratorRouter(Configurable, abc.ABC):
    """Spawns the generator meshes, routes generation calls across them and pulls
    model's state dict.

    This is layer 1 of the two-layer routing design: it routes each call across
    generator *meshes* (replicas). Within the chosen mesh, ``IntraGeneratorRouter``
    then routes the request across that mesh's data-parallel ranks.

    Look-aside routing:
        The router never sees a generate call's payload. ``reserve`` picks a
        generator, adds one unit of load to it and returns its rank-0 handle; the
        caller sends the generate call there directly, then calls
        ``release_reservation``, with the call's error if it failed, so that the
        router can tell whether another generator may serve it.

    Singleton:
        Routing decisions read and write mutable states such as
        ``_GeneratorHandle.draining``, and so on. These states are not backed by
        shared storage, so if there are multiple router instances, they cannot
        know each others' routing decisions. Instead of using shared storage,
        we solve the problem by enforcing the singleton pattern:
          * there should be only 1 router mesh in a training job;
          * this mesh should consists of only 1 actor.

        This pattern is simpler to implement, and should be good enough to handle
        the RL job's scale because the router only exchanges small routing
        messages, and the number of concurrent requests should be reasonable for
        a singleton to handle.

    Owner:
        The router spawns the generators (``spawn_generators``), so their actor
        framework reports a generator's failure to the router, which takes it out
        of service when ``Config.fault_tolerance`` is set. This class decides what
        to do; the actor framework's adapter implements how to spawn and stop a
        generator (``_spawn_generator``, ``_stop_generator``) and which errors mean a
        generator is gone (``_is_generator_unavailable``).

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
        generation (no draining). When False, each generator drains before its
        pull: it finishes the requests it admitted, and holds back new ones
        until the load is applied. With ``model_state_dict_load_mode="gpu_fetch"``,
        the load includes the network transfer, so in-flight requests stall for
        it, or with draining, held-back requests wait for it.

        Draining only waits for a generator's admitted requests (one turn each)
        to finish; between turns of a multi-turn rollout the generator is
        idle, so a model state dict load may land mid-rollout and successive turns can run
        under different policy versions. A turn whose session is pinned to a
        draining generator waits for it instead of moving to another one."""

        fault_tolerance: GeneratorFaultToleranceConfig | None = None
        """Take failed generators out of service; None fails the run on any generator
        failure."""

    def __init__(self, config: Config, *, num_generators: int):
        if num_generators < 1:
            raise ValueError("InterGeneratorRouter requires at least one generator")
        self._config = config
        self._num_generators = num_generators
        self._strategy = config.strategy.build()
        self._reservations: dict[str, _GeneratorHandle] = {}
        # One handle per generator index, set by `spawn_generators`.
        self._generators: list[_GeneratorHandle] = []
        # Stops of failed generators.
        self._background_tasks: set[asyncio.Task[None]] = set()
        # Fault tolerance handles failures only while running: before the engine loops
        # start, a failure fails the setup.
        self._phase: Literal["setup", "running", "closing"] = "setup"
        # The error that ends the run, which `wait_engine_loop` raises.
        self._fatal_error: Exception | None = None
        self._fatal = asyncio.Event()

    @abc.abstractmethod
    async def _spawn_generator(self, index: int) -> ServiceHandle:
        """Spawn generator ``index`` on its own processes."""

    @abc.abstractmethod
    async def _stop_generator(self, index: int) -> None:
        """Stop generator ``index``'s processes, if any, which fails every call still
        in flight to it."""

    @abc.abstractmethod
    def _is_generator_unavailable(self, error: BaseException) -> bool:
        """Whether a call failed with ``error`` because its generator is gone, so
        another generator may serve it."""

    async def spawn_generators(self) -> None:
        """Spawn the generators; must run before any other call."""
        handles = await asyncio.gather(
            *(self._spawn_generator(i) for i in range(self._num_generators))
        )
        self._generators = [
            _GeneratorHandle(index=i, actor=handle.all_ranks, rank0_actor=handle.rank0)
            for i, handle in enumerate(handles)
        ]

    def reserve(self, request_id: str, *, routing_session_id: str | None) -> Any:
        """Pick a healthy generator for one generate call, reserve one load unit on
        it, and return its rank-0 handle.

        Raises:
            RuntimeError: once a failure fails the run.
        """
        assert (
            request_id not in self._reservations
        ), f"request_id {request_id!r} already has a reservation"
        if self._fatal_error is not None:
            raise RuntimeError(
                "generator fault tolerance gave up"
            ) from self._fatal_error

        routing_ctx = RoutingContext(session_id=routing_session_id)
        pinned = self._strategy.pinned_candidate(routing_ctx)
        # A session pinned to a failed generator is re-pinned below.
        if pinned is not None and pinned.draining and pinned.state == "healthy":
            # Stay on the draining generator, which queues the turn until its load is
            # applied; the strategy would re-pin the session just because it drains.
            h = pinned
        else:
            healthy = [g for g in self._generators if g.state == "healthy"]
            assert healthy, "the last healthy generator's failure fails the run"
            # Prefer generators that aren't draining: a draining one holds new work
            # back, which its reserved load doesn't reflect. All of them drain
            # together at the start of a load, so then use any.
            candidates = [g for g in healthy if not g.draining]
            h = self._strategy.choose(routing_ctx, candidates or healthy)
        # TODO: one load unit per call ignores the request's size, the KV that sessions
        #   hold between turns (vLLM reports only running requests' KV), and tokens
        #   generated so far. Reserve by a token or KV estimate, ideally refreshed from
        #   generator-reported KV usage, and stop sending to nearly full generators.
        h.reserved_load += 1
        self._reservations[request_id] = h
        # Returning the handle keeps callers simple. If router CPU becomes a bottleneck,
        #   return the generator index and have callers keep the handle list.
        return h.rank0_actor

    def release_reservation(
        self, request_id: str, *, error: BaseException | None = None
    ) -> bool:
        """Free the load unit ``reserve`` placed for ``request_id``. Routing
        bookkeeping only; unlike ``release_session``, it doesn't call a generator.

        Args:
            request_id: The call's request id.
            error: What the call raised, if it failed.

        Returns:
            True if the call failed because its generator is gone and fault
            tolerance is on: the generator is reported failed, and the caller may
            retry the call, which ``reserve`` routes to another generator.
        """
        h = self._reservations.pop(request_id)
        h.reserved_load -= 1
        return error is not None and self._absorb_failed_call(h, error, "generate")

    def _absorb_failed_call(
        self, h: _GeneratorHandle, error: BaseException, method: str
    ) -> bool:
        """With fault tolerance, report ``h`` failed if its ``method`` call failed with
        ``error`` because it is gone, and return True; otherwise return False, and the
        caller raises ``error``."""
        if (
            self._config.fault_tolerance is None
            or self._phase != "running"
            or not self._is_generator_unavailable(error)
        ):
            return False
        self._report_failure(h, f"{method} failed: {error!r}")
        return True

    async def _fanout(self, method: str, *args, **kwargs) -> None:
        """Call ``method`` on every rank of every live generator, concurrently.

        A failure never cancels the other calls. Without fault tolerance, the first
        error propagates at once. With it, a generator whose call finds it gone is
        reported failed, and any other error propagates once every call is done.
        """
        targets = [h for h in self._generators if h.state != "dead"]
        calls = [getattr(h.actor, method).call(*args, **kwargs) for h in targets]
        if self._config.fault_tolerance is None:
            await asyncio.gather(*calls)
            return
        results = await asyncio.gather(*calls, return_exceptions=True)
        errors = [
            result
            for h, result in zip(targets, results)
            if isinstance(result, BaseException)
            and not self._absorb_failed_call(h, result, method)
        ]
        if errors:
            raise errors[0]

    async def wait_engine_loop(self) -> None:
        """Wait for every generator rank's engine loop; raise on the first one that crashes.

        ``call()`` returns only once every rank of a mesh has replied, but when one rank's
        engine loop crashes, its TP peers stay blocked in collectives and may not reply until
        those time out. ``stream()`` yields each rank's reply as it arrives, so the first
        crash raises at once.

        With fault tolerance, a crashed generator is taken out of service instead; this
        raises once the run must fail (see ``GeneratorFaultToleranceConfig``).
        """
        if self._config.fault_tolerance is None:
            await asyncio.gather(*(_wait_engine_loop(h) for h in self._generators))
            return
        watchers = asyncio.gather(
            *(self._watch_engine_loop(i) for i in range(len(self._generators)))
        )
        fatal = asyncio.ensure_future(self._fatal.wait())
        try:
            await asyncio.wait([watchers, fatal], return_when=asyncio.FIRST_COMPLETED)
            if self._fatal_error is not None:
                raise self._fatal_error
            await watchers
        finally:
            watchers.cancel()
            fatal.cancel()
            await asyncio.gather(watchers, fatal, return_exceptions=True)

    async def _watch_engine_loop(self, index: int) -> None:
        """Wait for generator ``index``'s engine loop until it exits or fails."""
        h = self._generators[index]
        try:
            await _wait_engine_loop(h)
        except Exception as e:
            self._report_failure(h, f"its engine loop failed: {e!r}")

    async def pull_model_state_dict(self, policy_version: int) -> None:
        """Pull the given policy version's state dict into every live generator.

        With fault tolerance, a generator whose pull fails is reported failed and the
        others keep the new version.

        Args:
            policy_version: Trainer policy version whose state dict to pull.
        """

        async def _pull_one(h: _GeneratorHandle) -> None:
            # Transfer over RDMA while the generator remains available. A no-op
            # with model_state_dict_load_mode="gpu_fetch", where the load does it.
            await h.actor.prefetch_model_state_dict.call()
            drain = not self._config.hot_swap
            h.draining = drain
            try:
                await h.rank0_actor.load_model_state_dict.call_one(
                    policy_version, drain=drain
                )
            finally:
                h.draining = False

        # Start the pulls in parallel. Technically we could do rolling sync to
        # maintain availability during model state dict loads, but that's not a priority
        # for now.
        # TODO(perf): stagger the per-generator fetches when num_generators is large so they don't
        #   all read the trainer's CPU-staged weights at once -- bounds trainer host RAM. Matters for
        #   big models / many generators, not at small scale.
        targets = [h for h in self._generators if h.state != "dead"]
        fault_tolerant = self._config.fault_tolerance is not None
        results = await asyncio.gather(
            *[_pull_one(h) for h in targets], return_exceptions=fault_tolerant
        )
        if not fault_tolerant:
            return
        for h, result in zip(targets, results):
            # Any error fails the generator, since its weights are then unknown.
            if isinstance(result, BaseException):
                self._report_failure(h, f"its model state dict pull failed: {result!r}")

    async def model_state_dict_layouts(
        self,
    ) -> list[list[tuple[int, dict[str, TensorShard]]]]:
        """Each generator's ``(rank, shard layout)`` pairs, in any order."""
        results = await asyncio.gather(
            *(h.actor.model_state_dict_layout.call() for h in self._generators)
        )
        return [list(result.values()) for result in results]

    async def connect_model_state_dict_transport(
        self,
        config: ModelStateDictTransport.Config,
        *,
        trainer_sources: list[TrainerSource],
    ) -> None:
        """Connect every generator to every trainer rank's
        ``ModelStateDictTransport`` staging buffer."""

        async def connect(h: _GeneratorHandle) -> None:
            await h.actor.connect_model_state_dict_transport.call(
                config, trainer_sources=trainer_sources, generator_index=h.index
            )

        await asyncio.gather(*(connect(h) for h in self._generators))

    async def initialize_torchstore_clients(self) -> None:
        """Initialize the TorchStore client of every generator rank.

        TODO: delete once ``ModelStateDictTransport`` replaces TorchStore.
        """

        async def connect(h: _GeneratorHandle) -> None:
            await h.actor.initialize_torchstore_client.call(requester_index=h.index)

        await asyncio.gather(*(connect(h) for h in self._generators))

    async def start_engine_loop(self) -> None:
        """Start the engine loop on every rank of every generator."""
        await self._fanout("start_engine_loop")
        self._phase = "running"

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
        if pinned is None or pinned.state == "dead":
            return
        try:
            await pinned.rank0_actor.release_sessions.call_one([routing_session_id])
        except Exception as e:
            if not self._absorb_failed_call(pinned, e, "release_sessions"):
                raise

    async def release_groups(self, group_ids: list[int]) -> None:
        """Tell every generator that these rollout groups are finished."""
        await self._fanout("release_groups", group_ids)

    async def close_generators(self) -> list[Any | BaseException]:
        """Close every live generator, returning each one's result or exception (None
        for a dead one)."""
        self._phase = "closing"

        async def close(h: _GeneratorHandle) -> Any:
            if h.state == "dead":
                return None
            return await h.actor.close.call()

        return await asyncio.gather(
            *(close(h) for h in self._generators), return_exceptions=True
        )

    def _report_failure(self, h: _GeneratorHandle, reason: str) -> None:
        """Take a failed generator out of service and stop it. Idempotent: later
        reports of it are no-ops. Fault tolerance only."""
        ft = self._config.fault_tolerance
        assert ft is not None
        if self._phase == "closing" or h.state == "dead":
            return
        h.state = "dead"
        logger.warning(f"Generator {h.index} failed: {reason}")
        num_healthy = sum(g.state == "healthy" for g in self._generators)
        if num_healthy < ft.min_healthy_generators:
            self._fail(
                h.index,
                RuntimeError(
                    f"generator {h.index} failed, leaving {num_healthy} healthy "
                    f"generator(s), fewer than min_healthy_generators="
                    f"{ft.min_healthy_generators}: {reason}"
                ),
            )
            return
        task = asyncio.create_task(self._stop_generator(h.index))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _fail(self, index: int, error: Exception) -> None:
        """Fail the run with ``error``, which ``wait_engine_loop`` raises, and stop the
        failed generator ``index``, so that calls still in flight to it fail too."""
        logger.error(f"Generator fault tolerance gives up: {error}")
        if self._fatal_error is None:
            self._fatal_error = error
            self._fatal.set()
        task = asyncio.create_task(self._stop_generator(index))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)


async def _wait_engine_loop(generator: _GeneratorHandle) -> None:
    """Wait for every rank of ``generator``'s engine loop; raise as soon as one fails."""
    for reply in generator.actor.wait_engine_loop.stream():
        await reply
