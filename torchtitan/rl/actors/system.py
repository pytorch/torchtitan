# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The actor framework interface the RL controller places its services through."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from torchtitan.config import Configurable

if TYPE_CHECKING:
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.rl.generator.generator import Generator
    from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
    from torchtitan.rl.rollout.rollouter import RolloutWorker
    from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
    from torchtitan.rl.trainer.batcher import Batcher
    from torchtitan.rl.trainer.trainer import Trainer


@dataclass(frozen=True, slots=True)
class ServiceHandle:
    """Handles to the actors of one SPMD service: the trainer, or one generator."""

    all_ranks: Any
    """Endpoint calls on this reach every rank, e.g. ``all_ranks.push_model_state_dict.call()``."""

    rank0: Any
    """Endpoint calls on this reach rank 0 only, e.g. ``rank0.generate.call_one(...)``."""


@dataclass(frozen=True, slots=True)
class TrainerHandle(ServiceHandle):
    """Handles to the trainer's actors, also grouped by data-parallel rank."""

    by_dp_rank: list[Any]
    """``by_dp_rank[i]`` reaches the pp * cp * tp actors at DP rank ``i``, which train
    the batcher's column ``i``, e.g. ``by_dp_rank[i].forward_backward.call(...)``."""


class ActorSystem(Configurable, abc.ABC):
    """Places the RL services on processes and returns handles to them.

    The services (``Trainer``, ``Generator``, ``InterGeneratorRouter``,
    ``Batcher``, ``RolloutWorker``) are framework-free classes; an actor system
    runs each in its framework's actors. The controller builds it from
    ``Controller.Config.actor_system`` (or takes one its launcher built) and
    drives it: ``provision`` once, then a ``spawn_*`` call per service, and
    ``stop`` on close. The rollouter spawns and stops its worker pool through it.

    The name follows Akka/Pekko, where an actor system is the container that
    creates actors, places them and stops them
    (https://pekko.apache.org/docs/pekko/1.3/general/actor-systems.html). This
    class has that role for the RL services: it owns where they run
    (``provision``) and their lifecycle (``spawn_*``, ``stop``, ``shutdown``),
    not only how actors execute, which a name like "runtime" would suggest.

    Handles follow one call convention. An endpoint ``handle.<method>`` is called
    with:

    - ``.call(...)``: on every actor of the handle; the result holds each rank's
      value, read with ``.get(rank)``;
    - ``.call_one(...)``: on a single-actor handle; returns its value;
    - ``.choose(...)``: on any one actor of the handle;
    - ``.stream(...)``: on every actor, yielding each rank's value as it arrives;
    - ``.broadcast(...)``: on every actor, without waiting for or returning a result.

    These are Monarch's endpoint calls; another framework's actor system wraps its
    handles to provide them.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    @abc.abstractmethod
    async def provision(
        self,
        *,
        trainer_parallelism: ParallelismConfig,
        generator_parallelism: InferenceParallelismConfig,
        num_generators: int,
    ) -> None:
        """Reserve the processes for the trainer, ``num_generators`` generators, the
        generator router and the batcher, before any of them is spawned."""

    @abc.abstractmethod
    async def spawn_trainer(
        self, config: Trainer.Config, *, dp_degree: int, **kwargs: Any
    ) -> TrainerHandle:
        """Spawn the trainer, whose ranks form ``dp_degree`` DP ranks; ``kwargs`` go to
        ``Trainer``'s constructor."""

    @abc.abstractmethod
    async def spawn_generator_router(
        self,
        config: InterGeneratorRouter.Config,
        *,
        generator_config: Generator.Config,
        **generator_kwargs: Any,
    ) -> Any:
        """Spawn the provisioned generators and the singleton router over them;
        returns the router's single-actor handle, the only way to reach the
        generators. ``generator_kwargs`` go to ``Generator``'s constructor."""

    @abc.abstractmethod
    async def spawn_batcher(self, config: Batcher.Config, **kwargs: Any) -> Any:
        """Spawn the batcher in its own process; returns a single-actor handle.
        ``kwargs`` go to ``Batcher``'s constructor."""

    @abc.abstractmethod
    async def spawn_rollout_workers(
        self, config: RolloutWorker.Config, *, num_workers: int, num_threads: int
    ) -> Any:
        """Spawn a pool of ``num_workers`` rollout worker processes on the
        controller host, each running ``num_threads`` rollout threads."""

    @abc.abstractmethod
    async def stop_rollout_workers(self) -> None:
        """Stop the rollout worker pool, if any."""

    @abc.abstractmethod
    async def initialize_torchstore(self) -> None:
        """Initialize TorchStore, with its storage colocated with the trainer, and the
        TorchStore client of every trainer and generator actor.

        TODO: move to a model state dict transport, which owns its placement.
        """

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop every process ``provision`` reserved, best-effort."""

    @abc.abstractmethod
    def shutdown(self) -> None:
        """Release the framework's client in this process, after the event loop
        has finished."""
