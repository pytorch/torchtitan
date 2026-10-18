# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Monarch actor system: the RL services as Monarch actors on this host."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import torchstore as ts
from monarch.actor import ProcMesh

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.monarch.batcher import BatcherActor
from torchtitan.rl.actors.monarch.generator import GeneratorActor
from torchtitan.rl.actors.monarch.provisioning import (
    compute_generator_world_size,
    compute_trainer_world_size,
    HostMeshes,
    shutdown,
    spawn_cpu_procs,
    spawn_proc_mesh,
)
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.router import InterGeneratorRouterActor
from torchtitan.rl.actors.monarch.torch_elastic import setup_torch_elastic_env
from torchtitan.rl.actors.monarch.trainer import TrainerActor
from torchtitan.rl.actors.system import ActorSystem, ServiceHandle, TrainerHandle
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.rollout.rollouter import RolloutWorker
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.trainer import Trainer

logger = logging.getLogger(__name__)


class MonarchActorSystem(ActorSystem):
    """Runs the RL services as Monarch actors on proc meshes.

    The trainer and each generator get their own GPU proc mesh: on this host,
    partitioned with ``CUDA_VISIBLE_DEVICES``, or on the ``host_meshes`` a
    launcher passes in. ``generator_env`` adds environment variables to the
    generator procs. The router, the batcher and the rollout worker pool get CPU
    procs on this host.

    Example:
        # A launcher that allocates its own hosts (e.g. from a Slurm job)
        actor_system = config.actor_system.build(host_meshes=host_meshes)
        await config.build().setup_async(actor_system=actor_system)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ActorSystem.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        host_meshes: HostMeshes | None = None,
        generator_env: dict[str, str] | None = None,
    ):
        self.config = config
        self._host_meshes = host_meshes
        self._generator_env = generator_env
        self._trainer_mesh: ProcMesh | None = None
        self._generator_meshes: list[ProcMesh] = []
        self._router_mesh: ProcMesh | None = None
        self._batcher_mesh: ProcMesh | None = None
        self._rollout_worker_mesh: ProcMesh | None = None
        self._trainer: Any | None = None
        self._generators: list[ServiceHandle] = []

    async def provision(
        self,
        *,
        trainer_parallelism: ParallelismConfig,
        generator_parallelism: InferenceParallelismConfig,
        num_generators: int,
    ) -> None:
        self._trainer_mesh, self._generator_meshes = spawn_proc_mesh(
            compute_trainer_world_size(trainer_parallelism),
            compute_generator_world_size(generator_parallelism),
            host_meshes=self._host_meshes,
            num_generators=num_generators,
            generator_env=self._generator_env,
        )
        # One process, so the router is a singleton and every caller reaches
        # it with `call_one`. It gets its own mesh rather than sharing the
        # controller's process so routing does not contend with the training
        # loop for the controller's GIL.
        self._router_mesh = spawn_cpu_procs(1)
        # Same for the batcher: build + pack would contend for the controller's GIL.
        # It drops its own buffers (maybe_release_oldest_batch): Monarch's drop must run on the owner's host.
        self._batcher_mesh = spawn_cpu_procs(1)

        await setup_torch_elastic_env(self._trainer_mesh)
        for generator_mesh in self._generator_meshes:
            await setup_torch_elastic_env(generator_mesh)

    async def spawn_trainer(
        self, config: Trainer.Config, *, dp_degree: int, **kwargs: Any
    ) -> TrainerHandle:
        assert self._trainer_mesh is not None, "provision before spawning"
        trainer = self._trainer_mesh.spawn("trainer", TrainerActor, config, **kwargs)
        self._trainer = trainer
        # Torch ranks follow this mesh's order and the trainer lays them out as
        # (dp, cp, tp) with no PP, so each DP rank's cp * tp ranks are consecutive.
        trainer_dp_cp_tp = trainer.flatten("rank").split(
            rank=("dp", "cp_tp"), dp=dp_degree
        )
        trainer_by_dp_rank = [
            trainer_dp_cp_tp.slice(dp=dp_rank) for dp_rank in range(dp_degree)
        ]
        # Check that layout once: if it drifts, one TP group's ranks get different columns.
        for dp_rank, trainer_at_dp_rank in enumerate(trainer_by_dp_rank):
            reported_dp_ranks = list(
                (await trainer_at_dp_rank.get_dp_rank.call()).values()
            )
            if set(reported_dp_ranks) != {dp_rank}:
                raise RuntimeError(
                    f"Trainer ranks at DP rank {dp_rank} report DP ranks "
                    f"{reported_dp_ranks}; the actor system assumes the trainer orders "
                    "ranks as (dp, cp, tp)."
                )
        return TrainerHandle(
            all_ranks=trainer,
            rank0=trainer.flatten("rank").slice(rank=0),
            by_dp_rank=trainer_by_dp_rank,
        )

    async def spawn_generator_router(
        self,
        config: InterGeneratorRouter.Config,
        *,
        generator_config: Generator.Config,
        **generator_kwargs: Any,
    ) -> Any:
        assert self._router_mesh is not None, "provision before spawning"
        for idx, generator_mesh in enumerate(self._generator_meshes):
            actor_name = (
                "generator" if len(self._generator_meshes) == 1 else f"generator_{idx}"
            )
            generator = generator_mesh.spawn(
                actor_name, GeneratorActor, generator_config, **generator_kwargs
            )
            self._generators.append(
                ServiceHandle(
                    all_ranks=generator,
                    rank0=generator.flatten("rank").slice(rank=0),
                )
            )
        return self._router_mesh.spawn(
            "generator_router",
            InterGeneratorRouterActor,
            config,
            generators=self._generators,
        )

    async def spawn_batcher(self, config: Batcher.Config, **kwargs: Any) -> Any:
        assert self._batcher_mesh is not None, "provision before spawning"
        return self._batcher_mesh.spawn("batcher", BatcherActor, config, **kwargs)

    async def spawn_rollout_workers(
        self, config: RolloutWorker.Config, *, num_workers: int, num_threads: int
    ) -> Any:
        self._rollout_worker_mesh = spawn_cpu_procs(num_workers)
        return self._rollout_worker_mesh.spawn(
            "rollout_worker",
            RolloutWorkerActor,
            worker_config=config,
            num_threads=num_threads,
            # None when structured logging is off. Else each worker calls it with its own source and rank,
            # so it writes its own JSONL next to the controller's.
            structured_logger_init_fn=sl.get_structured_logger_subprocess_init_fn(),
        )

    async def stop_rollout_workers(self) -> None:
        worker_mesh = self._rollout_worker_mesh
        self._rollout_worker_mesh = None
        if worker_mesh is not None:
            await worker_mesh.stop()

    async def initialize_torchstore(self) -> None:
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # https://github.com/meta-pytorch/torchstore
        await ts.initialize(
            mesh=self._trainer_mesh,
            strategy=ts.TorchStoreStrategy(),
            client_type=ts.ClientType.ROUTING,
        )
        # TorchStore clients are cached per process. Initialize every actor
        # rank before the role-less state-dict APIs use that cache.
        assert self._trainer is not None, "spawn the trainer first"
        await asyncio.gather(
            self._trainer.initialize_torchstore_client.call(),
            *(
                generator.all_ranks.initialize_torchstore_client.call(
                    requester_index=requester_index
                )
                for requester_index, generator in enumerate(self._generators)
            ),
        )

    async def stop(self) -> None:
        meshes = [
            mesh
            for mesh in (
                self._router_mesh,
                self._batcher_mesh,
                self._trainer_mesh,
                *self._generator_meshes,
            )
            if mesh is not None
        ]
        self._router_mesh, self._batcher_mesh = None, None
        self._trainer_mesh, self._generator_meshes = None, []
        self._trainer, self._generators = None, []
        for i, mesh in enumerate(meshes):
            try:
                await mesh.stop()
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)

    def shutdown(self) -> None:
        shutdown()
