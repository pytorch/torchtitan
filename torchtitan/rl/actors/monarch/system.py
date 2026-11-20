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
from monarch.actor import ProcMesh, this_host

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.observability import structured_logger as sl
from torchtitan.rl._env import apply_rollout_worker_env_defaults
from torchtitan.rl.actors.monarch.batcher import BatcherActor
from torchtitan.rl.actors.monarch.generator import GeneratorActor
from torchtitan.rl.actors.monarch.provisioning import (
    batcher_host,
    compute_generator_world_size,
    compute_trainer_world_size,
    generator_role_names,
    HostMeshes,
    rollout_worker_hosts,
    shutdown,
    spawn_proc_mesh,
)
from torchtitan.rl.actors.monarch.rollout_worker import RolloutWorkerActor
from torchtitan.rl.actors.monarch.router import InterGeneratorRouterActor
from torchtitan.rl.actors.monarch.torch_elastic import setup_torch_elastic_env
from torchtitan.rl.actors.monarch.trainer import TrainerActor
from torchtitan.rl.actors.system import (
    ActorSystem,
    RolloutWorkersHandle,
    ServiceHandle,
    TrainerHandle,
)
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.model_state_dict_transport import (
    ModelStateDictTransport,
    shared_rank_layouts,
)
from torchtitan.rl.rollout.rollouter import BaseRolloutWorker, RolloutWorkerRoles
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.trainer.batcher import Batcher
from torchtitan.rl.trainer.trainer import Trainer

logger = logging.getLogger(__name__)


class MonarchActorSystem(ActorSystem):
    """Runs the RL services as Monarch actors on proc meshes.

    The trainer and each generator get their own GPU proc mesh: on this host,
    partitioned with ``CUDA_VISIBLE_DEVICES``, or on the ``host_meshes`` a
    launcher passes in (e.g. ``MonarchControllerWithLauncher`` in
    ``launcher/base.py``, with the hosts of a submitted job). The router gets a CPU
    proc on this host. The batcher gets one on this host, or with RDMA microbatches on
    the host ``batcher_host`` picks. The rollout workers spread over the hosts that
    ``rollout_worker_hosts`` picks for the rollouter's ``worker_roles``.

    Example: a ``Controller`` subclass that allocated its hosts as
    ``self.host_meshes``::

        async def _build_actor_system(self) -> MonarchActorSystem:
            return self.config.actor_system.build(host_meshes=self.host_meshes)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ActorSystem.Config):
        # TODO: derive this default from each host's CPU count, so it fits each machine. Needs the
        #   launcher to report each host's cores (Slurm: `SLURM_JOB_CPUS_PER_NODE`). Monarch has no
        #   CPU query before spawning.
        max_rollout_workers_per_host: int = 128
        """The most rollout workers one host may run. Setup raises when `rollouter.worker_roles` would
        put more on one host. It catches a pool sized by mistake. Example:
        `rollouter.rollouts_per_worker=1` with 256 training rollouts in flight asks for 256 workers.
        Each holds ~0.8 GiB of private memory after setup, so 128 on one host take ~100 GiB."""

    def __init__(
        self,
        config: Config,
        *,
        host_meshes: HostMeshes | None = None,
    ):
        self.config = config
        self._host_meshes = host_meshes
        self._trainer_mesh: ProcMesh | None = None
        self._generator_meshes: list[ProcMesh] = []
        self._router_mesh: ProcMesh | None = None
        self._batcher_mesh: ProcMesh | None = None
        # One proc mesh per host that runs rollout workers.
        self._rollout_worker_meshes: list[ProcMesh] = []
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
        )
        # One process, so the router is a singleton and every caller reaches
        # it with `call_one`. It gets its own mesh rather than sharing the
        # controller's process so routing does not contend with the training
        # loop for the controller's GIL.
        self._router_mesh = this_host().spawn_procs(per_host={"cpus": 1})

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
        # (pp, dp, cp, tp), so each DP rank owns cp * tp consecutive ranks in
        # every pipeline stage.
        trainer_pp_dp_cp_tp = trainer.flatten("rank").split(
            rank=("pp", "dp", "cp_tp"),
            pp=config.parallelism.pipeline_parallel_degree,
            dp=dp_degree,
        )
        trainer_by_dp_rank = [
            trainer_pp_dp_cp_tp.slice(dp=dp_rank) for dp_rank in range(dp_degree)
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
                    "ranks as (pp, dp, cp, tp)."
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
        actor_names = generator_role_names(len(self._generator_meshes))
        for actor_name, generator_mesh in zip(actor_names, self._generator_meshes):
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
        # Its own process: build + pack would contend for the controller's GIL.
        # It drops its own buffers (maybe_release_oldest_batch): Monarch's drop must run on the owner's host.
        # Without RDMA, each batch goes back to the controller, so the batcher stays on the controller's host.
        host = (
            this_host()
            if config.rdma_min_tensor_bytes is None
            else batcher_host(host_meshes=self._host_meshes)
        )
        self._batcher_mesh = host.spawn_procs(per_host={"cpus": 1})
        batcher = self._batcher_mesh.spawn("batcher", BatcherActor, config, **kwargs)
        logger.info(
            "The batcher runs on host %s", await batcher.get_hostname.call_one()
        )
        return batcher

    async def spawn_rollout_workers(
        self,
        config: BaseRolloutWorker.Config,
        *,
        num_workers: int,
        num_threads: int,
        worker_roles: RolloutWorkerRoles,
    ) -> RolloutWorkersHandle:
        # ======== Pick the hosts and each host's worker count ========
        hosts = rollout_worker_hosts(
            host_meshes=self._host_meshes, worker_roles=worker_roles
        )
        # Split the workers evenly. The first hosts take one extra each until none are left.
        # 5 workers on 3 hosts: 5 // 3 = 1 each, and 5 % 3 = 2 extra -> [2, 2, 1].
        min_workers_per_host, num_extra_workers = divmod(num_workers, len(hosts))
        most_workers_on_a_host = min_workers_per_host + (1 if num_extra_workers else 0)
        if most_workers_on_a_host > self.config.max_rollout_workers_per_host:
            raise ValueError(
                f"rollouter.worker_roles ({worker_roles}) puts {most_workers_on_a_host} of the "
                f"{num_workers} rollout workers on one of {len(hosts)} host(s), more than "
                "actor_system.max_rollout_workers_per_host "
                f"({self.config.max_rollout_workers_per_host}). Please raise "
                "actor_system.max_rollout_workers_per_host or rollouter.rollouts_per_worker, or "
                "set more roles to True in rollouter.worker_roles."
            )

        # ======== Spawn each host's workers ========
        # Two views of the same workers:
        #   by_host[i]   every worker on host i, one `.call()` for all of them (setup, log step)
        #   by_worker[j] worker j alone, for `.call_one()`: the rollouter sends each group to one worker
        by_host: list[Any] = []
        by_worker: list[Any] = []
        for host_index, host in enumerate(hosts):
            num_host_workers = min_workers_per_host + (
                1 if host_index < num_extra_workers else 0
            )
            if num_host_workers == 0:  # fewer workers than hosts
                continue
            procs = host.spawn_procs(
                per_host={"cpus": num_host_workers},
                bootstrap=apply_rollout_worker_env_defaults,
            )
            # Save the mesh right away, so `stop_rollout_workers` stops it even if a later host
            # fails to spawn.
            self._rollout_worker_meshes.append(procs)
            host_workers = procs.spawn(
                "rollout_worker",
                RolloutWorkerActor,
                worker_config=config,
                num_threads=num_threads,
                structured_logger_init_fn=sl.get_structured_logger_subprocess_init_fn(),
                first_worker_index=len(by_worker),
            )
            by_host.append(host_workers)
            by_worker += [
                host_workers.flatten("rank").slice(rank=rank)
                for rank in range(num_host_workers)
            ]

        logger.info(
            "Rollout workers per host: %s",
            [len(host_workers) for host_workers in by_host],
        )
        return RolloutWorkersHandle(by_host=by_host, by_worker=by_worker)

    async def stop_rollout_workers(self) -> None:
        meshes, self._rollout_worker_meshes = self._rollout_worker_meshes, []

        async def stop_mesh(mesh: ProcMesh) -> None:
            await mesh.stop()

        # Stop the worker meshes of all hosts at once. `stop_mesh` wraps each call in a coroutine: if one
        # `mesh.stop()` raises right away, `gather` still runs the others.
        results = await asyncio.gather(
            *(stop_mesh(mesh) for mesh in meshes), return_exceptions=True
        )
        for mesh_index, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(
                    "rollout worker mesh %d failed to stop", mesh_index, exc_info=result
                )

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

    async def initialize_model_state_dict_transport(
        self, config: ModelStateDictTransport.Config
    ) -> None:
        assert self._trainer is not None, "spawn the trainer first"
        results = await asyncio.gather(
            *(
                generator.all_ranks.model_state_dict_layout.call()
                for generator in self._generators
            )
        )
        generator_rank_layouts = shared_rank_layouts(
            [list(r.values()) for r in results]
        )
        trainer_sources = list(
            (
                await self._trainer.register_model_state_dict_source.call(
                    config, generator_rank_layouts=generator_rank_layouts
                )
            ).values()
        )
        await asyncio.gather(
            *(
                generator.all_ranks.connect_model_state_dict_transport.call(
                    config,
                    trainer_sources=trainer_sources,
                    generator_index=generator_index,
                )
                for generator_index, generator in enumerate(self._generators)
            )
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
