# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch proc meshes for the RL roles: GPU meshes for the trainer and generators,
CPU meshes on the controller host, and the Monarch client's shutdown."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

from monarch.actor import (
    default_bootstrap_cmd,
    HostMesh,
    ProcMesh,
    shutdown_context,
    this_host,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.rl._env import apply_env_defaults, RL_ROLE_ENV
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig

logger = logging.getLogger(__name__)


def _preimport_torch() -> None:
    """``bootstrap`` setup callable: pre-import torch on the spawned proc."""
    # Env defaults must be set before torch initializes CUDA in this proc.
    apply_env_defaults()
    # TODO: Remove once Monarch/PyTorch fixes concurrent import during unpickling.
    import torch  # noqa: F401


def _bootstrap_generator() -> None:
    """``bootstrap`` setup callable for Generator."""
    import os

    # TODO: this can be removed if it is addressed upstream in: https://github.com/tile-ai/tvm/issues/55
    # Generator encounters the following issue:
    # 1. vLLM's jit_monitor transitively imports readline through the following chain:
    #      vLLM -> tilelang  -> tvm  -> tvm's base.py -> import readline
    #      https://github.com/vllm-project/vllm/blob/b23bd73f540175f9e117eaee5029cd7d8df63964/vllm/utils/jit_monitor.py#L451
    #      https://github.com/tile-ai/tvm/blob/28a0d34420d2fa9bc71fc891445a3f1396fca759/python/tvm/base.py#L71
    #    readline does tcsetattr on its import, which touches the controlling terminal.
    # 2. Monarch spawns procs in a background process group.
    #
    # As a result, when spawning the Generator mesh, readline-import touches the
    # controlling terminal in background processes, and kernel sends SIGTTOU, which stops
    # the process.
    #
    # Strictly speaking this should be a tvm bug, since it should not assume the import
    # happens in a foreground process. To workaround it, we do a proactive readline-import
    # warmup here with SIGTTOU being temporarily blocked. Then the subsequent readline-import
    # in tvm will not do tcsetattr again since it is already done here.
    if os.isatty(0):
        import signal

        _prev_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTTOU})
        try:
            import readline  # noqa: F401
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, _prev_mask)

    _preimport_torch()


class PerHostProvisioner:
    """Allocates non-overlapping GPU ranges within a single host.

    On the same host, the trainer and generator run on separate GPU
    meshes (e.g. GPUs 0-3 for training, GPUs 4-7 for generation). Each
    call to `allocate(n)` reserves the next *n* GPUs and returns the launch
    env (`CUDA_VISIBLE_DEVICES` plus any `extra_env`) for those GPUs. The
    returned env vars are applied to the spawned process via ``bootstrap_command``.
    """

    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.next_gpu = 0

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(
        self, num_gpus: int, *, extra_env: dict[str, str] | None = None
    ) -> dict[str, str]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        env = {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)}
        if extra_env:
            env.update(extra_env)
        return env


@dataclass
class HostMeshes:
    trainer: HostMesh
    generators: list[HostMesh]
    gpus_per_node: int


def compute_trainer_world_size(p: ParallelismConfig) -> int:
    """Compute world size from all parallel dimensions."""
    dp_shard = max(p.data_parallel_shard_degree, 1)
    return (
        p.data_parallel_replicate_degree
        * dp_shard
        * p.tensor_parallel_degree
        * p.pipeline_parallel_degree
        * p.context_parallel_degree
    )


def compute_generator_world_size(p: InferenceParallelismConfig) -> int:
    """Number of GPU processes for one generator (vLLM) instance."""
    return p.data_parallel_degree * p.tensor_parallel_degree


def _spawn_proc_mesh(
    host_mesh: HostMesh,
    role_world_size: int,
    provisioner: PerHostProvisioner,
    *,
    bootstrap: Callable[[], None],
    extra_env: dict[str, str] | None = None,
) -> ProcMesh:
    """Spawn one role's proc mesh on ``host_mesh``, splitting ``role_world_size``
    evenly across the mesh's hosts. ``extra_env`` is applied in each proc's bootstrap.

    Assumes ``_check_gpus_per_host`` passed: the split is even and the GPUs fit.
    """
    nodes = len(host_mesh)
    role_gpus_per_node = role_world_size // nodes
    env = provisioner.allocate(role_gpus_per_node, extra_env=extra_env)
    return host_mesh.spawn_procs(
        per_host={"gpus": role_gpus_per_node},
        bootstrap=bootstrap,
        bootstrap_command=default_bootstrap_cmd().with_env(env),
    )


def _check_gpus_per_host(
    host_meshes: HostMeshes, trainer_world_size: int, per_generator_world_size: int
) -> None:
    """Fail before spawning anything if the requested GPUs don't fit on the hosts.

    A role's GPUs are split equally across the hosts of its mesh:

        trainer, 4 GPUs, 2-host mesh -> 2 GPUs on each host           OK
        trainer, 3 GPUs, 2-host mesh -> 1.5 GPUs on each host         fails

    Roles given the same mesh object share its GPUs, trainer first:

        meshes A and B: one host with 4 GPUs each
        trainer=A (2 GPUs), generators=[A, A, B, B, B, B] (1 GPU each)
            A: trainer 0,1 | gen0 2 | gen1 3
            B: gen2 0 | gen3 1 | gen4 2 | gen5 3          OK
        generators=[A, A, A, B, B, B]
            A: trainer 0,1 | gen0 2 | gen1 3 | gen2 ?     5 > 4 GPUs: fails

    It has to be the same object: two `this_host()` calls make two meshes, and each
    starts at GPU 0.
    """
    trainer_roles = [("trainer", host_meshes.trainer, trainer_world_size)]
    generator_roles = [
        (f"generator{i}", host_mesh, per_generator_world_size)
        for i, host_mesh in enumerate(host_meshes.generators)
    ]
    roles = trainer_roles + generator_roles
    # For each mesh object (by id), the GPUs per host each of its roles needs.
    role_gpus_by_mesh: dict[int, dict[str, int]] = {}
    for role, host_mesh, role_world_size in roles:
        nodes = len(host_mesh)
        if role_world_size % nodes != 0:
            raise ValueError(
                f"{role} world size ({role_world_size}) must be evenly divisible by "
                f"its host count ({nodes})"
            )
        role_gpus_by_mesh.setdefault(id(host_mesh), {})[role] = role_world_size // nodes

    # Check that the roles on each mesh fit in its GPUs per host.
    for role_gpus in role_gpus_by_mesh.values():
        total_gpus = sum(role_gpus.values())
        if total_gpus > host_meshes.gpus_per_node:
            raise ValueError(
                f"Roles on one host mesh need {total_gpus} GPUs per host, but "
                f"gpus_per_node is {host_meshes.gpus_per_node}: {role_gpus}"
            )


def spawn_proc_mesh(
    trainer_world_size: int,
    per_generator_world_size: int,
    host_meshes: HostMeshes | None = None,
    *,
    num_generators: int = 1,
    generator_env: dict[str, str] | None = None,
) -> tuple[ProcMesh, list[ProcMesh]]:
    """Spawn the trainer and generator proc meshes.

    Args:
        trainer_world_size: Number of GPU procs to spawn for the trainer.
        per_generator_world_size: Number of GPU procs to spawn for each generator.
        host_meshes: Caller-provided trainer/generator host meshes. When
            provided, each role is spawned on its provided host mesh; roles given
            the same host mesh object get non-overlapping GPU ranges on it. None means
            both roles are spawned on ``this_host()`` by using non-overlapping
            GPU ranges.
        num_generators: Number of generator proc meshes to spawn.

    Returns:
        The ``(trainer_mesh, generator_meshes)`` proc meshes.
    """
    total_generator_gpus = num_generators * per_generator_world_size
    total_gpus = trainer_world_size + total_generator_gpus
    # Role tags for host memory metrics; the procs' children inherit them.
    trainer_env = {RL_ROLE_ENV: "trainer"}
    generator_env = {**(generator_env or {}), RL_ROLE_ENV: "generator"}
    logger.info(
        f"{num_generators} generator(s) * {per_generator_world_size} GPUs + "
        f"{trainer_world_size} trainer GPUs = {total_gpus} total"
    )

    if host_meshes is not None:
        trainer_host_mesh = host_meshes.trainer
        generator_host_meshes = host_meshes.generators
        gpus_per_node = host_meshes.gpus_per_node

        assert len(generator_host_meshes) == num_generators, (
            f"expected {num_generators} generator host mesh(es), "
            f"got {len(generator_host_meshes)}"
        )

        _check_gpus_per_host(host_meshes, trainer_world_size, per_generator_world_size)
        # Roles on the same mesh share one provisioner, which hands out the next free GPUs:
        #   mesh A: trainer -> 0,1 | gen0 -> 2 | gen1 -> 3
        # Keyed by id(mesh) because a HostMesh can't be a dict key.
        provisioners: dict[int, PerHostProvisioner] = {}
        for host_mesh in [trainer_host_mesh, *generator_host_meshes]:
            provisioners.setdefault(
                id(host_mesh), PerHostProvisioner(total_gpus=gpus_per_node)
            )
        trainer_mesh = _spawn_proc_mesh(
            trainer_host_mesh,
            trainer_world_size,
            provisioners[id(trainer_host_mesh)],
            bootstrap=_preimport_torch,
            extra_env=trainer_env,
        )
        generator_meshes = [
            _spawn_proc_mesh(
                gen_host_mesh,
                per_generator_world_size,
                provisioners[id(gen_host_mesh)],
                bootstrap=_bootstrap_generator,
                extra_env=generator_env,
            )
            for gen_host_mesh in generator_host_meshes
        ]
    else:
        # Single-node mode: partition GPUs on this_host() via
        # CUDA_VISIBLE_DEVICES
        host_mesh = this_host()
        provisioner = PerHostProvisioner(total_gpus=total_gpus)
        trainer_mesh = host_mesh.spawn_procs(
            per_host={"gpus": trainer_world_size},
            bootstrap=_preimport_torch,
            bootstrap_command=default_bootstrap_cmd().with_env(
                provisioner.allocate(trainer_world_size, extra_env=trainer_env)
            ),
        )
        generator_meshes = [
            host_mesh.spawn_procs(
                per_host={"gpus": per_generator_world_size},
                bootstrap=_bootstrap_generator,
                bootstrap_command=default_bootstrap_cmd().with_env(
                    provisioner.allocate(
                        per_generator_world_size, extra_env=generator_env
                    )
                ),
            )
            for _ in range(num_generators)
        ]

    return trainer_mesh, generator_meshes


def spawn_cpu_procs(num_procs: int, *, role: str) -> ProcMesh:
    """Spawn ``num_procs`` CPU procs on the controller host, tagged ``role`` for host memory metrics."""
    return this_host().spawn_procs(
        per_host={"cpus": num_procs},
        bootstrap_command=default_bootstrap_cmd().with_env({RL_ROLE_ENV: role}),
    )


def shutdown() -> None:
    """Shut down the Monarch client (normally an atexit hook) and wait for it."""
    shutdown_context().get()
