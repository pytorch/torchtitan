# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Monarch proc meshes for the RL roles: GPU meshes for the trainer and generators,
CPU meshes on the controller host, the rollout workers' hosts, and the Monarch client's
shutdown."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from monarch.actor import (
    default_bootstrap_cmd,
    HostMesh,
    ProcMesh,
    shutdown_context,
    this_host,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.rl._env import apply_env_defaults
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.tools.utils import get_visible_devices_env_var

if TYPE_CHECKING:
    from torchtitan.rl.rollout.rollouter import RolloutWorkerRoles

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
    env (the visible-devices mask) for those GPUs. The
    returned env vars are applied to the spawned process via ``bootstrap_command``.
    The mask's variable comes from this (the controller's) device type, so the
    controller runs on a node with the same accelerator as the workers, as the
    Monarch launchers do.
    """

    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.next_gpu = 0
        self.env_var = get_visible_devices_env_var()

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> dict[str, str]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        return {self.env_var: ",".join(str(g) for g in gpu_ids)}


@dataclass
class HostMeshes:
    trainer: HostMesh
    generators: list[HostMesh]
    gpus_per_node: int
    controller_on_first_trainer_host: bool
    """True when the controller runs on the trainer's first host: that host also has the
    controller's role. False: the controller runs outside the job's hosts."""


def generator_role_names(num_generators: int) -> list[str]:
    """Generator mesh and actor names: ``generator``, or ``generator_<i>`` for several."""
    if num_generators == 1:
        return ["generator"]
    return [f"generator_{i}" for i in range(num_generators)]


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
) -> ProcMesh:
    """Spawn one role's proc mesh on ``host_mesh``, splitting ``role_world_size``
    evenly across the mesh's hosts.

    Assumes ``_check_gpus_per_host`` passed: the split is even and the GPUs fit.
    """
    nodes = len(host_mesh)
    role_gpus_per_node = role_world_size // nodes
    env = provisioner.allocate(role_gpus_per_node)
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
        )
        generator_meshes = [
            _spawn_proc_mesh(
                gen_host_mesh,
                per_generator_world_size,
                provisioners[id(gen_host_mesh)],
                bootstrap=_bootstrap_generator,
            )
            for gen_host_mesh in generator_host_meshes
        ]
    else:
        # Single-node mode: partition GPUs on this_host() via
        # the visible-devices mask
        host_mesh = this_host()
        provisioner = PerHostProvisioner(total_gpus=total_gpus)
        trainer_mesh = host_mesh.spawn_procs(
            per_host={"gpus": trainer_world_size},
            bootstrap=_preimport_torch,
            bootstrap_command=default_bootstrap_cmd().with_env(
                provisioner.allocate(trainer_world_size)
            ),
        )
        generator_meshes = [
            host_mesh.spawn_procs(
                per_host={"gpus": per_generator_world_size},
                bootstrap=_bootstrap_generator,
                bootstrap_command=default_bootstrap_cmd().with_env(
                    provisioner.allocate(per_generator_world_size)
                ),
            )
            for _ in range(num_generators)
        ]

    return trainer_mesh, generator_meshes


def spawn_cpu_procs(num_procs: int) -> ProcMesh:
    """Spawn ``num_procs`` CPU procs on the controller host."""
    return this_host().spawn_procs(per_host={"cpus": num_procs})


def rollout_worker_hosts(
    *, host_meshes: HostMeshes | None, worker_roles: "RolloutWorkerRoles"
) -> list[HostMesh]:
    """Return the hosts that `worker_roles` picks, one single-host mesh each, in job order:
    trainer hosts, generator hosts, then the controller's machine when it is outside the job.

    Each host gets its roles, then the rule of `RolloutWorkerRoles` picks it: at least one of its
    roles is True and none is False. Example: trainer hosts t0, t1 and generator hosts g0, g1. The
    controller's host depends on the launcher:

        the controller on t0 (`controller_on_first_trainer_host`):
            t0: trainer + controller | t1: trainer | g0, g1: generator
        the controller outside the job:
            t0, t1: trainer | g0, g1: generator | this_host(): controller

    The controller's host is always returned as `this_host()`, also when it is t0:

        trainer=True, generator=None, controller=None, the controller on t0     -> [this_host(), t1]
        trainer=True, generator=None, controller=False, the controller on t0    -> [t1]
        trainer=True, generator=None, controller=False, the controller outside  -> [t0, t1]
        trainer=None, generator=None, controller=True, the controller on t0     -> [this_host()]
        trainer=None, generator=None, controller=True, the controller outside   -> [this_host()]
        no launcher (`host_meshes` is None), any roles                           -> [this_host()]
    """
    if host_meshes is None:
        return [this_host()]

    # ======== Give each job host its roles ========
    # Keyed by (id(mesh), host index): roles that share a host share its mesh object, and a
    # HostMesh can't be a dict key. `flatten("hosts")` gives each mesh one "hosts" axis to slice:
    # a launcher's mesh already has it, but a `this_host()` mesh has no axis.
    single_hosts: dict[tuple[int, int], HostMesh] = {}
    roles_by_host: dict[tuple[int, int], set[str]] = {}
    role_meshes = [("trainer", host_meshes.trainer)] + [
        ("generator", mesh) for mesh in host_meshes.generators
    ]
    for role, mesh in role_meshes:
        for index in range(len(mesh)):
            key = (id(mesh), index)
            single_hosts.setdefault(key, mesh.flatten("hosts").slice(hosts=index))
            roles_by_host.setdefault(key, set()).add(role)
    if host_meshes.controller_on_first_trainer_host:
        controller_host_key = (id(host_meshes.trainer), 0)
        roles_by_host[controller_host_key].add("controller")
        # Spawn through `this_host()`, the controller's own machine. On Slurm and PBS, the batch
        # script that starts the controller typically runs on trainer host 0, so both are the same
        # machine. If the batch script ran on another node, a service at 127.0.0.1 on the
        # controller's machine stays reachable. The cost: trainer host 0's share of the workers then
        # runs on the controller's machine.
        single_hosts[controller_host_key] = this_host()

    # ======== Pick the hosts ========
    hosts: list[HostMesh] = []
    for key, roles in roles_by_host.items():
        values = {getattr(worker_roles, role) for role in roles}
        if True in values and False not in values:
            hosts.append(single_hosts[key])
    # A controller outside the job: its machine has no other role, so no other role can veto it.
    if not host_meshes.controller_on_first_trainer_host and worker_roles.controller:
        hosts.append(this_host())
    if not hosts:
        raise ValueError(
            f"rollouter.worker_roles ({worker_roles}) picks no host: each host with a role set to "
            "True also has a role set to False. The job's hosts have these roles: "
            f"{[sorted(roles) for roles in roles_by_host.values()]}. Please set each role that "
            "should not veto a host to None."
        )
    return hosts


def shutdown() -> None:
    """Shut down the Monarch client (normally an atexit hook) and wait for it."""
    shutdown_context().get()
