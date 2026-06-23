# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with VLLMGenerator (vLLM) and PolicyTrainer (TorchTitan)
   running on separate GPU meshes
2. Weight synchronization across meshes via TorchStore: the trainer publishes its
   model state dict and the generator pulls it into its own parallelism layout,
   with direct GPU-to-GPU RDMA transfer when available
3. Envs driven rollouts; reward and advantage computation live inline
   in the controller.

Command to run:
python3 -m torchtitan.experiments.rl.train \
    --module alphabet_sort --config rl_grpo_qwen3_0_6b_varlen \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

# must run before torch import. Set it as early as possible to avoid other
# imports transitively importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import monarch
from monarch.actor import HostMesh, ProcMesh, this_host
from monarch.job import SlurmJob

from torchtitan.config import ConfigManager, ParallelismConfig
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.trainer import RLTrainer
from torchtitan.observability import structured_logger as sl


logger = logging.getLogger(__name__)


# Opt-in escape hatch when ibverbs RDMA is detected but unstable on the node
# (e.g. RDMABuffer create fails despite is_ibverbs_available() being True).
# RL_RDMA_DISABLE_IBVERBS=1 forces Monarch to use its TCP fallback transport.
# The TCP fallback's per-chunk default timeout (3s) is also too short for full
# state-dict transfers, so we also monkey-patch read_into/write_from to pass a
# much larger timeout. torchstore calls these without a timeout kwarg.
#
# Must be installed in every process that calls RDMABuffer.read_into /
# write_from -- not just the controller. The actual RDMA ops run in the
# worker subprocesses spawned by SlurmJob, which boot from
# `run_worker_loop_forever` and never import this module's top-level code.
# So _bootstrap calls this helper again per-subprocess.
def _install_rdma_tcp_fallback() -> None:
    if os.environ.get("RL_RDMA_DISABLE_IBVERBS") != "1":
        return

    monarch.configure(rdma_disable_ibverbs=True)

    from monarch._src.rdma.rdma import RDMABuffer

    if getattr(RDMABuffer, "_titan_rl_tcp_timeout_patched", False):
        return

    tcp_timeout_s = int(os.environ.get("RL_RDMA_TCP_TIMEOUT_S", "300"))
    orig_read_into = RDMABuffer.read_into
    orig_write_from = RDMABuffer.write_from

    def read_into_with_timeout(self, dst, *, timeout=tcp_timeout_s):
        return orig_read_into(self, dst, timeout=timeout)

    def write_from_with_timeout(self, src, *, timeout=tcp_timeout_s):
        return orig_write_from(self, src, timeout=timeout)

    RDMABuffer.read_into = read_into_with_timeout
    RDMABuffer.write_from = write_from_with_timeout
    RDMABuffer._titan_rl_tcp_timeout_patched = True


_install_rdma_tcp_fallback()


class PerHostProvisioner:
    """Allocates non-overlapping GPU ranges within a single host.

    On the same host, the trainer and generator run on separate GPU
    meshes (e.g. GPUs 0-3 for training, GPUs 4-7 for generation). Each
    call to `allocate(n)` reserves the next *n* GPUs and returns a
    bootstrap callable that sets `CUDA_VISIBLE_DEVICES` before CUDA
    initializes in the spawned process, ensuring each mesh only sees its
    own devices.
    """

    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.next_gpu = 0

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> Callable[[], None]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            # TODO: Remove once Monarch/PyTorch fixes concurrent import during unpickling.
            import torch  # noqa: F401

            # RDMA TCP fallback patch must be installed in every worker subprocess.
            # Workers boot from `run_worker_loop_forever` and don't import this
            # module's top-level code, so do it here before any actor RPC runs.
            _install_rdma_tcp_fallback()

        return _bootstrap


@dataclass
class HostMeshes:
    trainer: HostMesh
    generators: list[HostMesh]
    gpus_per_node: int


def _maybe_launch_slurm_job(
    trainer_world_size: int,
    per_generator_world_size: int,
    num_generators: int,
) -> HostMeshes | None:
    """Launch a Monarch SlurmJob and return HostMeshes when requested via env.

    Activated by ``RL_LAUNCHER=slurm``. Submits one sbatch covering the trainer
    and every generator on disjoint nodes: the trainer gets
    ``trainer_world_size`` GPUs and each of ``num_generators`` generators gets
    ``per_generator_world_size`` GPUs, every role occupying whole nodes (its
    world size must be divisible by ``RL_SLURM_GPUS_PER_NODE``). For single-node
    or colocated runs, omit ``RL_LAUNCHER`` and use the in-process
    ``this_host()`` path instead.

    Env vars:
      RL_SLURM_PARTITION    (required)
      RL_SLURM_GPUS_PER_NODE (optional, default 8)
      RL_SLURM_TIME         (optional, HH:MM:SS)
      RL_SLURM_QOS          (optional, e.g. h100_dev)
      RL_SLURM_ACCOUNT      (optional, e.g. ram)
      RL_SLURM_CPUS_PER_TASK (optional; partition default if unset, often too small)
      RL_SLURM_MEM          (optional; partition default if unset, often too small)

    Returns None when ``RL_LAUNCHER`` is unset or not ``slurm``.
    """
    launcher = os.environ.get("RL_LAUNCHER", "").lower()
    if launcher != "slurm":
        return None

    partition = os.environ.get("RL_SLURM_PARTITION")
    if not partition:
        raise ValueError("RL_LAUNCHER=slurm requires RL_SLURM_PARTITION to be set")
    gpus_per_node = int(os.environ.get("RL_SLURM_GPUS_PER_NODE", "8"))
    time_limit = os.environ.get("RL_SLURM_TIME")  # HH:MM:SS, optional
    qos = os.environ.get("RL_SLURM_QOS")
    account = os.environ.get("RL_SLURM_ACCOUNT")
    cpus_per_task_env = os.environ.get("RL_SLURM_CPUS_PER_TASK")
    cpus_per_task = int(cpus_per_task_env) if cpus_per_task_env else None
    mem = os.environ.get("RL_SLURM_MEM")

    if trainer_world_size % gpus_per_node != 0:
        raise ValueError(
            f"trainer_world_size ({trainer_world_size}) must be divisible "
            f"by RL_SLURM_GPUS_PER_NODE ({gpus_per_node})"
        )
    if per_generator_world_size % gpus_per_node != 0:
        raise ValueError(
            f"per_generator_world_size ({per_generator_world_size}) must be "
            f"divisible by RL_SLURM_GPUS_PER_NODE ({gpus_per_node})"
        )
    trainer_nodes = trainer_world_size // gpus_per_node
    generator_nodes = per_generator_world_size // gpus_per_node
    # Disjoint nodes per role: the trainer plus one host mesh per generator.
    meshes = {"trainer": trainer_nodes}
    for i in range(num_generators):
        meshes[f"generator_{i}"] = generator_nodes

    # SlurmJob takes partition/time/gpus_per_node directly; anything else
    # (qos, account, ...) goes through slurm_args, which is templated as
    # `#SBATCH <arg>` lines into the generated sbatch script.
    extra_slurm_args: list[str] = []
    if qos:
        extra_slurm_args.append(f"--qos={qos}")
    if account:
        extra_slurm_args.append(f"--account={account}")
    logger.info(
        "Launching SlurmJob: trainer=%d node(s), %d generator(s) x %d node(s), "
        "gpus_per_node=%d, partition=%s, qos=%s, account=%s, "
        "cpus_per_task=%s, mem=%s",
        trainer_nodes,
        num_generators,
        generator_nodes,
        gpus_per_node,
        partition,
        qos or "(default)",
        account or "(default)",
        cpus_per_task if cpus_per_task is not None else "(default)",
        mem or "(default)",
    )

    job = SlurmJob(
        meshes=meshes,
        gpus_per_node=gpus_per_node,
        partition=partition,
        time_limit=time_limit,
        cpus_per_task=cpus_per_task,
        mem=mem,
        job_name="torchtitan_rl",
        slurm_args=extra_slurm_args,
        # The worker srun must use the same Python the controller is
        # running.
        python_exe=sys.executable,
        # Don't let SlurmJob fall through to its share_node() codepath
        # (triggered when exclusive=False AND partition is set), which
        # would query clusterscope for cpus_per_task / mem and quietly
        # overwrite the values we just set.
        exclusive=True,
    )
    if os.environ.get("MONARCH_BATCH_JOB") == "1":
        # In-allocation batch client: the controller runs inside the allocation
        # the submitter already created. Reconnect to the workers the runner
        # started (via the cached BatchJob) instead of submitting a new job.
        state = job.state()
    else:
        job.apply()
        state = job.state()
    return HostMeshes(
        trainer=state.trainer,
        generators=[
            getattr(state, f"generator_{i}") for i in range(num_generators)
        ],
        gpus_per_node=gpus_per_node,
    )


def _compute_trainer_world_size(p: ParallelismConfig) -> int:
    """Compute world size from all parallel dimensions."""
    dp_shard = max(p.data_parallel_shard_degree, 1)
    return (
        p.data_parallel_replicate_degree
        * dp_shard
        * p.tensor_parallel_degree
        * p.pipeline_parallel_degree
        * p.context_parallel_degree
    )


def _compute_generator_world_size(p: InferenceParallelismConfig) -> int:
    """Number of GPU processes for one generator (vLLM) instance."""
    return p.data_parallel_degree * p.tensor_parallel_degree


def _spawn_proc_mesh(
    host_mesh: HostMesh,
    role_world_size: int,
    gpus_per_node: int,
    *,
    role: str,
) -> ProcMesh:
    """Spawn one role's proc mesh on ``host_mesh``, splitting ``role_world_size``
    evenly across the mesh's hosts.
    """
    nodes = len(host_mesh)
    assert role_world_size % nodes == 0, (
        f"{role} world size ({role_world_size}) must be evenly divisible by its "
        f"host count ({nodes})"
    )
    role_gpus_per_node = role_world_size // nodes
    provisioner = PerHostProvisioner(total_gpus=gpus_per_node)
    return host_mesh.spawn_procs(
        per_host={"gpus": role_gpus_per_node},
        bootstrap=provisioner.allocate(role_gpus_per_node),
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
            provided, each role is spawned on its provided host mesh. None means
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

        trainer_mesh = _spawn_proc_mesh(
            trainer_host_mesh, trainer_world_size, gpus_per_node, role="trainer"
        )
        generator_meshes = [
            _spawn_proc_mesh(
                gen_host_mesh,
                per_generator_world_size,
                gpus_per_node,
                role="generator",
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
            bootstrap=provisioner.allocate(trainer_world_size),
        )
        generator_meshes = [
            host_mesh.spawn_procs(
                per_host={"gpus": per_generator_world_size},
                bootstrap=provisioner.allocate(per_generator_world_size),
            )
            for _ in range(num_generators)
        ]

    return trainer_mesh, generator_meshes


async def main():
    config = ConfigManager().parse_args()
    assert isinstance(config, RLTrainer.Config)
    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        enable=config.trainer.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    rl_trainer: RLTrainer = config.build()
    try:
        trainer_world_size = _compute_trainer_world_size(config.trainer.parallelism)
        per_generator_world_size = _compute_generator_world_size(
            config.generator.parallelism
        )
        host_meshes = _maybe_launch_slurm_job(
            trainer_world_size, per_generator_world_size, config.num_generators
        )
        trainer_mesh, generator_meshes = spawn_proc_mesh(
            trainer_world_size,
            per_generator_world_size,
            host_meshes=host_meshes,
            num_generators=config.num_generators,
        )
        await rl_trainer.setup_async(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
        )
        await rl_trainer.train()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await rl_trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
