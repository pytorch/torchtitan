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
    --module torchtitan.experiments.rl.examples.alphabet_sort --config rl_grpo_qwen3_0_6b_varlen \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

# must run before torch import. Set it as early as possible to avoid other
# imports transitively importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from monarch.actor import HostMesh, ProcMesh, this_host

from torchtitan.config import ConfigManager, ParallelismConfig
from torchtitan.experiments.rl.trainer import RLTrainer
from torchtitan.observability import structured_logger as sl


logger = logging.getLogger(__name__)


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

        return _bootstrap


@dataclass
class HostMeshes:
    trainer: HostMesh
    generator: HostMesh
    gpus_per_node: int


def _compute_world_size(p: ParallelismConfig) -> int:
    """Compute world size from all parallel dimensions."""
    dp_shard = max(p.data_parallel_shard_degree, 1)
    return (
        p.data_parallel_replicate_degree
        * dp_shard
        * p.tensor_parallel_degree
        * p.pipeline_parallel_degree
        * p.context_parallel_degree
    )


def spawn_proc_mesh(
    trainer_world_size: int,
    generator_world_size: int,
    host_meshes: HostMeshes | None = None,
    *,
    num_generators: int = 1,
) -> tuple[ProcMesh, list[ProcMesh]]:
    """Spawn the trainer and generator proc meshes.

    Args:
        trainer_world_size: Number of GPU procs to spawn for the trainer.
        generator_world_size: Number of GPU procs to spawn for each generator.
        host_meshes: Caller-provided trainer/generator host meshes. When
            provided, each role is spawned on its provided host mesh. None means
            both roles are spawned on ``this_host()`` by using non-overlapping
            GPU ranges.
        num_generators: Number of generator proc meshes to spawn.

    Returns:
        The ``(trainer_mesh, generator_meshes)`` proc meshes.
    """
    total_generator_gpus = num_generators * generator_world_size
    total_gpus = trainer_world_size + total_generator_gpus
    logger.info(
        f"{num_generators} generator(s) * {generator_world_size} GPUs + "
        f"{trainer_world_size} trainer GPUs = {total_gpus} total"
    )

    if host_meshes is not None:
        if num_generators > 1:
            raise NotImplementedError(
                "multi-host multi-generator topology is not yet supported"
            )

        trainer_host_mesh = host_meshes.trainer
        generator_host_mesh = host_meshes.generator
        gpus_per_node = host_meshes.gpus_per_node

        trainer_nodes = trainer_host_mesh.sizes["hosts"]
        generator_nodes = generator_host_mesh.sizes["hosts"]
        # Validate that world sizes are evenly divisible by node counts
        assert trainer_world_size % trainer_nodes == 0, (
            f"trainer_world_size ({trainer_world_size}) must be "
            f"evenly divisible by trainer_nodes ({trainer_nodes})"
        )
        assert generator_world_size % generator_nodes == 0, (
            f"generator_world_size ({generator_world_size}) must be "
            f"evenly divisible by generator_nodes ({generator_nodes})"
        )

        # Compute GPUs per node for each role based on the world size and
        # number of nodes allocated to that role
        trainer_gpus_per_node = trainer_world_size // trainer_nodes
        generator_gpus_per_node = generator_world_size // generator_nodes

        # Use PerHostProvisioner to set CUDA_VISIBLE_DEVICES so each role
        # only sees its own GPUs and doesn't conflict with other
        # processes on the node
        trainer_provisioner = PerHostProvisioner(total_gpus=gpus_per_node)
        generator_provisioner = PerHostProvisioner(total_gpus=gpus_per_node)

        trainer_mesh = trainer_host_mesh.spawn_procs(
            per_host={"gpus": trainer_gpus_per_node},
            bootstrap=trainer_provisioner.allocate(trainer_gpus_per_node),
        )
        generator_mesh = generator_host_mesh.spawn_procs(
            per_host={"gpus": generator_gpus_per_node},
            bootstrap=generator_provisioner.allocate(generator_gpus_per_node),
        )
        generator_meshes = [generator_mesh]
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
                per_host={"gpus": generator_world_size},
                bootstrap=provisioner.allocate(generator_world_size),
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
        trainer_world_size = _compute_world_size(config.trainer.parallelism)
        generator_world_size = _compute_world_size(config.generator.parallelism)
        trainer_mesh, generator_meshes = spawn_proc_mesh(
            trainer_world_size,
            generator_world_size,
            host_meshes=None,
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
