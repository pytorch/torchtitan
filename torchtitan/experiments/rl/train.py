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
import datetime
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

# must run before torch import. Set it as early as possible to avoid other
# imports transitively importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from monarch.actor import HostMesh, ProcMesh, this_host

from torchtitan.config import ConfigManager, ParallelismConfig
from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.observability import structured_logger as sl


logger = logging.getLogger(__name__)


def breakable_cudagraph_env(generator_cfg) -> dict[str, str]:
    """Per-proc launch env a FULL_AND_PIECEWISE generator needs: ``VLLM_USE_BREAKABLE_CUDAGRAPH``.

    rl/models/attention.py's ``@eager_break_during_capture`` reads this env at MODULE IMPORT and
    makes prefill attention a cudagraph break (run eager at replay). The import happens before the
    generator actor's ``__init__`` and in the vLLM EngineCore worker subprocesses -- which do NOT
    inherit a runtime-set ``os.environ`` -- so setting it at runtime is too late. It must go in the
    proc's LAUNCH env (the spawn bootstrap, or the MAST role.env). Without it the decorator no-ops
    and prefill attention is captured as ``output.fill_(0)`` (zeroed) -> the model never reads the
    prompt -> coherent-but-unrelated output. FULL_DECODE_ONLY never captures prefill so it needs
    nothing. Shared so the OSS spawn path and the fbcode MAST launcher use one source of truth.
    """
    cg = getattr(generator_cfg, "cudagraph", None)
    if (
        cg is not None
        and getattr(cg, "enable", False)
        and getattr(cg, "mode", "") == "FULL_AND_PIECEWISE"
    ):
        return {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"}
    return {}


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

    def allocate(
        self, num_gpus: int, *, extra_env: dict[str, str] | None = None
    ) -> Callable[[], None]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            # Per-proc LAUNCH env. Set here (in the bootstrap) -- not at runtime in the actor's
            # __init__ -- because rl/models/attention.py's @eager_break_during_capture reads
            # VLLM_USE_BREAKABLE_CUDAGRAPH at MODULE IMPORT, which happens before __init__ and
            # in the vLLM EngineCore worker subprocs (which do not inherit a runtime-set env).
            if extra_env:
                os.environ.update(extra_env)
            # TODO: Remove once Monarch/PyTorch fixes concurrent import during unpickling.
            import torch  # noqa: F401

        return _bootstrap


@dataclass
class HostMeshes:
    trainer: HostMesh
    generators: list[HostMesh]
    gpus_per_node: int


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
    extra_env: dict[str, str] | None = None,
) -> ProcMesh:
    """Spawn one role's proc mesh on ``host_mesh``, splitting ``role_world_size``
    evenly across the mesh's hosts. ``extra_env`` is applied in each proc's bootstrap.
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
        bootstrap=provisioner.allocate(role_gpus_per_node, extra_env=extra_env),
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
            bootstrap=provisioner.allocate(trainer_world_size),
        )
        generator_meshes = [
            host_mesh.spawn_procs(
                per_host={"gpus": per_generator_world_size},
                bootstrap=provisioner.allocate(
                    per_generator_world_size, extra_env=generator_env
                ),
            )
            for _ in range(num_generators)
        ]

    return trainer_mesh, generator_meshes


async def main():
    # Monarch is making breaking changes to its message dispatching mechanism.
    # The recommended way to maintain the current behavior, which is what we want,
    # is to use @concurrent_endpoint. But that decorator is not available in
    # monarch's stable release yet. So we pin this env var for now, until the
    # new release is cut. More details can be found in:
    # https://github.com/meta-pytorch/monarch/pull/4243
    # https://github.com/meta-pytorch/monarch/pull/4211
    os.environ["MONARCH_ACTOR_QUEUE_DISPATCH"] = "0"

    config = ConfigManager().parse_args()
    assert isinstance(config, Controller.Config)

    # Give each local run its own dump_folder so rollouts / metrics / structured
    # logs from different runs (and different configs) don't pile into one shared
    # directory (the recorder appends, so a shared dir mixes runs). Append
    # "<config>-<timestamp>" to whatever dump_folder is configured.
    _cfg_name = "run"
    for _i, _a in enumerate(sys.argv):
        if _a == "--config" and _i + 1 < len(sys.argv):
            _cfg_name = sys.argv[_i + 1]
        elif _a.startswith("--config="):
            _cfg_name = _a.split("=", 1)[1]
    _run_id = f"{_cfg_name}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    config.dump_folder = os.path.join(config.dump_folder, _run_id)
    print(f"[train] this run's dump_folder: {config.dump_folder}")

    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        enable=config.trainer.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    rl_trainer: Controller = config.build()
    try:
        trainer_world_size = _compute_trainer_world_size(config.trainer.parallelism)
        per_generator_world_size = _compute_generator_world_size(
            config.generator.parallelism
        )
        trainer_mesh, generator_meshes = spawn_proc_mesh(
            trainer_world_size,
            per_generator_world_size,
            host_meshes=None,
            num_generators=config.num_generators,
            generator_env=breakable_cudagraph_env(config.generator),
        )
        await rl_trainer.setup_async(
            trainer_mesh=trainer_mesh,
            generator_meshes=generator_meshes,
        )
        await rl_trainer.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await rl_trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
