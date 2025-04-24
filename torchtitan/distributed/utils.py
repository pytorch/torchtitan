# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import math
import os
from collections.abc import Generator, Iterable
from datetime import timedelta

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module, device_type


def _dist_reduce(
    x: torch.Tensor,
    reduceOp: str,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    """Perform distributed reduction on a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        reduceOp (str): Reduce operation to perform.
        mesh (DeviceMesh): Device mesh to use for reduction.
        extra_pg (dist.ProcessGroup, optional): Extra process group to use for reduction.
            Defaults to None. If provided, this all_reduce will be called for the extra
            process group, and then the result will be all_reduced for the mesh.
    """
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()

    if extra_pg is not None:
        x = funcol.all_reduce(x, reduceOp=reduceOp, group=extra_pg)

    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()


def dist_max(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh, extra_pg=extra_pg
    )


def dist_mean(
    x: torch.Tensor,
    mesh: DeviceMesh,
    extra_pg: dist.ProcessGroup | None = None,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh, extra_pg=extra_pg
    )


def set_determinism(
    world_mesh: DeviceMesh | None,
    device: torch.device,
    seed: int | None = None,
    deterministic: bool = False,
    distinct_seed_mesh_dim: str = "pp",
) -> None:
    """
    Set the same DTensor manual seed for all dimensions in world mesh, but only different seeds
    across dimension denoted by `distinct_seed_mesh_dim`. An example use case is pipeline parallelism,
    where we want to have the same seed across SPMD groups, but different seeds across PP groups.

    Currently, does not set seeds for the CUDA RNG since TorchTitan always uses DTensor for SPMD parallelisms,
    and DTensor manages its own RNG tracker, but we could extend to support both if needed.

    Set Determinism flags for increased reproducibility with loss of performance.
    """
    if deterministic:
        logger.info("Deterministic algorithm enabled (expect perf degradation).")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # env var for deterministic CuBLAS
        # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    if not world_mesh:
        if seed is not None:
            torch.manual_seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
            logger.debug(f"Single-process job using seed: {seed}")
        return

    # to ensure we can control which ranks have same or different seeds, all ranks agree on a starting seed.
    # if user provides one, we use this. Otherwise rank 0 rolls the dice and everyone else uses that.
    if seed is None:
        # Extract the seed for torch's main generator on rank 0 and standardizes on using that to build
        # seeds for unique SPMD groups
        seed_tensor = torch.get_rng_state()[:8].to(device)
        torch.distributed.broadcast(seed_tensor, src=0)
        seed = seed_tensor.to("cpu").view(torch.uint64).item()

    # Set distinct seed for each rank in mesh dimensions, with dimension name provdied by `distinct_seed_mesh_dim`
    # For PP + SPMD cases, we want to separate the world into the SPMD mesh and the PP mesh,
    # and choose a unique seed for each rank on the PP mesh.
    # TODO(jianiw): We could further extend this to support mutiple distinct dimensions instead of just one.
    if (
        c10d.get_world_size() > 1
        and distinct_seed_mesh_dim in world_mesh.mesh_dim_names
    ):
        distinct_mesh = world_mesh[distinct_seed_mesh_dim]
        seed += distinct_mesh.get_local_rank()
        seed %= 2**64

        logger.debug(
            f"{distinct_seed_mesh_dim} rank {distinct_mesh.get_local_rank()}, Global rank {c10d.get_rank()} using seed: {seed}"
        )
        duplicate_seed_mesh = list(
            filter(
                lambda name: name != distinct_seed_mesh_dim, world_mesh.mesh_dim_names
            )
        )
        duplicate_seed_mesh = (
            world_mesh[duplicate_seed_mesh] if len(duplicate_seed_mesh) else None
        )
    else:
        duplicate_seed_mesh = world_mesh
        logger.debug(f"Global Rank {c10d.get_rank()} using seed: {seed}")

    # The native RNGs and python RNG may not be important, except for the 1-D PP case, but we seed them for consistency.
    torch.manual_seed(seed)
    # PYTHONHASHSEED can be a decimal number in the range [0, 2**32 - 1]
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)

    # As long as we are not in the 1-D (PP-only) case, we will have a seed to use for all ranks of the SPMD mesh.
    # IF PP is also used, this seed is unique per PP rank.
    if duplicate_seed_mesh and duplicate_seed_mesh.get_coordinate() is not None:
        torch.distributed.tensor._random.manual_seed(seed, duplicate_seed_mesh)


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str,
):
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError:
        print(
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


def get_train_context(
    enable_loss_parallel: bool, enable_compiled_autograd: bool
) -> Generator[None, None, None]:
    @contextlib.contextmanager
    def context(cp_context: Generator[None, None, None] | None = None):
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(
                    torch._dynamo.utils.maybe_enable_compiled_autograd(True)
                )

            if cp_context is not None:
                from torch.nn.attention import sdpa_kernel, SDPBackend

                stack.enter_context(
                    sdpa_kernel(
                        [
                            SDPBackend.FLASH_ATTENTION,
                            SDPBackend.EFFICIENT_ATTENTION,
                            SDPBackend.CUDNN_ATTENTION,
                        ]
                    )
                )
                stack.enter_context(cp_context)

            yield

    return context


def init_distributed(job_config):
    def _warn_overwrite_env(env, val):
        if env in os.environ:
            logger.warning(
                f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
            )
        os.environ[env] = val

    def _get_distributed_backend(job_config):
        backend = "nccl"
        if device_type in torch.distributed.Backend.default_device_backend_map:
            backend = torch.distributed.Backend.default_device_backend_map.get(
                device_type
            )
        if job_config.training.enable_cpu_offload:
            backend = f"{device_type}:{backend},cpu:gloo"
        return backend

    TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
    TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
    DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
    ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
    SKIP_CLEANUP = "3"

    # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
    # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
    # This could be done only when flight recorder is enabled, but its nice to be consistent to avoid subtle
    # behavior differences
    _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

    # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
    _warn_overwrite_env(TRACE_BUFFER_SIZE, str(job_config.comm.trace_buf_size))
    if job_config.comm.trace_buf_size > 0:
        # dump on timeout by default if trace buffer is enabled
        _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
        dump_dir = f"{job_config.job.dump_folder}/comm_trace"
        os.makedirs(dump_dir, exist_ok=True)
        _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/rank_")

    torch.distributed.init_process_group(
        backend=_get_distributed_backend(job_config),
        timeout=timedelta(seconds=job_config.comm.init_timeout_seconds),
    )


def set_pg_timeouts(timeout, world_mesh):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, because
    otherwise you may face a race where the slow rank has not reached the timeout reduction point
    yet due to slow operations permitted under the old timeout value, but other faster ranks may
    start issuing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(
        f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}"
    )
    # Ensure that all the ranks have reached the point of setting the new timeout-
    # otherwise, some ranks may issue collectives with the new/shorter timeout and
    # those may time out, before other ranks have finished with initialization done
    # under the old/slow timeout.
    torch.distributed.barrier(device_ids=[device_module.current_device()])
    device_module.synchronize()

    groups = [world_mesh.get_group(mesh_dim) for mesh_dim in range(world_mesh.ndim)]

    # None represents the 'default' PG, not part of the mesh
    groups.append(None)
    for group in groups:
        torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: pipeline parallel device mesh. If not None, will reduce gradient norm across PP stages.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(
        grads, norm_type, error_if_nonfinite, foreach
    )

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.

        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm
