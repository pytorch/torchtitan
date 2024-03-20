# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os

from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property

import torch

from torch.distributed.device_mesh import init_device_mesh
from torchtrain.logging_utils import logger
from torchtrain.parallelisms.parallelize_llama import parallelize_llama

models_parallelize_fns = {
    "llama": parallelize_llama,
}

TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
SKIP_CLEANUP = "3"


def _warn_overwrite_env(env, val):
    if env in os.environ:
        logger.warning(
            f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
        )
    os.environ[env] = val


def set_pg_timeouts(timeout, world_mesh):
    logger.info(
        f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}"
    )
    # Ensure that all the ranks have reached the point of setting the new timeout-
    # otherwise, some ranks may issue collectives with the new/shorter timeout and
    # those may time out, before other ranks have finished with initialization done
    # under the old/slow timeout.
    torch.distributed.barrier()
    torch.cuda.synchronize()

    groups = (
        [world_mesh.get_group()] if world_mesh.ndim == 1 else world_mesh.get_group()
    )

    # None represents the 'default' PG, not part of the mesh
    groups.append(None)
    for group in groups:
        torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)


def init_distributed(job_config):
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
        "nccl", timeout=timedelta(seconds=job_config.comm.init_timeout_seconds)
    )


@dataclass
class ParallelDims:
    dp: int
    sp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp, sp, pp = self.dp, self.sp, self.pp
        if dp == -1:
            self.dp = dp = self.world_size // (sp * pp)
        assert dp >= 1, dp
        assert sp >= 1, sp
        assert pp >= 1, pp
        assert (
            dp * sp * pp == self.world_size
        ), f"Invalid parallel dims: dp({dp}) * sp({sp}) * pp({pp}) != WORLD_SIZE({self.world_size})"

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.dp, self.sp, self.pp], ["dp", "sp", "pp"], strict=True
        ):
            if d > 1:
                dims.append(d)
                names.append(name)
        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        return init_device_mesh(device_type, dims, mesh_dim_names=names)

    @property
    def dp_enabled(self):
        return self.dp > 1

    @property
    def sp_enabled(self):
        return self.sp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.sp > 1 and self.enable_loss_parallel

    @cached_property
    def model_parallel_size(self):
        return self.sp * self.pp
