# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Union

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import DeviceMesh
from torchtrain.logging_utils import logger


def dist_max(x: Union[int, float], mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x).cuda()
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.MAX.name, group=mesh)


def dist_mean(x: Union[int, float], mesh: DeviceMesh) -> float:
    tensor = torch.tensor(x).cuda()
    return funcol.all_reduce(tensor, reduceOp=c10d.ReduceOp.AVG.name, group=mesh)


def _warn_overwrite_env(env, val):
    if env in os.environ:
        logger.warning(
            f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
        )
    os.environ[env] = val


def set_pg_timeouts(timeout, world_mesh):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, becuase
    otherwise you may face a race where the slow rank has not reached the timeout reduction point
    yet due to slow operations permitted under the old timeout value, but other faster ranks may
    start issueing collectives under the new shorter timeout and then immediately timeout.
    """
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


TRACE_BUFFER_SIZE = "TORCH_NCCL_TRACE_BUFFER_SIZE"
TRACE_FILE = "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"
DUMP_ON_TIMEOUT = "TORCH_NCCL_DUMP_ON_TIMEOUT"
ASYNC_ERROR_HANDLING = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
SKIP_CLEANUP = "3"


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
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"


@dataclass
class Background:
    black = "\033[40m"
    red = "\033[41m"
    green = "\033[42m"
    yellow = "\033[43m"
    blue = "\033[44m"
    magenta = "\033[45m"
    cyan = "\033[46m"
    white = "\033[47m"
    reset = "\033[49m"


@dataclass
class Style:
    bright = "\033[1m"
    dim = "\033[2m"
    normal = "\033[22m"
    reset = "\033[0m"
