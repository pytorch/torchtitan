# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import time

import torch
from torchtitan.config_manager import JobConfig
from torchtitan.logging_utils import logger

# the number of warmup steps before the active step in each profiling cycle
WARMUP = 3


@contextlib.contextmanager
def maybe_enable_profiling(config: JobConfig, *pos_args, **kwargs):
    # get user defined profiler settings
    enable_profiling = config.profiling.enable_profiling

    if enable_profiling:
        dump_dir = config.job.dump_folder
        save_trace_dir = config.profiling.save_traces_folder
        trace_dir = os.path.join(dump_dir, save_trace_dir)
        profile_freq = config.profiling.profile_freq

        _global_iter_count = 0

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            nonlocal _global_iter_count
            _global_iter_count += profile_freq
            curr_trace_dir_name = "iteration_" + str(_global_iter_count)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping traces at step {_global_iter_count}")
            begin = time.monotonic()
            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")
            logger.info(
                f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds"
            )
            # Profiling is a heavy operation which could cost very different amount of time
            # across all ranks. Insert a barrier to make sure all ranks have finished profiling
            # before moving on.
            # TODO: Can we find a cleaner way?
            torch.distributed.barrier()

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = WARMUP, 1
        wait = profile_freq - (active + warmup)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None
