# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import pickle
import time

import torch

from torchtitan.config_manager import JobConfig
from torchtitan.tools.logging import logger

# the number of warmup steps before the active step in each profiling cycle
WARMUP = 3

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


@contextlib.contextmanager
def maybe_enable_profiling(config: JobConfig, *, global_step: int = 0):
    # get user defined profiler settings
    enable_profiling = config.profiling.enable_profiling

    if enable_profiling:
        dump_dir = config.job.dump_folder
        save_trace_dir = config.profiling.save_traces_folder
        trace_dir = os.path.join(dump_dir, save_trace_dir)
        profile_freq = config.profiling.profile_freq

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()
            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")
            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        warmup, active = WARMUP, 1
        wait = profile_freq - (active + warmup)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        gpu_device_profiled = None
        if torch.cuda.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.CUDA
        elif torch.xpu.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.XPU
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                gpu_device_profiled,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=True,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_memory_snapshot(config: JobConfig, *, global_step: int = 0):
    enable_snapshot = config.profiling.enable_memory_snapshot
    if enable_snapshot:
        snapshot_folder = config.profiling.save_memory_snapshot_folder
        snapshot_dir = os.path.join(config.job.dump_folder, snapshot_folder)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        rank = torch.distributed.get_rank()

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                torch.cuda.memory._record_memory_history(
                    max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES
                )
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                if not exit_ctx and self.step_num % self.freq != 0:
                    return
                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name)
                if not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)
                logger.info(f"Dumping memory snapshot at step {curr_step}")
                begin = time.monotonic()
                with open(
                    f"{curr_snapshot_dir}/rank{rank}_memory_snapshot.pickle", "wb"
                ) as output:
                    pickle.dump(torch.cuda.memory._snapshot(), output)
                logger.info(
                    f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
                )

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        profiler = MemoryProfiler(global_step, config.profiling.profile_freq)
        try:
            yield profiler
        except torch.OutOfMemoryError as e:
            profiler.step(exit_ctx=True)
    else:
        yield None
