# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import pickle
import time
from dataclasses import dataclass

import torch
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


# TODO: introduce an owner class, namely Profiler
@dataclass(kw_only=True, slots=True)
class ProfilingConfig:
    enable_profiling: bool = False
    """Whether to enable pytorch profile"""

    save_traces_folder: str = "profile_traces"
    """Trace files location"""

    profile_freq: int = 10
    """How often to collect profile traces, in iterations"""

    profiler_active: int = 1
    """
    The steps profiler is active for.

    This is used to configure torch.profile.schedule.
    """

    profiler_warmup: int = 3
    """
    The number of warmup steps before the active step in each profiling cycle.

    This is used to configure torch.profile.schedule.
    """

    enable_memory_snapshot: bool = False
    """Whether to dump memory snapshot"""

    save_memory_snapshot_folder: str = "memory_snapshot"
    """Memory snapshot files location"""


@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    # get user defined profiler settings
    enable_profiling = profiling_config.enable_profiling

    if enable_profiling:
        trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)
        profile_freq, warmup, active = (
            profiling_config.profile_freq,
            profiling_config.profiler_warmup,
            profiling_config.profiler_active,
        )

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name, leaf_folder)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()

            output_file = os.path.join(curr_trace_dir, f"rank{rank}_trace.json")
            prof.export_chrome_trace(output_file)
            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

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
            # pyrefly: ignore [bad-argument-type]
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
def maybe_enable_memory_snapshot(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    enable_snapshot = profiling_config.enable_memory_snapshot
    if enable_snapshot:
        snapshot_dir = os.path.join(
            base_folder, profiling_config.save_memory_snapshot_folder
        )
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        rank = torch.distributed.get_rank()

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                device_module.memory._record_memory_history(
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
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name, leaf_folder)
                if not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)
                logger.info(f"Dumping memory snapshot at step {curr_step}")
                begin = time.monotonic()
                output_file = os.path.join(
                    curr_snapshot_dir, f"rank{rank}_memory_snapshot.pickle"
                )
                with open(output_file, "wb") as output:
                    pickle.dump(device_module.memory._snapshot(), output)
                logger.info(
                    f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
                )

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        profiler = MemoryProfiler(global_step, profiling_config.profile_freq)
        try:
            yield profiler
        except torch.OutOfMemoryError:
            profiler.step(exit_ctx=True)
            raise
    else:
        yield None
