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

from torchtitan.config import Profiling as ProfilingConfig
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000

# Kernel name fragments used to classify events by type.
_COMM_KEYWORDS: tuple[str, ...] = ("nccl",)
_COMPUTE_KEYWORDS: tuple[str, ...] = ("gemm", "aten", "cublas")


class OverlapAnalyzer:
    """Analyzes compute-communication overlap from a PyTorch profiler trace.

    Uses ``prof.key_averages()`` to extract Kineto events and computes three
    aggregate metrics:

    * **Total NCCL time** – sum of CUDA time for kernels whose name contains
      ``"nccl"``.
    * **Total compute time** – sum of CUDA time for kernels whose name contains
      ``"gemm"``, ``"aten"``, or ``"cublas"``.
    * **Overlap efficiency** – fraction of communication time that is hidden
      behind compute, expressed as a percentage::

          (compute_us + comm_us - trace_duration_us) / comm_us * 100

      Values close to 100 % indicate that nearly all communication is
      overlapped with compute (*OPTIMAL*); values near 0 % indicate that the
      workload is *COMMUNICATION BOUND*.

    Args:
        prof: A ``torch.profiler.profile`` object whose trace has been
            collected (i.e. the profiler context has already been stepped or
            exited so that events are available via ``key_averages()``).
    """

    def __init__(self, prof: torch.profiler.profile) -> None:
        self._prof = prof

    def analyze(self) -> None:
        """Run the overlap analysis and log a summary to the console.

        The summary includes Total Compute Time, Total NCCL Time, Overlap
        Efficiency, and a qualitative Status recommendation. When communication
        time is zero (e.g. single-device run) the method logs a notice and
        returns early instead of dividing by zero.
        """
        events = self._prof.key_averages()

        comm_us: float = 0.0
        compute_us: float = 0.0
        trace_duration_us: float = 0.0

        for evt in events:
            name_lower = evt.key.lower()
            cuda_time = evt.cuda_time_total  # microseconds

            if any(kw in name_lower for kw in _COMM_KEYWORDS):
                comm_us += cuda_time
            elif any(kw in name_lower for kw in _COMPUTE_KEYWORDS):
                compute_us += cuda_time

            # Accumulate total trace wall-time via self_cpu_time_total for all
            # top-level events; use the maximum single-event self CUDA time as
            # a proxy for the profiled window duration when the profiler does
            # not expose it directly.
            trace_duration_us = max(trace_duration_us, cuda_time)

        if comm_us == 0.0:
            logger.info(
                "[OverlapAnalyzer] No NCCL kernels found in trace "
                "(single-device run or comm kernels not captured). Skipping overlap report."
            )
            return

        # Clamp numerator to [0, comm_us] to avoid nonsensical values caused
        # by the coarse trace_duration_us proxy.
        raw_overlap = compute_us + comm_us - trace_duration_us
        overlap_pct = max(0.0, min(raw_overlap / comm_us * 100.0, 100.0))

        status = "OPTIMAL" if overlap_pct >= 50.0 else "COMMUNICATION BOUND"

        logger.info(
            "[OverlapAnalyzer] Compute-Communication Overlap Report\n"
            f"  Total Compute Time : {compute_us / 1e3:.2f} ms\n"
            f"  Total NCCL Time    : {comm_us / 1e3:.2f} ms\n"
            f"  Overlap Efficiency : {overlap_pct:.1f} %\n"
            f"  Status             : {status}"
        )


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
        run_diagnostics = (
            profiling_config.experimental_diagnostics
            and torch.distributed.is_initialized()
            and rank == 0
        )

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

            if run_diagnostics:
                OverlapAnalyzer(prof).analyze()

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
        except torch.OutOfMemoryError as e:
            profiler.step(exit_ctx=True)
            raise
    else:
        yield None
