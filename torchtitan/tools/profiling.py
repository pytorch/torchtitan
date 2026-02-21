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

_COMM_KEYWORDS: tuple[str, ...] = ("nccl",)
_COMPUTE_KEYWORDS: tuple[str, ...] = (
    "gemm", "aten", "cublas", "cutlass", "cudnn", "triton", "flash",
)


class OverlapAnalyzer:
    """Analyzes compute-communication overlap from a PyTorch profiler trace.

    Computes overlap efficiency: the fraction of NCCL communication time that
    runs concurrently with compute kernels. Values close to 100% indicate
    optimal overlap; values near 0% indicate the workload is communication bound.

    Note:
        This analysis uses aggregated kernel times from ``key_averages()``, which
        sums durations across all invocations. When multiple kernels run concurrently
        on different CUDA streams, this may underestimate actual overlap. For precise
        timeline analysis, inspect the exported Chrome trace directly.

    Args:
        prof: A ``torch.profiler.profile`` object with collected trace data.
    """

    def __init__(self, prof: torch.profiler.profile) -> None:
        self._prof = prof

    def _get_trace_duration_us(self) -> float:
        """Compute trace duration from raw event timestamps."""
        try:
            events = self._prof.events()
        except (AttributeError, RuntimeError, AssertionError):
            return 0.0

        if not events:
            return 0.0

        min_start = float("inf")
        max_end = float("-inf")

        for evt in events:
            if hasattr(evt, "time_range") and hasattr(evt.time_range, "start"):
                try:
                    min_start = min(min_start, evt.time_range.start)
                    max_end = max(max_end, evt.time_range.end)
                except (AttributeError, TypeError):
                    continue

        if min_start == float("inf") or max_end == float("-inf"):
            return 0.0

        return max(0.0, max_end - min_start)

    def analyze(self) -> None:
        """Run overlap analysis and log a summary to the console."""
        key_averages = self._prof.key_averages()

        comm_us: float = 0.0
        compute_us: float = 0.0

        for evt in key_averages:
            name_lower = evt.key.lower()
            device_time = evt.self_device_time_total

            if any(kw in name_lower for kw in _COMM_KEYWORDS):
                comm_us += device_time
            elif any(kw in name_lower for kw in _COMPUTE_KEYWORDS):
                compute_us += device_time

        if comm_us == 0.0:
            logger.info(
                "[OverlapAnalyzer] No NCCL kernels found in trace. "
                "Skipping overlap report."
            )
            return

        trace_duration_us = self._get_trace_duration_us()
        if trace_duration_us == 0.0:
            trace_duration_us = compute_us + comm_us

        trace_duration_us = max(trace_duration_us, compute_us, comm_us)

        raw_overlap = compute_us + comm_us - trace_duration_us
        overlap_pct = max(0.0, min(raw_overlap / comm_us * 100.0, 100.0))

        status = "OPTIMAL" if overlap_pct >= 50.0 else "COMMUNICATION BOUND"

        logger.info(
            "[OverlapAnalyzer] Compute-Communication Overlap Report\n"
            f"  Total Compute Time : {compute_us / 1e3:.2f} ms\n"
            f"  Total NCCL Time    : {comm_us / 1e3:.2f} ms\n"
            f"  Total Trace Time   : {trace_duration_us / 1e3:.2f} ms\n"
            f"  Overlap Efficiency : {overlap_pct:.1f}% (conservative lower bound)\n"
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
