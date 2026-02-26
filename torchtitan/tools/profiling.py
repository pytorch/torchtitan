# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torchtitan.config import Configurable
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000

_COMM_KEYWORDS: tuple[str, ...] = ("nccl",)
_COMPUTE_KEYWORDS: tuple[str, ...] = (
    "gemm",
    "aten",
    "cublas",
    "cutlass",
    "cudnn",
    "triton",
    "flash",
)


class ProfileAnalyzer(ABC):
    """Abstract base class for profiler trace analyzers.

    Implement this interface to create custom analyzers that run automatically
    after each profiler trace is exported.
    """

    @abstractmethod
    def analyze(self, prof: torch.profiler.profile) -> None:
        """Analyze a completed profiler trace.

        Args:
            prof: A ``torch.profiler.profile`` object with collected trace data.
        """
        ...


class CommsComputeOverlapAnalyzer(ProfileAnalyzer):
    """Analyzes compute-communication overlap from a PyTorch profiler trace.

    Computes overlap efficiency: the fraction of NCCL communication time that
    runs concurrently with compute kernels. Values close to 100% indicate
    optimal overlap; values near 0% indicate the workload is communication bound.

    Note:
        This analysis uses aggregated kernel times from ``key_averages()``, which
        sums durations across all invocations. When multiple kernels run concurrently
        on different CUDA streams, this may underestimate actual overlap. For precise
        timeline analysis, inspect the exported Chrome trace directly.
    """

    def _get_trace_duration_us(self, prof: torch.profiler.profile) -> float:
        """Compute trace duration from raw event timestamps."""
        try:
            events = prof.events()
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

    def analyze(self, prof: torch.profiler.profile) -> None:
        """Run overlap analysis and log a summary to the console."""
        key_averages = prof.key_averages()

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
                "[CommsComputeOverlapAnalyzer] No NCCL kernels found in trace. "
                "Skipping overlap report."
            )
            return

        trace_duration_us = self._get_trace_duration_us(prof)
        if trace_duration_us == 0.0:
            trace_duration_us = compute_us + comm_us

        trace_duration_us = max(trace_duration_us, compute_us, comm_us)

        raw_overlap = compute_us + comm_us - trace_duration_us
        overlap_pct = max(0.0, min(raw_overlap / comm_us * 100.0, 100.0))

        # Calculate GPU time spent on unoverlapped communication (unutilized for compute)
        unoverlapped_comm_pct = 100.0 - overlap_pct

        logger.info(
            "[CommsComputeOverlapAnalyzer] Compute-Communication Overlap Report\n"
            f"  Total Compute Time               : {compute_us / 1e3:.2f} ms\n"
            f"  Total NCCL Time                  : {comm_us / 1e3:.2f} ms\n"
            f"  Total Trace Time                 : {trace_duration_us / 1e3:.2f} ms\n"
            f"  Overlap Efficiency               : {overlap_pct:.1f} %\n"
            f"  GPU Unutilized due to NCCL comms : {unoverlapped_comm_pct:.1f} %"
        )


class MemoryProfiler:
    """Records periodic memory snapshots during training.

    Started by :meth:`Profiler.build_memory_profiler` when memory snapshots are
    enabled. Call :meth:`step` once per training iteration to trigger periodic
    dumps; pass ``exit_ctx=True`` to force a final dump on OOM.
    """

    def __init__(
        self,
        step_num: int,
        freq: int,
        snapshot_dir: str,
        leaf_folder: str,
        rank: int,
    ) -> None:
        device_module.memory._record_memory_history(
            max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES
        )
        # when resume training, we start from the last step
        self.step_num = step_num
        self.freq = freq
        self._snapshot_dir = snapshot_dir
        self._leaf_folder = leaf_folder
        self._rank = rank

    def step(self, exit_ctx: bool = False) -> None:
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
        curr_snapshot_dir = os.path.join(
            self._snapshot_dir, dir_name, self._leaf_folder
        )
        if not os.path.exists(curr_snapshot_dir):
            os.makedirs(curr_snapshot_dir, exist_ok=True)
        logger.info(f"Dumping memory snapshot at step {curr_step}")
        begin = time.monotonic()
        output_file = os.path.join(
            curr_snapshot_dir, f"rank{self._rank}_memory_snapshot.pickle"
        )
        with open(output_file, "wb") as output:
            pickle.dump(device_module.memory._snapshot(), output)
        logger.info(
            f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
        )


class Profiler(Configurable):
    """Owns profiling and memory snapshot lifecycle for a training run.

    Instantiate via ``Profiler.Config.build()`` and use as a context manager in
    the training loop via :meth:`active`.

    If ``config.enable_overlap_analysis`` is ``True``, a
    :class:`CommsComputeOverlapAnalyzer` is automatically added to the internal
    analyzers list and run after each trace export.

    Example::

        with self.profiler.active(global_step=step, base_folder=folder) as prof:
            for step in training_loop:
                ...
                prof.step()

    Args:
        config: A ``Profiler.Config`` instance.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable_profiling: bool = False
        """Whether to enable pytorch profiler."""

        save_traces_folder: str = "profile_traces"
        """Trace files location."""

        profile_freq: int = 10
        """How often to collect profile traces, in iterations."""

        profiler_active: int = 1
        """
        The steps profiler is active for.

        This is used to configure torch.profiler.schedule.
        """

        profiler_warmup: int = 3
        """
        The number of warmup steps before the active step in each profiling cycle.

        This is used to configure torch.profiler.schedule.
        """

        enable_memory_snapshot: bool = False
        """Whether to dump memory snapshot."""

        save_memory_snapshot_folder: str = "memory_snapshot"
        """Memory snapshot files location."""

        enable_overlap_analysis: bool = False
        """
        Enable compute-communication overlap analysis after each profiler trace.

        When set to true, a CommsComputeOverlapAnalyzer is run after each trace
        export to report compute-communication overlap efficiency. This is only
        active when profiling is enabled and the process group is initialized
        (distributed run).
        """

    def __init__(self, config: "Profiler.Config") -> None:
        self._config = config
        self._analyzers: list[ProfileAnalyzer] = []
        if config.enable_overlap_analysis:
            self._analyzers.append(CommsComputeOverlapAnalyzer())
        # TODO: support list[ProfileAnalyzer.Config] in Profiler.Config for extensible analyzers
        # runtime args stored by active(); defaults used if active() not called
        self._global_step: int = 0
        self._base_folder: str = ""
        self._leaf_folder: str = ""
        self.torch_profiler = None
        self.memory_profiler = None

    def active(
        self,
        *,
        global_step: int = 0,
        base_folder: str = "",
        leaf_folder: str = "",
    ) -> "Profiler":
        """Store runtime args and return self for use as a context manager.

        Example::

            with self.profiler.active(global_step=step, base_folder=folder) as prof:
                prof.step()
        """
        self._global_step = global_step
        self._base_folder = base_folder
        self._leaf_folder = leaf_folder
        return self

    def __enter__(self) -> "Profiler":
        self.torch_profiler = self.build_torch_profiler(
            global_step=self._global_step,
            base_folder=self._base_folder,
            leaf_folder=self._leaf_folder,
        )
        self.memory_profiler = self.build_memory_profiler(
            global_step=self._global_step,
            base_folder=self._base_folder,
            leaf_folder=self._leaf_folder,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.torch_profiler is not None:
            self.torch_profiler.__exit__(exc_type, exc_val, exc_tb)
            self.torch_profiler = None
        if self.memory_profiler is not None:
            if isinstance(exc_val, torch.OutOfMemoryError):
                self.memory_profiler.step(exit_ctx=True)
            self.memory_profiler = None
        return False

    def step(self) -> None:
        """Advance all active profilers by one training step."""
        if self.torch_profiler is not None:
            self.torch_profiler.step()
        if self.memory_profiler is not None:
            self.memory_profiler.step()

    def build_torch_profiler(
        self,
        *,
        global_step: int,
        base_folder: str,
        leaf_folder: str,
    ):
        """Create, start, and return the torch profiler, or ``None`` if disabled.

        Calls ``torch.profiler.profile.__enter__()`` so the returned handle is
        already active. :meth:`__exit__` is responsible for stopping it.
        """
        cfg = self._config
        if not cfg.enable_profiling:
            return None

        trace_dir = os.path.join(base_folder, cfg.save_traces_folder)
        profile_freq, warmup, active = (
            cfg.profile_freq,
            cfg.profiler_warmup,
            cfg.profiler_active,
        )

        rank = torch.distributed.get_rank()
        run_diagnostics = (
            bool(self._analyzers)
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
                for analyzer in self._analyzers:
                    analyzer.analyze(prof)

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

        torch_profiler = torch.profiler.profile(
            # pyrefly: ignore [bad-argument-type]
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                gpu_device_profiled,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=True,
        )
        torch_profiler.__enter__()
        torch_profiler.step_num = global_step
        return torch_profiler

    def build_memory_profiler(
        self,
        *,
        global_step: int,
        base_folder: str,
        leaf_folder: str,
    ):
        """Create and return a :class:`MemoryProfiler`, or ``None`` if disabled.

        :class:`MemoryProfiler.__init__` starts memory history recording immediately.
        """
        cfg = self._config
        if not cfg.enable_memory_snapshot:
            return None

        snapshot_dir = os.path.join(base_folder, cfg.save_memory_snapshot_folder)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        rank = torch.distributed.get_rank()

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        return MemoryProfiler(global_step, cfg.profile_freq, snapshot_dir, leaf_folder, rank)
