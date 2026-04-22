# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import time
from dataclasses import dataclass
from typing import Annotated

import torch
import tyro


from torchtitan.config import Configurable
from torchtitan.config.function import Function
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import device_module

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000


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

    Example::

        with Profiler(config, global_step=step, base_folder=folder) as prof:
            for step in training_loop:
                ...
                prof.step()

    Args:
        config: A ``Profiler.Config`` instance.
        global_step: The training step at which profiling begins.  When
            resuming from a checkpoint this should be the loaded step so that
            trace directories are named correctly (e.g. ``iteration_100``
            instead of ``iteration_0``) and memory-snapshot frequency alignment
            is preserved.
        base_folder: Root directory for profiler trace and memory snapshot output.
        leaf_folder: Optional subdirectory appended to trace/snapshot paths
            (e.g. per-replica folder in fault-tolerant training).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        enable_profiling: bool = False
        """Whether to enable pytorch profiler."""

        save_traces_folder: str = "profile_traces"
        """Trace files location."""

        profile_freq: int = 10
        """How often to collect profile traces, in iterations."""

        profiler_repeat: int | None = None
        """
        The number of times to repeat the profiling cycle

        This is used to configure torch.profiler.schedule.
        """

        profiler_skip_first: int | None = None
        """
        The number of initial profiling cycles to skip

        This is used to configure torch.profiler.schedule.
        """

        profiler_skip_first_wait: int | None = None
        """
        The number of initial profiling cycles to skip the wait time

        This is used to configure torch.profiler.schedule.
        """

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

        profiler_repeat: int | None = None
        """
        The number of times to repeat the profiling cycle

        This is used to configure torch.profiler.schedule.
        """

        profiler_skip_first: int | None = None
        """
        The number of initial profiling cycles to skip

        This is used to configure torch.profiler.schedule.
        """

        profiler_skip_first_wait: int | None = None
        """
        The number of initial profiling cycles to skip the wait time

        This is used to configure torch.profiler.schedule.
        """

        enable_memory_snapshot: bool = False
        """Whether to dump memory snapshot."""

        save_memory_snapshot_folder: str = "memory_snapshot"
        """Memory snapshot files location."""

        trace_post_processor: Annotated[
            Function.Config | None, tyro.conf.Suppress
        ] = None
        """Optional hook invoked with the trace path after each export.

        Wraps ``fn(trace_path: str) -> None``.
        Set programmatically (not via CLI) — tyro cannot parse Callable types.
        """

    def __init__(
        self,
        config: Config,
        *,
        global_step: int = 0,
        base_folder: str = "",
        leaf_folder: str = "",
    ) -> None:
        self._config = config
        self._global_step = global_step
        self._base_folder = base_folder
        self._leaf_folder = leaf_folder
        self.torch_profiler = None
        self.memory_profiler = None

    def active(
        self,
        *,
        global_step: int = 0,
        base_folder: str = "",
        leaf_folder: str = "",
    ) -> "Profiler":
        """Update runtime args and return self for use as a context manager.

        This allows a pre-built :class:`Profiler` (e.g. built during
        ``__init__``) to be activated later with runtime parameters::

            self.profiler = config.profiler.build()
            ...
            with self.profiler.active(global_step=step, base_folder=folder) as p:
                ...
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

        additional_params = {
            key: val
            for key, val in [
                ("repeat", cfg.profiler_repeat),
                ("skip_first", cfg.profiler_skip_first),
                ("skip_first_wait", cfg.profiler_skip_first_wait),
            ]
            if val is not None
        }

        rank = torch.distributed.get_rank()
        post_processor = (
            cfg.trace_post_processor.build() if cfg.trace_post_processor else None
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

            if post_processor is not None:
                post_processor(output_file)

            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        additional_params = {
            key: val
            for key, val in [
                ("repeat", cfg.profiler_repeat),
                ("skip_first", cfg.profiler_skip_first),
                ("skip_first_wait", cfg.profiler_skip_first_wait),
            ]
            if val is not None
        }

        wait = profile_freq - (active + warmup)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif torch.xpu.is_available():
            activities.append(torch.profiler.ProfilerActivity.XPU)

        torch_profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=wait, warmup=warmup, active=active, **additional_params
            ),
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
        return MemoryProfiler(
            global_step, cfg.profile_freq, snapshot_dir, leaf_folder, rank
        )
