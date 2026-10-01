# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch profiler + memory-snapshot lifecycle."""

import inspect
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import torch
import torch._C._profiler
import tyro

from torchtitan.config import Configurable
from torchtitan.distributed.cuda_graph import get_cuda_graph_annotations
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.utils import device_module

# torch's export_chrome_trace gained cuda_graph_annotations when the offline joiner
# (torch.cuda._annotate_cuda_graph_trace) was removed. Older versions still export, just
# without the CUDA graph annotations baked in.
logger = logging.getLogger(__name__)


_EXPORT_SUPPORTS_ANNOTATIONS = (
    "cuda_graph_annotations"
    in inspect.signature(torch.profiler.profile.export_chrome_trace).parameters
)

# Paths expects by meta internal tooling
PROFILE_DIR = "profiling/traces"  # Profiler.Config.save_traces_folder default
PROFILE_ITER_DIR = "iteration_{step}"  # PROFILE_DIR/{PROFILE_ITER_DIR}
PROFILE_FILE = "rank{rank}_trace.json.gz"  # PROFILE_DIR/PROFILE_ITER_DIR/{PROFILE_FILE}
PFTRACE_PROFILE_FILE = "rank{rank}_trace.pftrace.gz"

MEMORY_DIR = (
    "profiling/memory_snapshot"  # Profiler.Config.save_memory_snapshot_folder default
)
MEMORY_STEP_DIR = "step_{step:012d}"  # MEMORY_DIR/{MEMORY_STEP_DIR}
MEMORY_EXIT_DIR = "step_{step:012d}_exit"  # OOM dump variant
MEMORY_FILE = (
    "{rank:06d}_step_{step}.pickle"  # MEMORY_DIR/MEMORY_STEP_DIR/{MEMORY_FILE}
)


@dataclass(kw_only=True, slots=True)
class CuspyProfilerConfig:
    """Configure Cuspy profiling, used instead of Kineto for CUDA activity."""

    trace_format: Literal["json", "pftrace"] = "json"
    """Trace output format. Perfetto protobuf traces require Cuspy."""

    enable_cuda_sync_events: bool = False
    """Collect CUDA synchronization activities."""

    enable_environment_counters: bool = False
    """Collect GPU power, clock, thermal, and cooling counters."""

    enable_graph_dependencies: bool = False
    """Record CUDA graph dependency edges."""

    enable_event_node_ids: bool = False
    """Associate CUDA events with CUDA graph event-record nodes."""

    performance_metrics: list[str] = field(default_factory=list)
    """CUPTI PM-sampling metric names to collect."""

    pm_sampling_interval_ms: float | None = None
    """PM-sampling interval; ``None`` uses torch's default."""

    pm_lookback_window_ms: float | None = None
    """PM sample buffer length. Sampling starts at profiler warmup, so keep warmup
    plus active steps within this window or the latest samples are lost. ``None``
    uses torch's default."""

    def __post_init__(self) -> None:
        if not self.performance_metrics and (
            self.pm_sampling_interval_ms is not None
            or self.pm_lookback_window_ms is not None
        ):
            raise ValueError(
                "pm_sampling_interval_ms and pm_lookback_window_ms require "
                "performance_metrics."
            )


def _cuspy_unavailable_reasons(config: CuspyProfilerConfig) -> list[str]:
    """Return why Cuspy cannot run ``config`` in this environment; empty when it can.

    Probes without taking a CUPTI subscription: holding one makes Kineto's one-shot
    CUPTI init fail for the rest of the process.
    """
    reasons = []
    if getattr(torch.profiler, "CuspyConfig", None) is None:
        reasons.append("this torch build has no torch.profiler.CuspyConfig")
    elif config.trace_format == "pftrace":
        # torch has no capability flag for the native encoder; builds without it
        # compile a stub that raises on any input, so encode an empty trace.
        try:
            torch._C._profiler._cuspy.encode_pftrace(0, [], [], [], [])
        except RuntimeError as error:
            reasons.append(f'trace_format="pftrace" is unsupported: {error}')
    # Private module, and importing it raises without the cupti-python package.
    try:
        from torch.profiler._cuspy import cupti_python
    except ModuleNotFoundError as error:
        reasons.append(str(error))
        return reasons
    try:
        version = cupti_python.pylibcupti().get_version()
    except (AttributeError, OSError, RuntimeError) as error:
        reasons.append(
            f"could not load {cupti_python.LIBCUPTI_SONAME} ({error}); this torch "
            f"is built against CUDA {torch.version.cuda}"
        )
        return reasons
    if version < cupti_python.LIBCUPTI_MIN_VERSION:
        reasons.append(
            f"Cuspy needs libcupti >= {cupti_python.LIBCUPTI_MIN_VERSION}, found "
            f"{version}; install a newer nvidia-cuda-cupti"
        )
    return reasons


def _cuspy_activity_config(config: CuspyProfilerConfig) -> dict[Any, Any]:
    profiler_configs: list[Any] = [
        torch.profiler.CuspyConfig(
            enable_cuda_sync_events=config.enable_cuda_sync_events,
            enable_environment_counters=config.enable_environment_counters,
            enable_graph_dependencies=config.enable_graph_dependencies,
            enable_event_node_ids=config.enable_event_node_ids,
        )
    ]
    if config.performance_metrics:
        profiler_configs.append(
            torch.profiler.PerformanceMetricsConfig(
                metric_names=config.performance_metrics,
                sampling_interval_ms=config.pm_sampling_interval_ms,
                lookback_window_ms=config.pm_lookback_window_ms,
            )
        )
    return {
        torch.profiler.ProfilerActivity.CUDA: torch.profiler.ProfilerActivityConfig(
            profiler_configs=profiler_configs
        )
    }


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
        max_entries: int,
    ) -> None:
        device_module.memory._record_memory_history(
            # stacks="python" records only Python frames (not C++), which is much
            # cheaper to capture and serialize, keeping the snapshot dump from
            # taking minutes -- the default stacks="all" symbolizes C++ frames,
            # which is very slow for torchtitan workload.
            stacks="python",
            max_entries=max_entries,
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
            dir_name = MEMORY_STEP_DIR.format(step=curr_step)
        else:
            # dump as step_000000000000_exit if OOM at iter 1
            curr_step = self.step_num - 1
            dir_name = MEMORY_EXIT_DIR.format(step=curr_step)
        curr_snapshot_dir = os.path.join(
            self._snapshot_dir, dir_name, self._leaf_folder
        )
        if not os.path.exists(curr_snapshot_dir):
            os.makedirs(curr_snapshot_dir, exist_ok=True)
        logger.info(f"Dumping memory snapshot at step {curr_step}")
        begin = time.monotonic()
        output_file = os.path.join(
            curr_snapshot_dir, MEMORY_FILE.format(rank=self._rank, step=curr_step)
        )
        with open(output_file, "wb") as output:
            # Protocol 4 for compatibility with pytorch.org/memory_viz JS parser
            pickle.dump(device_module.memory._snapshot(), output, protocol=4)
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

        save_traces_folder: str = PROFILE_DIR
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

        cuspy: Annotated[CuspyProfilerConfig | None, tyro.conf.Suppress] = None
        """Cuspy profiler settings; ``None`` profiles CUDA with Kineto. Set from a
        Python configuration."""

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

        memory_snapshot_freq: int | None = None
        """How often to collect memory snapshots, in iterations.

        Defaults to ``profile_freq`` when unset for backward compatibility.
        """

        save_memory_snapshot_folder: str = MEMORY_DIR
        """Memory snapshot files location."""

        memory_snapshot_max_entries: int = 1_000_000
        """Max alloc/free events recorded per memory snapshot (ring buffer).

        Caps the history passed to ``_record_memory_history``; the oldest events
        are dropped once full. Bounds host memory and snapshot size / dump time.
        """

        def __post_init__(self) -> None:
            if self.enable_profiling and self.profile_freq < (
                self.profiler_warmup + self.profiler_active
            ):
                raise ValueError(
                    "profiler.profile_freq must be greater than or equal to "
                    "profiler_warmup + profiler_active."
                )

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
        def caused_by_oom(exc: BaseException | None) -> bool:
            """Whether an OutOfMemoryError appears anywhere in the cause chain.

            Pipeline parallelism does not re-raise the OOM it catches; it wraps
            the failure in a plain RuntimeError to attach the stage's shapes.
            """
            seen: set[int] = set()
            while exc is not None and id(exc) not in seen:
                if isinstance(exc, torch.OutOfMemoryError):
                    return True
                seen.add(id(exc))
                exc = exc.__cause__ or exc.__context__
            return False

        if self.torch_profiler is not None:
            self.torch_profiler.__exit__(exc_type, exc_val, exc_tb)
            self.torch_profiler = None
        if self.memory_profiler is not None:
            if caused_by_oom(exc_val):
                self.memory_profiler.step(exit_ctx=True)
            self.memory_profiler = None
        return False

    def step(self) -> None:
        """Advance all active profilers by one training step."""

        if self.torch_profiler is not None:
            with sl.log_trace_span("kineto_profiler_step_call"):
                self.torch_profiler.step()
        if self.memory_profiler is not None:
            with sl.log_trace_span("memory_profiler_step_call"):
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
        cuspy = cfg.cuspy

        trace_dir = os.path.join(base_folder, cfg.save_traces_folder)
        profile_freq, warmup, active = (
            cfg.profile_freq,
            cfg.profiler_warmup,
            cfg.profiler_active,
        )

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = PROFILE_ITER_DIR.format(step=prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name, leaf_folder)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()

            profile_file = (
                PFTRACE_PROFILE_FILE
                if cuspy is not None and cuspy.trace_format == "pftrace"
                else PROFILE_FILE
            )
            output_file = os.path.join(curr_trace_dir, profile_file.format(rank=rank))
            if cuspy is not None:
                prof.export_chrome_trace(output_file)
            else:
                # CUDA graph annotations are baked in during the export rather than
                # joined onto the written file afterwards: re-reading and rewriting a
                # gzipped trace paid the compression cost twice.
                annotations = get_cuda_graph_annotations()
                if annotations and not _EXPORT_SUPPORTS_ANNOTATIONS:
                    logger.warning(
                        "This torch does not support cuda_graph_annotations on "
                        "export_chrome_trace; the trace will have no CUDA graph kernel "
                        "annotations."
                    )
                    annotations = None
                extra = {"cuda_graph_annotations": annotations} if annotations else {}
                prof.export_chrome_trace(output_file, **extra)

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
        activities: list[Any] = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            if cuspy is not None:
                if reasons := _cuspy_unavailable_reasons(cuspy):
                    raise ValueError(
                        "profiler.cuspy is set, but Cuspy cannot run here:\n  - "
                        + "\n  - ".join(reasons)
                    )
                activities.append(_cuspy_activity_config(cuspy))
            else:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif torch.xpu.is_available():
            if cuspy is not None:
                raise ValueError("profiler.cuspy requires CUDA.")
            activities.append(torch.profiler.ProfilerActivity.XPU)
        elif cuspy is not None:
            raise ValueError("profiler.cuspy requires CUDA.")

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

        memory_snapshot_freq = (
            cfg.profile_freq
            if cfg.memory_snapshot_freq is None
            else cfg.memory_snapshot_freq
        )
        if memory_snapshot_freq <= 0:
            raise ValueError(
                "Memory snapshot frequency must be greater than zero; set "
                "profiler.memory_snapshot_freq or profiler.profile_freq to a "
                "positive value."
            )

        snapshot_dir = os.path.join(base_folder, cfg.save_memory_snapshot_folder)
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        rank = torch.distributed.get_rank()

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        return MemoryProfiler(
            global_step,
            memory_snapshot_freq,
            snapshot_dir,
            leaf_folder,
            rank,
            cfg.memory_snapshot_max_entries,
        )
