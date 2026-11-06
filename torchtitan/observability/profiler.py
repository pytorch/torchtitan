# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch profiler traces and memory snapshots, and their lifecycle."""

import inspect
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from torchtitan.config import Configurable
from torchtitan.distributed.accelerator_graph import (
    get_accelerator_graph_annotations,
    get_cuspy_cuda_graph_annotation_config,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.tools.utils import device_module, device_type

# torch's export_chrome_trace gained cuda_graph_annotations when the offline joiner
# (torch.cuda._annotate_cuda_graph_trace) was removed. Older versions still export, just
# without the CUDA graph annotations baked in.
logger = logging.getLogger(__name__)


_EXPORT_SUPPORTS_ANNOTATIONS = (
    "cuda_graph_annotations"
    in inspect.signature(torch.profiler.profile.export_chrome_trace).parameters
)

# Paths expects by meta internal tooling
PROFILE_DIR = "profiling/traces"  # trace profiler save_folder default
PROFILE_ITER_DIR = "iteration_{step}"  # PROFILE_DIR/{PROFILE_ITER_DIR}
PROFILE_FILE = "rank{rank}_trace.json.gz"  # PROFILE_DIR/PROFILE_ITER_DIR/{PROFILE_FILE}
PFTRACE_PROFILE_FILE = "rank{rank}_trace.pftrace.gz"
PY_STACKS_FILE = "rank{rank}_cuda_graph_py_stacks.json.gz"

MEMORY_DIR = (
    "profiling/memory_snapshot"  # MemorySnapshotProfiler.Config.save_folder default
)
MEMORY_STEP_DIR = "step_{step:012d}"  # MEMORY_DIR/{MEMORY_STEP_DIR}
MEMORY_EXIT_DIR = "step_{step:012d}_exit"  # OOM dump variant
MEMORY_FILE = (
    "{rank:06d}_step_{step}.pickle"  # MEMORY_DIR/MEMORY_STEP_DIR/{MEMORY_FILE}
)


def cuspy_unavailable_reasons(config: "CuspyTraceProfiler.Config") -> list[str]:
    """Return why Cuspy cannot run ``config`` in this environment; empty when it can.

    Probes without taking a CUPTI subscription: holding one makes Kineto's one-shot
    CUPTI init fail for the rest of the process.
    """
    if getattr(torch.profiler, "CuspyConfig", None) is None:
        return ["this torch build has no torch.profiler.CuspyConfig"]
    reasons = []
    if config.trace_format == "pftrace":
        from torch._C import _profiler

        # torch has no capability flag for the native encoder; builds without it
        # compile a stub that raises on any input, so encode an empty trace.
        try:
            _profiler._cuspy.encode_pftrace(0, [], [], [], [])
        except RuntimeError as error:
            reasons.append(f'trace_format="pftrace" is unsupported: {error}')
    # Private module, and importing it raises without the cupti-python package.
    try:
        from torch.profiler._cuspy import cupti_python
    except ModuleNotFoundError as error:
        reasons.append(f"{error}; install it with `pip install cupti-python`")
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


class TraceProfiler(Configurable, ABC):
    """Records ``torch.profiler.profile`` traces on a fixed schedule.

    Every ``freq`` steps it warms up for ``warmup`` steps, records ``active``
    steps, and writes one trace per rank to
    ``{save_folder}/iteration_{step}/{leaf_folder}/``. Subclasses choose the
    activities collected and how the trace is written.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        save_folder: str = PROFILE_DIR
        """Trace files location, relative to the run's output folder."""

        freq: int = 10
        """How often to collect profile traces, in iterations."""

        warmup: int = 3
        """The number of warmup steps before the active steps in each cycle."""

        active: int = 1
        """The number of steps recorded in each cycle."""

        repeat: int | None = None
        """The number of profiling cycles; ``None`` repeats until training ends."""

        skip_first: int | None = None
        """The number of initial steps to skip before the first cycle."""

        skip_first_wait: int | None = None
        """Whether to skip the wait phase of the first cycle."""

        def __post_init__(self) -> None:
            if self.freq < self.warmup + self.active:
                raise ValueError(
                    f"Trace profiler freq ({self.freq}) must be greater than or "
                    f"equal to warmup + active ({self.warmup} + {self.active})."
                )

        def maybe_cuda_graph_annotation_config(self) -> dict[str, Any] | None:
            """``torch.cuda.graph`` annotation options this trace profiler needs at
            capture, or ``None`` for the default."""
            return None

    def __init__(
        self,
        config: Config,
        *,
        global_step: int,
        base_folder: str,
        leaf_folder: str,
        rank: int,
    ) -> None:
        self._config = config
        self._rank = rank
        trace_dir = os.path.join(base_folder, config.save_folder)

        def trace_handler(prof):
            curr_trace_dir = os.path.join(
                trace_dir, PROFILE_ITER_DIR.format(step=prof.step_num), leaf_folder
            )
            os.makedirs(curr_trace_dir, exist_ok=True)
            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()
            self.export_trace(prof, curr_trace_dir)
            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

        activities = self.profile_activities()
        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")
        os.makedirs(trace_dir, exist_ok=True)

        additional_params = {
            key: val
            for key, val in [
                ("repeat", config.repeat),
                ("skip_first", config.skip_first),
                ("skip_first_wait", config.skip_first_wait),
            ]
            if val is not None
        }
        self._torch_profiler = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=config.freq - (config.active + config.warmup),
                warmup=config.warmup,
                active=config.active,
                **additional_params,
            ),
            on_trace_ready=trace_handler,
            record_shapes=True,
            # Neuron can link framework events to source lines through the stacks.
            with_stack=device_type == "neuron",
        )
        self._torch_profiler.__enter__()
        self._torch_profiler.step_num = global_step

    @abstractmethod
    def profile_activities(self) -> list[Any]:
        """The ``activities`` passed to ``torch.profiler.profile``."""

    @abstractmethod
    def export_trace(self, prof: torch.profiler.profile, trace_dir: str) -> None:
        """Write this rank's trace for the cycle that just finished to ``trace_dir``."""

    def step(self) -> None:
        with sl.log_trace_span("kineto_profiler_step_call"):
            self._torch_profiler.step()

    def close(self) -> None:
        self._torch_profiler.__exit__(None, None, None)


class KinetoTraceProfiler(TraceProfiler):
    """Records CPU and accelerator activity with Kineto."""

    @dataclass(kw_only=True, slots=True)
    class Config(TraceProfiler.Config):
        pass

    def profile_activities(self) -> list[Any]:
        activities: list[Any] = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        elif torch.xpu.is_available():
            activities.append(torch.profiler.ProfilerActivity.XPU)
        elif device_type == "neuron":
            activities.append(torch.profiler.ProfilerActivity.PrivateUse1)
        return activities

    def export_trace(self, prof: torch.profiler.profile, trace_dir: str) -> None:
        # CUDA graph annotations are baked in during the export rather than
        # joined onto the written file afterwards: re-reading and rewriting a
        # gzipped trace paid the compression cost twice.
        annotations = get_accelerator_graph_annotations()
        if annotations and not _EXPORT_SUPPORTS_ANNOTATIONS:
            logger.warning(
                "This torch does not support cuda_graph_annotations on "
                "export_chrome_trace; the trace will have no CUDA graph kernel "
                "annotations."
            )
            annotations = None
        output_file = os.path.join(trace_dir, PROFILE_FILE.format(rank=self._rank))
        if annotations:
            prof.export_chrome_trace(output_file, cuda_graph_annotations=annotations)
        else:
            prof.export_chrome_trace(output_file)


class CuspyTraceProfiler(TraceProfiler):
    """Records CPU activity with Kineto and CUDA activity with Cuspy, PyTorch's
    in-process CUPTI collector."""

    @dataclass(kw_only=True, slots=True)
    class Config(TraceProfiler.Config):
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

        record_py_stacks: bool = False
        """Record each CUDA graph kernel's Python launch stack at capture and save them
        beside the trace as ``rank{rank}_cuda_graph_py_stacks.json.gz``, keyed by
        ``(graph id << 32) | graph node id`` of the trace's kernel events (see
        ``torch.cuda.graph_annotations.get_kernel_py_stacks``)."""

        performance_metrics: list[str] = field(default_factory=list)
        """CUPTI PM-sampling metric names to collect, e.g. ``sm__cycles_active.avg``.
        See "PM sampling configuration" in torchtitan/observability/README.md for how
        to list the metrics available on the current GPU."""

        pm_sampling_interval_ms: float | None = None
        """PM-sampling interval; ``None`` uses torch's default."""

        pm_lookback_window_ms: float | None = None
        """PM sample buffer length. Sampling starts at profiler warmup, so keep warmup
        plus active steps within this window or the earliest samples are lost. ``None``
        uses torch's default."""

        def __post_init__(self) -> None:
            super(CuspyTraceProfiler.Config, self).__post_init__()
            if not self.performance_metrics and (
                self.pm_sampling_interval_ms is not None
                or self.pm_lookback_window_ms is not None
            ):
                raise ValueError(
                    "pm_sampling_interval_ms and pm_lookback_window_ms require "
                    "performance_metrics."
                )
            if (
                self.pm_sampling_interval_ms is not None
                and self.pm_sampling_interval_ms <= 0
            ):
                raise ValueError(
                    "pm_sampling_interval_ms must be positive, got "
                    f"{self.pm_sampling_interval_ms}."
                )
            if (
                self.pm_sampling_interval_ms is not None
                and self.pm_lookback_window_ms is not None
                and self.pm_lookback_window_ms < self.pm_sampling_interval_ms
            ):
                raise ValueError(
                    f"pm_lookback_window_ms ({self.pm_lookback_window_ms}) must be at least "
                    f"pm_sampling_interval_ms ({self.pm_sampling_interval_ms})."
                )

        def maybe_cuda_graph_annotation_config(self) -> dict[str, Any] | None:
            return get_cuspy_cuda_graph_annotation_config(
                enable_event_node_ids=self.enable_event_node_ids,
                record_py_stacks=self.record_py_stacks,
            )

    _config: Config  # pyrefly: ignore [bad-override]

    def profile_activities(self) -> list[Any]:
        config = self._config
        if not torch.cuda.is_available():
            raise ValueError(
                "CuspyTraceProfiler is configured, but no CUDA device is available."
            )
        if reasons := cuspy_unavailable_reasons(config):
            raise ValueError(
                "CuspyTraceProfiler is configured, but Cuspy cannot run here:\n  - "
                + "\n  - ".join(reasons)
            )
        if not torch.cuda.graph_annotations.is_available():
            logger.warning(
                "CUDA graph annotations are unavailable (see "
                "torch.cuda.graph_annotations.is_available for requirements); "
                "kernels replayed from CUDA graphs will not carry mark_kernels "
                "metadata in the Cuspy trace."
            )
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
        return [
            torch.profiler.ProfilerActivity.CPU,
            {
                torch.profiler.ProfilerActivity.CUDA: torch.profiler.ProfilerActivityConfig(
                    profiler_configs=profiler_configs
                )
            },
        ]

    def export_trace(self, prof: torch.profiler.profile, trace_dir: str) -> None:
        profile_file = (
            PFTRACE_PROFILE_FILE
            if self._config.trace_format == "pftrace"
            else PROFILE_FILE
        )
        # Cuspy picks its Perfetto encoder from the ".pftrace" in the path.
        prof.export_chrome_trace(
            os.path.join(trace_dir, profile_file.format(rank=self._rank))
        )
        if not self._config.record_py_stacks:
            return
        if torch.cuda.graph_annotations.get_kernel_py_stacks():
            torch.cuda.graph_annotations.dump_kernel_py_stacks(
                os.path.join(trace_dir, PY_STACKS_FILE.format(rank=self._rank))
            )
        else:
            logger.warning(
                "CuspyTraceProfiler.Config.record_py_stacks is set, but no CUDA graph "
                "launch stacks were recorded; stacks are only recorded for CUDA "
                "graphs, so check that CUDA graphs are enabled."
            )


class MemorySnapshotProfiler(Configurable):
    """Records allocator history and dumps a memory snapshot every ``freq`` steps,
    plus a final one if training dies of OOM."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        save_folder: str = MEMORY_DIR
        """Memory snapshot files location, relative to the run's output folder."""

        freq: int = 10
        """How often to dump memory snapshots, in iterations."""

        max_entries: int = 1_000_000
        """Max alloc/free events recorded per memory snapshot (ring buffer).

        Caps the history passed to ``_record_memory_history``; the oldest events
        are dropped once full. Bounds host memory and snapshot size / dump time.
        """

        def __post_init__(self) -> None:
            if self.freq <= 0:
                raise ValueError(
                    f"Memory snapshot freq must be greater than zero, got {self.freq}."
                )

    def __init__(
        self,
        config: Config,
        *,
        global_step: int,
        base_folder: str,
        leaf_folder: str,
        rank: int,
    ) -> None:
        self._freq = config.freq
        self._snapshot_dir = os.path.join(base_folder, config.save_folder)
        self._leaf_folder = leaf_folder
        self._rank = rank
        # when resume training, we start from the last step
        self._step_num = global_step
        os.makedirs(self._snapshot_dir, exist_ok=True)
        logger.info(
            f"Memory profiler active. Snapshot will be saved at {self._snapshot_dir}"
        )
        device_module.memory._record_memory_history(
            # stacks="python" records only Python frames (not C++), which is much
            # cheaper to capture and serialize, keeping the snapshot dump from
            # taking minutes -- the default stacks="all" symbolizes C++ frames,
            # which is very slow for torchtitan workload.
            stacks="python",
            max_entries=config.max_entries,
        )

    def step(self) -> None:
        with sl.log_trace_span("memory_profiler_step_call"):
            self._step_num += 1
            if self._step_num % self._freq == 0:
                self._dump(MEMORY_STEP_DIR.format(step=self._step_num), self._step_num)

    def close(self, exc: BaseException | None) -> None:
        """Stop; ``exc`` is the exception ending training, if any."""

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

        if caused_by_oom(exc):
            # dump as step_000000000000_exit if OOM at iter 1
            self._dump(MEMORY_EXIT_DIR.format(step=self._step_num), self._step_num)

    def _dump(self, dir_name: str, step: int) -> None:
        curr_snapshot_dir = os.path.join(
            self._snapshot_dir, dir_name, self._leaf_folder
        )
        os.makedirs(curr_snapshot_dir, exist_ok=True)
        logger.info(f"Dumping memory snapshot at step {step}")
        begin = time.monotonic()
        output_file = os.path.join(
            curr_snapshot_dir, MEMORY_FILE.format(rank=self._rank, step=step)
        )
        with open(output_file, "wb") as output:
            # Protocol 4 for compatibility with pytorch.org/memory_viz JS parser
            pickle.dump(device_module.memory._snapshot(), output, protocol=4)
        logger.info(
            f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
        )


class Profiler(Configurable):
    """The profiling entry point for a training run: owns every enabled profiling
    tool -- torch profiler traces (``trace``), memory snapshots
    (``memory_snapshot``), and other collected data -- and drives their
    lifecycle. Callers hold only this class: it builds the enabled tools on
    ``__enter__``, steps them after every training step, and closes them on
    ``__exit__``, passing along the exception that ended training.

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
        trace: TraceProfiler.Config | None = None
        """Trace profiler, ``KinetoTraceProfiler`` or ``CuspyTraceProfiler``;
        ``None`` disables traces."""

        memory_snapshot: MemorySnapshotProfiler.Config | None = None
        """Memory snapshots; ``None`` disables them."""

        def maybe_cuda_graph_annotation_config(self) -> dict[str, Any] | None:
            """``torch.cuda.graph`` annotation options the trace profiler needs at
            capture, or ``None`` for the default."""
            if self.trace is None:
                return None
            return self.trace.maybe_cuda_graph_annotation_config()

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
        self.trace: TraceProfiler | None = None
        self.memory_snapshot: MemorySnapshotProfiler | None = None

    def __enter__(self) -> "Profiler":
        config = self._config
        if config.trace is None and config.memory_snapshot is None:
            return self
        kwargs = {
            "global_step": self._global_step,
            "base_folder": self._base_folder,
            "leaf_folder": self._leaf_folder,
            "rank": torch.distributed.get_rank(),
        }
        if config.trace is not None:
            self.trace = config.trace.build(**kwargs)
        if config.memory_snapshot is not None:
            self.memory_snapshot = config.memory_snapshot.build(**kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.trace is not None:
            self.trace.close()
            self.trace = None
        if self.memory_snapshot is not None:
            self.memory_snapshot.close(exc_val)
            self.memory_snapshot = None
        return False

    def step(self) -> None:
        """Advance the trace profiler and memory snapshots by one training step."""
        if self.trace is not None:
            self.trace.step()
        if self.memory_snapshot is not None:
            self.memory_snapshot.step()
