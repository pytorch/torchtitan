# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Always-on recording of every CUDA kernel's GPU start/end time, per training
step.

Per rank:

    CUPTI --> KernelSpanRecorder --> _KernelSpanFiles --> rank{r}/steps_<first>_<last>.npz
              assigns kernels         writes every          oldest files deleted beyond
              to steps                steps_per_file steps  num_steps_retained steps

File format: one .npz per file, holding one rank's S steps and K kernels.

    array          dtype/shape   meaning
    rank           int64         rank that wrote the file
    steps          int64[S]      step numbers
    step_offsets   int64[S+1]    rows of step i: step_offsets[i]:step_offsets[i+1]
    graph_node_id  uint64[K]     (exec graph id << 32) | node index; 0 = eager
    start_ns       int64[K]      GPU start, unix ns
    end_ns         int64[K]      GPU end, unix ns
    stream_id      uint64[K]     CUDA stream
    node_table     uint8[]       JSON {graph_node_id: {kernel_name, grid, block,
                                 annotations}} for the graphs these steps ran

The four [K] arrays are the columns of one kernel table, one row per kernel,
grouped by step. For example, a file with two steps and five kernels:

    steps        = [1001, 1002]
    step_offsets = [0, 3, 5]

    row  graph_node_id   start_ns  end_ns  stream_id
    0    (7 << 32) | 12  ...       ...     21          step 1001
    1    (7 << 32) | 13  ...       ...     21          step 1001
    2    0               ...       ...     7           step 1001 (eager kernel)
    3    (7 << 32) | 12  ...       ...     21          step 1002
    4    (7 << 32) | 13  ...       ...     21          step 1002

Graph node ids are only valid within one process, so each file carries its own
node table. scripts/kernel_spans_to_trace.py converts steps to a Perfetto trace.

Caveats: a step's end is the host time of its step() call, and the host runs
ahead of the GPU, so a step can include the tail of earlier steps. Unwritten
steps are lost if the process is killed.
"""

import json
import logging
import os
from collections import deque

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from torchtitan.config import Configurable
from torchtitan.observability import structured_logger as sl

logger = logging.getLogger(__name__)

KERNEL_SPANS_DIR = (
    "profiling/kernel_spans"  # KernelSpanRecorder.Config.save_folder default
)
KERNEL_SPANS_RANK_DIR = "rank{rank}"  # KERNEL_SPANS_DIR/{KERNEL_SPANS_RANK_DIR}
STEPS_FILE = "steps_{first_step:012d}_{last_step:012d}.npz"

# One row per kernel. Times are unix nanoseconds; graph_node_id is 0 for eager
# kernels.
_SPAN_DTYPE = np.dtype(
    [
        ("graph_node_id", "<u8"),
        ("start_ns", "<i8"),
        ("end_ns", "<i8"),
        ("stream_id", "<u8"),
    ]
)


class KernelSpanRecorder(Configurable):
    """Assigns each kernel CUPTI delivers to the training step it started in.

    CUPTI delivers kernel records late and in batches, so when step() is called
    (the step "ends" on the host), some of its kernels haven't been delivered
    yet. The step is "finished" once a kernel that started after its end has
    been delivered.

    State:
    - self._open_step_ends: (step, host end time) of ended, unfinished steps.
    - self._unassigned_spans: delivered kernels not yet in a finished step,
      sorted by start time.

    step(): append the step's end time, flush CUPTI (no device sync), and move
    each finished step's kernels to self._files (see _close_finished_steps).

    Kernels are named from CUDA graph nodes, so training must use CUDA graphs;
    kernels outside graphs are recorded without names.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        save_folder: str = KERNEL_SPANS_DIR
        """Step file location, relative to the run's output folder."""

        num_steps_retained: int = 1000
        """Number of steps to keep on disk per rank, rounded up to whole files. A step
        takes about 32 bytes per kernel, e.g. 1.6 MB for a step of 50k kernels, so the
        default keeps about 1.6 GB per rank at that size."""

        steps_per_file: int = 100
        """Number of finished steps held in host memory and then written as one
        file, on the training thread. Steps not yet written are lost if the process is killed.
        Writing every step costs up to ~0.5 ms/step at 50k-200k kernels/step; batching
        100 steps about halves that, and a 100-step file of ~5k-kernel steps takes
        ~20 ms to write."""

        def __post_init__(self) -> None:
            if self.num_steps_retained <= 0:
                raise ValueError(
                    "profiler.kernel_spans.num_steps_retained must be positive, got "
                    f"{self.num_steps_retained}."
                )
            if self.steps_per_file <= 0:
                raise ValueError(
                    "profiler.kernel_spans.steps_per_file must be positive, got "
                    f"{self.steps_per_file}."
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
        # Imported here: they need Cuspy and cupti-python, and profiler.py imports
        # this module for the config.
        # TODO: these are private torch APIs; move to a public per-step span
        # observer once torch exposes one.
        from cupti.cupti import ActivityKind
        from torch.profiler._cuspy.observers.node_timer import NodeTimerObserver

        from torchtitan.observability.profiler import (
            cuspy_unavailable_reasons,
            CuspyTraceProfiler,
        )

        if not torch.cuda.is_available():
            raise ValueError(
                "profiler.kernel_spans is set, but no CUDA device is available."
            )
        if reasons := cuspy_unavailable_reasons(CuspyTraceProfiler.Config()):
            raise ValueError(
                "profiler.kernel_spans is set, but Cuspy cannot run here:\n  - "
                + "\n  - ".join(reasons)
            )
        if not torch.cuda.graph_annotations.is_available():
            raise ValueError(
                "profiler.kernel_spans needs CUDA graph node ids, which are "
                "unavailable here; see torch.cuda.graph_annotations.is_available for "
                "the driver and cuda.bindings requirements."
            )
        self._observer = NodeTimerObserver([ActivityKind.CONCURRENT_KERNEL])
        if not self._observer.available:
            raise RuntimeError(
                "profiler.kernel_spans could not register a Cuspy observer; "
                "see the preceding 'Cuspy observer unavailable' warning."
            )
        # The last completed training step. step() increments it, so the first
        # call ends step global_step + 1.
        self._step = global_step
        # (step, time the step ended on the host, unix ns) of each ended step whose
        # kernels may not all have been delivered yet, oldest first.
        self._open_step_ends: list[tuple[int, int]] = []
        # Delivered kernels not yet assigned to a step, sorted by start time: they
        # belong to steps in _open_step_ends or to the step currently running.
        self._unassigned_spans = np.empty(0, dtype=_SPAN_DTYPE)
        # exec graph id -> {graph node id: node info}, recorded once per graph.
        self._graph_nodes: dict[int, dict[int, dict[str, Any]]] = {}
        self._instantiate_hook = torch.cuda.graphs.register_graph_instantiate_hook(
            self._record_graph_nodes
        )
        files_dir = os.path.join(
            base_folder,
            config.save_folder,
            leaf_folder,
            KERNEL_SPANS_RANK_DIR.format(rank=rank),
        )
        self._files = _KernelSpanFiles(
            files_dir,
            rank=rank,
            num_steps_retained=config.num_steps_retained,
            steps_per_file=config.steps_per_file,
            graph_nodes=self._graph_nodes,
        )
        logger.info(
            f"Kernel span recording active. Keeping the last "
            f"{config.num_steps_retained} steps at {files_dir}"
        )

    def step(self) -> None:
        """End the current training step: append its host end time to
        self._open_step_ends, flush CUPTI without a device sync, and pass the
        kernels of every step that has finished to self._files (see
        _close_finished_steps)."""
        with sl.log_trace_span("kernel_span_recorder_step_call"):
            self._step += 1
            self._open_step_ends.append(
                (
                    self._step,
                    self._observer.convert_time(self._observer.now_record_ns()),
                )
            )
            # TODO: _cuspy is private; flush through a public observer API once
            # torch has one.
            self._observer._cuspy.flush(sync=False)
            self._close_finished_steps(close_all=False)

    def close(self) -> None:
        """Stop recording and write everything recorded so far to disk.

        Removes the graph instantiate hook, flushes CUPTI with a device sync so
        every kernel that ran is delivered, assigns all delivered kernels to
        their steps, writes every unwritten step to a final .npz file, and
        unregisters the CUPTI observer. Kernels that started after the last
        step() call are dropped.
        """
        self._instantiate_hook.remove()
        # Fenced flush: blocks until every kernel that has run is delivered.
        self._observer._cuspy.flush(sync=True)
        self._close_finished_steps(close_all=True)
        self._files.write_remaining_steps()
        self._observer.close()

    def _record_graph_nodes(self, graph: torch.cuda.CUDAGraph) -> None:
        """Record the kernel name, launch geometry and annotations of every node of
        a just-instantiated CUDA graph.

        Runs as a graph instantiate hook, once per graph. By then the graph's
        annotations are keyed by the executable graph's node ids, which are the ids
        CUPTI reports for replayed kernels.
        """
        data = graph.get_graph_data()
        annotations = torch.cuda.graph_annotations.get_kernel_annotations()
        self._graph_nodes[data["exec_graph_id"]] = {
            node["tools_id"]: {
                "node_type": node["node_type"],
                "kernel_name": node["kernel_name"],
                "grid": node["grid_dim"],
                "block": node["block_dim"],
                "annotations": annotations.get(node["tools_id"]),
            }
            for node in data["nodes"]
        }

    def _close_finished_steps(self, *, close_all: bool) -> None:
        """Assign delivered kernels to the steps that have finished, and pass each
        finished step's kernels to self._files.add_step().

        Two things are compared. self._open_step_ends holds the host time at which
        each not-yet-finished step ended, recorded by step(). self._unassigned_spans
        holds the kernels CUPTI has delivered that don't belong to a finished step
        yet, sorted by GPU start time.

            open step ends:     |end 41              |end 42              |end 43
            unassigned kernels: k k k k k k k k k k k k k k k k k k k k k k ...
                                <----- step 42 ----->|<----- step 43 ---->|

        Step N's kernels are those whose start time is after step N-1's end and
        at or before step N's end. CUPTI delivers kernels in the order they ran,
        so once a kernel that started after step N's end has been delivered, no
        more kernels of step N can be delivered: step N is finished. For each finished
        step, its kernels are the leading rows of self._unassigned_spans up to its
        end time; those rows are removed from self._unassigned_spans and passed to
        self._files.add_step(). Kernels of unfinished steps stay in
        self._unassigned_spans for the next call.

        Args:
            close_all: Treat every step as finished. Used at shutdown, after a
                fenced flush has delivered every kernel that ran.
        """
        self._drain_cupti_kernel_spans()
        starts = self._unassigned_spans["start_ns"]
        latest_start_ns = starts[-1] if starts.size else -1
        complete_steps = [
            (step, step_end_ns)
            for step, step_end_ns in self._open_step_ends
            if close_all or step_end_ns <= latest_start_ns
        ]
        self._open_step_ends = self._open_step_ends[len(complete_steps) :]
        for step, step_end_ns in complete_steps:
            num_in_step = np.searchsorted(
                self._unassigned_spans["start_ns"], step_end_ns
            )
            self._files.add_step(step, self._unassigned_spans[:num_in_step])
            self._unassigned_spans = self._unassigned_spans[num_in_step:]

    def _drain_cupti_kernel_spans(self) -> None:
        """Move the kernels CUPTI delivered since the last call from the observer
        into self._unassigned_spans.

        Converts their start and end times from CUPTI's record clock to unix
        nanoseconds, appends them to self._unassigned_spans, and re-sorts it by
        start time. The re-sort is needed because a delivery can contain a kernel
        (on another stream) that started before kernels already in
        self._unassigned_spans.
        """
        graph_node_id, start_ns, end_ns, stream_id = self._observer.drain()
        delivered = np.empty(len(start_ns), dtype=_SPAN_DTYPE)
        delivered["graph_node_id"] = graph_node_id
        delivered["start_ns"] = self._observer.convert_time_array(start_ns)
        delivered["end_ns"] = self._observer.convert_time_array(end_ns)
        delivered["stream_id"] = stream_id
        self._unassigned_spans = np.sort(
            np.concatenate([self._unassigned_spans, delivered]), order="start_ns"
        )


class _KernelSpanFiles:
    """Writes finished steps to .npz files in files_dir.

    Finished steps are held in host memory (self._unwritten_steps) until
    steps_per_file of them have accumulated, then written together as one file.
    After each write, files are deleted oldest first as long as the remaining
    files hold at least num_steps_retained steps.

    Only files written by this instance are tracked and deleted; files already in
    files_dir (e.g. from before a restart) are left alone.
    """

    def __init__(
        self,
        files_dir: str,
        *,
        rank: int,
        num_steps_retained: int,
        steps_per_file: int,
        graph_nodes: dict[int, dict[int, dict[str, Any]]],
    ) -> None:
        os.makedirs(files_dir, exist_ok=True)
        self._dir = files_dir
        self._rank = rank
        self._num_steps_retained = num_steps_retained
        self._steps_per_file = steps_per_file
        # exec graph id -> {graph node id: node info}. Shared with
        # KernelSpanRecorder, whose graph instantiate hook fills it; read when
        # writing a file, to embed the tables of the graphs its kernels ran in.
        self._graph_nodes = graph_nodes
        # (step, its kernels as _SPAN_DTYPE rows) of complete steps not yet written.
        self._unwritten_steps: list[tuple[int, np.ndarray]] = []
        # (path, number of steps it holds) of each step file written, oldest first.
        self._files_and_num_steps: deque[tuple[str, int]] = deque()
        self._num_steps_on_disk = 0

    def add_step(self, step: int, spans: np.ndarray) -> None:
        """Append one finished step's kernels to self._unwritten_steps, and write
        them all to a file once it holds steps_per_file steps."""
        self._unwritten_steps.append((step, spans))
        if len(self._unwritten_steps) == self._steps_per_file:
            self._write_spans_file()

    def write_remaining_steps(self) -> None:
        """Write every step in self._unwritten_steps to one file, even if there
        are fewer than steps_per_file. Called at shutdown."""
        if self._unwritten_steps:
            self._write_spans_file()

    def _write_spans_file(self) -> None:
        """Write every step in self._unwritten_steps to one .npz file named after
        its first and last step, empty self._unwritten_steps, then delete old
        files (see _evict_old_files).

        The file embeds the node table of every CUDA graph its kernels ran in, so
        it can be decoded without the training process. See the module docstring
        for the arrays it holds.
        """
        steps, self._unwritten_steps = self._unwritten_steps, []
        path = os.path.join(
            self._dir,
            STEPS_FILE.format(first_step=steps[0][0], last_step=steps[-1][0]),
        )
        spans = np.concatenate([step_spans for _, step_spans in steps])
        # A graph node id is (exec graph id << 32) | node index; 0 for eager kernels.
        graphed = spans["graph_node_id"] != 0
        exec_graph_ids = np.unique(spans["graph_node_id"][graphed] >> 32)
        node_table = {
            str(node_id): node
            for graph_id in exec_graph_ids.tolist()
            for node_id, node in self._graph_nodes.get(graph_id, {}).items()
        }
        np.savez(
            path,
            rank=np.int64(self._rank),
            steps=np.array([step for step, _ in steps], dtype="<i8"),
            step_offsets=np.cumsum(
                [0] + [len(step_spans) for _, step_spans in steps], dtype="<i8"
            ),
            graph_node_id=spans["graph_node_id"],
            start_ns=spans["start_ns"],
            end_ns=spans["end_ns"],
            stream_id=spans["stream_id"],
            node_table=np.frombuffer(
                json.dumps(node_table, default=str).encode(), dtype=np.uint8
            ),
        )
        self._files_and_num_steps.append((path, len(steps)))
        self._num_steps_on_disk += len(steps)
        self._evict_old_files()

    def _evict_old_files(self) -> None:
        """Delete files oldest first, as long as the files left after each deletion
        still hold at least num_steps_retained steps."""
        while (
            num_steps_without_oldest := self._num_steps_on_disk
            - self._files_and_num_steps[0][1]
        ) >= self._num_steps_retained:
            oldest_path, _ = self._files_and_num_steps.popleft()
            os.remove(oldest_path)
            self._num_steps_on_disk = num_steps_without_oldest
