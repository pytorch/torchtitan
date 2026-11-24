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

Steps are cut on the GPU: each step() replays a one-kernel marker CUDA graph on
the current stream, and step N holds the kernels that started after marker N-1
and before marker N. A kernel on another stream whose record arrives after
marker N+1's is filed into a later step. Unwritten steps are lost if the
process is killed. On restart, files holding steps after the resumed step are
deleted.
"""

import json
import logging
import os
import re
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
_STEPS_FILE_RE = re.compile(r"steps_(\d{12})_(\d{12})\.npz")

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

    The host runs ahead of the GPU, so a step's end on the host says nothing
    about which kernels belong to it. Instead, step() replays a one-kernel
    marker CUDA graph on the current stream: the marker starts on the GPU after
    everything step N enqueued on that stream, so its GPU start time is step N's
    end. CUPTI delivers kernel records late and in batches; a step is finished
    once the next step's marker has been delivered.

    State:
    - self._steps_awaiting_marker: steps whose marker hasn't been delivered.
    - self._step_ends: (step, marker GPU start) of steps whose marker has been
      delivered but that aren't finished yet.
    - self._unassigned_spans: delivered kernels not yet in a finished step,
      sorted by start time.

    step(): replay the marker, flush CUPTI (no device sync), and move each
    finished step's kernels to self._files (see _close_finished_steps).

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
        from cupti.cupti import ActivityKind  # pyrefly: ignore[missing-import]
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
        # The marker graph ends each step on the GPU (see step()). It is captured
        # before the instantiate hook is registered, so it stays out of the node
        # tables, and it increments self._marker_buffer, which must outlive it.
        self._marker_buffer = torch.zeros(1, device="cuda")
        self._marker_graph, self._marker_graph_id = _capture_marker_graph(
            self._marker_buffer
        )
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
        self._init_step_state(
            global_step,
            _KernelSpanFiles(
                files_dir,
                global_step=global_step,
                rank=rank,
                num_steps_retained=config.num_steps_retained,
                steps_per_file=config.steps_per_file,
                graph_nodes=self._graph_nodes,
            ),
        )
        logger.info(
            f"Kernel span recording active. Keeping the last "
            f"{config.num_steps_retained} steps at {files_dir}"
        )

    def _init_step_state(self, global_step: int, files: "_KernelSpanFiles") -> None:
        """Set up the state that assigns kernels to steps; needs no GPU."""
        self._files = files
        # The last ended training step. step() increments it, so the first call
        # ends step global_step + 1.
        self._step = global_step
        # Steps whose marker kernel was launched but hasn't been delivered yet,
        # oldest first. Markers run in launch order, so the next delivered marker
        # ends the leftmost step.
        self._steps_awaiting_marker: deque[int] = deque()
        # (step, GPU start of its marker, unix ns) of each step whose marker has
        # been delivered but that isn't finished yet, oldest first.
        self._step_ends: list[tuple[int, int]] = []
        # Delivered kernels not yet assigned to a step, sorted by start time: they
        # belong to steps in _step_ends or to later steps.
        self._unassigned_spans = np.empty(0, dtype=_SPAN_DTYPE)

    def step(self) -> None:
        """End the current training step: replay the marker graph on the current
        stream, flush CUPTI without a device sync, and pass the kernels of every
        step that has finished to self._files (see _close_finished_steps)."""
        with sl.log_trace_span("kernel_span_recorder_step_call"):
            self._step += 1
            self._marker_graph.replay()
            self._steps_awaiting_marker.append(self._step)
            # TODO: _cuspy is private; flush through a public observer API once
            # torch has one.
            self._observer._cuspy.flush(sync=False)
            self._close_finished_steps(close_all=False)

    def close(self) -> None:
        """Stop recording and write everything recorded so far to disk.

        Removes the graph instantiate hook, flushes CUPTI with a device sync so
        every kernel that ran (including every marker) is delivered, assigns all
        delivered kernels to their steps, writes every unwritten step to a final
        .npz file, and unregisters the CUPTI observer. Kernels that started after
        the last marker are dropped.
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

        Step N's kernels are those that started on the GPU after marker N-1 and
        before marker N:

            delivered markers:  |marker 41           |marker 42           |marker 43
            unassigned kernels: k k k k k k k k k k k k k k k k k k k k k k ...
                                <----- step 42 ----->|<----- step 43 ---->|

        Delivery order across streams isn't guaranteed: a kernel on another
        stream that started before marker N can be delivered after it. So step N
        is finished only once marker N+1 has been delivered, a full step later.
        For each finished step, its kernels are the leading rows of
        self._unassigned_spans that started before its marker; those rows are
        removed from self._unassigned_spans and passed to self._files.add_step().

        Args:
            close_all: Treat every step whose marker was delivered as finished.
                Used at shutdown, after a fenced flush has delivered every
                kernel that ran.
        """
        self._drain_cupti_kernel_spans()
        num_finished = len(self._step_ends) - (0 if close_all else 1)
        for step, marker_start_ns in self._step_ends[:num_finished]:
            num_in_step = np.searchsorted(
                self._unassigned_spans["start_ns"], marker_start_ns
            )
            self._files.add_step(step, self._unassigned_spans[:num_in_step])
            self._unassigned_spans = self._unassigned_spans[num_in_step:]
        self._step_ends = self._step_ends[max(num_finished, 0) :]

    def _drain_cupti_kernel_spans(self) -> None:
        """Move the kernels CUPTI delivered since the last call from the observer
        into self._unassigned_spans, and the delivered markers into
        self._step_ends.

        Converts start and end times from CUPTI's record clock to unix
        nanoseconds. Marker kernels are recognized by their exec graph id; each
        one ends the oldest step in self._steps_awaiting_marker. The other
        kernels are appended to self._unassigned_spans, which is re-sorted by
        start time, because a delivery can contain a kernel (on another stream)
        that started before kernels already in self._unassigned_spans.
        """
        graph_node_id, start_ns, end_ns, stream_id = self._observer.drain()
        delivered = np.empty(len(start_ns), dtype=_SPAN_DTYPE)
        delivered["graph_node_id"] = graph_node_id
        delivered["start_ns"] = self._observer.convert_time_array(start_ns)
        delivered["end_ns"] = self._observer.convert_time_array(end_ns)
        delivered["stream_id"] = stream_id
        is_marker = (delivered["graph_node_id"] >> 32) == self._marker_graph_id
        for marker_start_ns in np.sort(delivered["start_ns"][is_marker]).tolist():
            self._step_ends.append(
                (self._steps_awaiting_marker.popleft(), marker_start_ns)
            )
        self._unassigned_spans = np.sort(
            np.concatenate([self._unassigned_spans, delivered[~is_marker]]),
            order="start_ns",
        )


def _capture_marker_graph(buffer: torch.Tensor) -> tuple[torch.cuda.CUDAGraph, int]:
    """Capture a CUDA graph of one tiny kernel that increments ``buffer``, and
    return it with its exec graph id, which CUPTI reports in the upper 32 bits
    of the kernel's graph node id."""
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        buffer.add_(1)
    graph.instantiate()
    return graph, graph.get_graph_data()["exec_graph_id"]


class _KernelSpanFiles:
    """Writes finished steps to .npz files in files_dir.

    Finished steps are held in host memory (self._unwritten_steps) until
    steps_per_file of them have accumulated, then written together as one file.
    After each write, files are deleted oldest first as long as the remaining
    files hold at least num_steps_retained steps.

    Files already in files_dir (from before a restart) are taken over: those
    holding any step after global_step belong to the abandoned timeline and are
    deleted, and the rest count toward num_steps_retained like files this
    instance writes.
    """

    def __init__(
        self,
        files_dir: str,
        *,
        global_step: int,
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
        # (path, number of steps it holds) of each step file on disk, oldest first.
        self._files_and_num_steps: deque[tuple[str, int]] = deque()
        self._num_steps_on_disk = 0
        self._take_over_existing_files(global_step)

    def _take_over_existing_files(self, global_step: int) -> None:
        """Track the step files a previous run left in self._dir.

        A file holding any step after global_step was written past the
        checkpoint this run resumed from, so its steps will be recorded again;
        delete it. The rest are tracked oldest first and evicted like files this
        instance writes.
        """
        existing = []
        for name in os.listdir(self._dir):
            match = _STEPS_FILE_RE.fullmatch(name)
            if match is None:
                continue
            first_step, last_step = int(match[1]), int(match[2])
            path = os.path.join(self._dir, name)
            if last_step > global_step:
                os.remove(path)
            else:
                existing.append((first_step, last_step, path))
        for first_step, last_step, path in sorted(existing):
            self._files_and_num_steps.append((path, last_step - first_step + 1))
            self._num_steps_on_disk += last_step - first_step + 1
        self._evict_old_files()

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
            self._files_and_num_steps
            and (
                num_steps_without_oldest := self._num_steps_on_disk
                - self._files_and_num_steps[0][1]
            )
            >= self._num_steps_retained
        ):
            oldest_path, _ = self._files_and_num_steps.popleft()
            os.remove(oldest_path)
            self._num_steps_on_disk = num_steps_without_oldest
