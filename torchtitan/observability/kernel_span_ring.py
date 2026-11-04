# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Always-on recording of every CUDA kernel's GPU start and end time, kept per
training step in a rolling set of files on disk.

How data flows, per rank:

    CUPTI --flush--> unfiled kernels --cut at step ends--> finished steps
                                                                |
                                       every steps_per_file steps, write one file
                                                                |
                                         files on disk, oldest deleted first

- KernelSpanRecorder receives kernel records from CUPTI and works out which
  training step each kernel belongs to.
- _KernelSpanRing buffers finished steps, writes them to files, and deletes old
  files so that about num_steps_retained steps stay on disk.

Each file, named steps_<first step>_<last step>.npz, holds one row per kernel:
its CUDA graph node id (0 for kernels launched outside CUDA graphs), GPU start
and end time (unix nanoseconds) and CUDA stream. Kernels are grouped by step;
step_offsets marks where each step's rows begin. The file also stores a node
table, recorded when each CUDA graph was instantiated, that maps graph node ids
to kernel names and graph annotations (mark_kernels regions, c10d collective
metadata). Node ids are only meaningful within one process, so every file
carries its own table. scripts/kernel_span_ring_to_trace.py turns chosen steps
into a Chrome/Perfetto trace.

Caveats: steps are cut at the moment each step ends on the host. The host
usually runs ahead of the GPU, so a step can include the tail of earlier steps.
Steps not yet written to a file are lost if the process is killed.

Imported lazily by torchtitan.observability.profiler.Profiler, since it needs
Cuspy and the cupti-python package.
"""

import json
import os
from collections import deque
from typing import Any

import numpy as np
import torch

# TODO: these are private torch APIs; move to a public per-step span observer once
# torch exposes one.
from cupti.cupti import ActivityKind
from torch.profiler._cuspy.observers.node_timer import NodeTimerObserver

from torchtitan.observability.profiler import (
    cuspy_unavailable_reasons,
    CuspyProfilerConfig,
)

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


class KernelSpanRecorder(NodeTimerObserver):
    """Receives kernel records from CUPTI and files each kernel into the training
    step it ran in.

    CUPTI records every kernel the GPU runs, but hands the records over late and
    in batches that don't line up with steps: the host queues work ahead of the
    GPU, so when a step ends on the host, some of its kernels haven't run or
    haven't been reported yet.

    The recorder keeps two things in memory:

    - step ends: for each step that has ended but isn't finished yet, the time it
      ended on the host. step() adds one per training step.
    - unfiled kernels: kernels CUPTI has reported that aren't assigned to a step
      yet, sorted by start time.

    Each step() call records the step's end, flushes CUPTI (without a device
    sync) to collect newly reported kernels, and cuts the unfiled kernels at the
    step ends of every finished step (see _close_steps). Finished steps go to a
    _KernelSpanRing, which writes them to disk. Kernels are named from the CUDA
    graph nodes they ran as, so training must run in CUDA graphs; kernels
    launched outside graphs are recorded without names.

    Args:
        ring_dir: Directory for this rank's step files. Files already there (e.g.
            from before a restart) are left alone and never deleted.
        global_step: The last completed training step; the first :meth:`step`
            call ends step ``global_step + 1``.
        num_steps_retained: Number of steps to keep on disk, rounded up to whole
            files.
        steps_per_file: Number of complete steps written per file.
    """

    def __init__(
        self,
        ring_dir: str,
        *,
        global_step: int,
        num_steps_retained: int,
        steps_per_file: int,
    ) -> None:
        if not torch.cuda.is_available():
            raise ValueError(
                "profiler.kernel_span_ring is set, but no CUDA device is available."
            )
        if reasons := cuspy_unavailable_reasons(CuspyProfilerConfig()):
            raise ValueError(
                "profiler.kernel_span_ring is set, but Cuspy cannot run here:\n  - "
                + "\n  - ".join(reasons)
            )
        if not torch.cuda.graph_annotations.is_available():
            raise ValueError(
                "profiler.kernel_span_ring needs CUDA graph node ids, which are "
                "unavailable here; see torch.cuda.graph_annotations.is_available for "
                "the driver and cuda.bindings requirements."
            )
        super().__init__([ActivityKind.CONCURRENT_KERNEL])
        if not self.available:
            raise RuntimeError(
                "profiler.kernel_span_ring could not register a Cuspy observer; "
                "see the preceding 'Cuspy observer unavailable' warning."
            )
        self._step = global_step
        # (step, time the step ended on the host, unix ns) of each ended step whose
        # kernels may not all have arrived yet, oldest first.
        self._open_step_ends: list[tuple[int, int]] = []
        # Delivered kernels not yet filed into a step, sorted by start time: they
        # belong to open steps or to the step currently running.
        self._unfiled_spans = np.empty(0, dtype=_SPAN_DTYPE)
        # exec graph id -> {graph node id: node info}, recorded once per graph.
        self._graph_nodes: dict[int, dict[int, dict[str, Any]]] = {}
        self._instantiate_hook = torch.cuda.graphs.register_graph_instantiate_hook(
            self._record_graph_nodes
        )
        self._ring = _KernelSpanRing(
            ring_dir,
            num_steps_retained=num_steps_retained,
            steps_per_file=steps_per_file,
            graph_nodes=self._graph_nodes,
        )

    def step(self) -> None:
        """End the current training step: record when it ended on the host, flush
        CUPTI without a device sync, and file the delivered kernels into steps."""
        self._step += 1
        self._open_step_ends.append(
            (self._step, self.convert_time(self.now_record_ns()))
        )
        self._cuspy.flush(sync=False)
        self._close_steps(close_all=False)

    def close(self) -> None:
        """Write every ended step and stop collecting.

        Kernels that started after the last step ended are dropped.
        """
        self._instantiate_hook.remove()
        # Fenced flush: blocks until every kernel that has run is delivered.
        self._cuspy.flush(sync=True)
        self._close_steps(close_all=True)
        self._ring.write_remaining()
        super().close()

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

    def _close_steps(self, *, close_all: bool) -> None:
        """Cut the kernel timeline at each step's end and hand the finished steps to
        the ring.

        There are two timelines. The step ends, which step() records the moment each
        step ends on the host. And the kernels, which CUPTI reports late and in
        chunks; the ones received so far are kept sorted by start time.

            step ends:  |end 41              |end 42              |end 43
            kernels:    k k k k k k k k k k k k k k k k k k k k k k ...
                        <----- step 42 ----->|<----- step 43 ---->|

        A step's kernels are the ones that started between the previous step's end
        and its own. Kernels arrive in the order they ran, so a step is finished
        once some kernel that started after its end has arrived. Each finished
        step's kernels are then cut off the front of the list and handed to the
        ring; the rest wait for the next call.

        Args:
            close_all: Treat every step as finished. Used at shutdown, after a
                fenced flush has delivered every kernel that ran.
        """
        self._drain_spans()
        starts = self._unfiled_spans["start_ns"]
        latest_start_ns = starts[-1] if starts.size else -1
        complete_steps = [
            (step, step_end_ns)
            for step, step_end_ns in self._open_step_ends
            if close_all or step_end_ns <= latest_start_ns
        ]
        self._open_step_ends = self._open_step_ends[len(complete_steps) :]
        for step, step_end_ns in complete_steps:
            num_in_step = np.searchsorted(self._unfiled_spans["start_ns"], step_end_ns)
            self._ring.add_step(step, self._unfiled_spans[:num_in_step])
            self._unfiled_spans = self._unfiled_spans[num_in_step:]

    def _drain_spans(self) -> None:
        """Add the kernels the last flush delivered to the unfiled kernels, keeping
        them sorted by start time.

        Times are converted from CUPTI's record clock to unix nanoseconds. A flush
        can deliver a kernel that started before one already delivered on another
        stream, so the merged kernels are re-sorted.
        """
        graph_node_id, start_ns, end_ns, stream_id = self.drain()
        delivered = np.empty(len(start_ns), dtype=_SPAN_DTYPE)
        delivered["graph_node_id"] = graph_node_id
        delivered["start_ns"] = self.convert_time_array(start_ns)
        delivered["end_ns"] = self.convert_time_array(end_ns)
        delivered["stream_id"] = stream_id
        self._unfiled_spans = np.sort(
            np.concatenate([self._unfiled_spans, delivered]), order="start_ns"
        )


class _KernelSpanRing:
    """Buffers complete steps and writes them ``steps_per_file`` at a time to one
    file, deleting the oldest files once the newer ones still hold
    ``num_steps_retained`` steps.

    Only files written by this instance are tracked and deleted; files already in
    ``ring_dir`` (e.g. from before a restart) are left alone.
    """

    def __init__(
        self,
        ring_dir: str,
        *,
        num_steps_retained: int,
        steps_per_file: int,
        graph_nodes: dict[int, dict[int, dict[str, Any]]],
    ) -> None:
        os.makedirs(ring_dir, exist_ok=True)
        self._dir = ring_dir
        self._num_steps_retained = num_steps_retained
        self._steps_per_file = steps_per_file
        # Filled by the recorder; read here to embed node tables.
        self._graph_nodes = graph_nodes
        # (step, its kernels as _SPAN_DTYPE rows) of complete steps not yet written.
        self._unwritten_steps: list[tuple[int, np.ndarray]] = []
        # (path, number of steps it holds) of each step file written, oldest first.
        self._files_and_num_steps: deque[tuple[str, int]] = deque()
        self._num_steps_on_disk = 0

    def add_step(self, step: int, spans: np.ndarray) -> None:
        """Buffer one complete step's kernels, writing a file every
        ``steps_per_file`` steps."""
        self._unwritten_steps.append((step, spans))
        if len(self._unwritten_steps) == self._steps_per_file:
            self._write_file()

    def write_remaining(self) -> None:
        """Write the buffered steps, at shutdown."""
        if self._unwritten_steps:
            self._write_file()

    def _write_file(self) -> None:
        """Write the buffered steps to one file and delete the oldest files.

        The file embeds the node table of every CUDA graph its spans ran in, so it
        can be decoded without the training process.
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
        """Delete the oldest files while the newer ones still hold
        ``num_steps_retained`` steps."""
        while (
            num_steps_without_oldest := self._num_steps_on_disk
            - self._files_and_num_steps[0][1]
        ) >= self._num_steps_retained:
            oldest_path, _ = self._files_and_num_steps.popleft()
            os.remove(oldest_path)
            self._num_steps_on_disk = num_steps_without_oldest
