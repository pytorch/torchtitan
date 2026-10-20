# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Always-on GPU kernel timeline: every step's kernels, recent steps kept in memory."""

import gzip
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.cuda._graph_annotations as graph_annotations

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class AlwaysOnProfilerConfig:
    """Record every training step's GPU kernels with Cuspy, keep the most recent
    steps in host memory, and write them as a trace periodically."""

    num_steps: int = 100
    """Steps kept in memory; older steps are dropped."""

    dump_freq: int = 1000
    """Write the kept steps as a trace every this many steps; 0 never writes."""

    save_folder: str = "profiling/always_on"
    """Trace files location, under the output folder."""

    def __post_init__(self) -> None:
        if self.num_steps < 1:
            raise ValueError(
                f"always_on.num_steps must be at least 1, got {self.num_steps}"
            )
        if self.dump_freq < 0:
            raise ValueError(
                f"always_on.dump_freq must be non-negative, got {self.dump_freq}"
            )


class AlwaysOnProfiler:
    """Bucket the kernel spans Cuspy delivers into training steps and keep the last
    ``config.num_steps`` of them.

    Only kernel start, end, stream and CUDA graph node are recorded -- no CPU events --
    so it can run for the whole job. Kernel names come from each CUDA graph at
    instantiation, and annotations and lanes from torch's CUDA graph annotations, so
    eager kernels are written unnamed.
    """

    def __init__(
        self,
        config: AlwaysOnProfilerConfig,
        *,
        trace_dir: str,
        global_step: int,
    ) -> None:
        from torch.cuda.graphs import register_graph_instantiate_hook
        from torch.profiler._cuspy.observers.node_timer import NodeTimerObserver

        self._config = config
        self._trace_dir = trace_dir
        self._observer = NodeTimerObserver()
        if not self._observer.available:
            raise ValueError("profiler.always_on could not register with Cuspy.")
        self._kernel_names: dict[int, str] = {}
        self._instantiate_hook = register_graph_instantiate_hook(
            self._record_kernel_names
        )
        # _steps[i] ran from _boundaries[i] to _boundaries[i + 1], in CUPTI's record
        # clock; the last step is still running.
        self._boundaries: deque[int] = deque([self._observer.now_record_ns()])
        self._steps: deque[int] = deque([global_step + 1])
        # Per step: (graph node id, start ns, end ns, stream) column chunks.
        self._spans: dict[int, list[tuple[np.ndarray, ...]]] = {}

    def _record_kernel_names(self, graph: torch.cuda.CUDAGraph) -> None:
        for node in graph.get_graph_data()["nodes"]:
            if node["kernel_name"]:
                self._kernel_names[node["tools_id"]] = torch._C._demangle(
                    node["kernel_name"]
                )

    def step(self) -> None:
        """Close the running step; call once per training step, after it is issued."""
        step = self._steps[-1]
        self._boundaries.append(self._observer.now_record_ns())
        self._steps.append(step + 1)
        # A plain CUPTI flush hands over completed buffers without syncing the device;
        # kernels still running arrive with a later step's drain.
        node, start, end, stream = self._observer.drain(flush=True)
        boundaries = np.fromiter(self._boundaries, dtype=np.int64)
        index = np.searchsorted(boundaries, start, side="right") - 1
        for i in np.unique(index[index >= 0]).tolist():
            mask = index == i
            self._spans.setdefault(self._steps[i], []).append(
                (node[mask], start[mask], end[mask], stream[mask])
            )
        while len(self._steps) - 1 > self._config.num_steps:
            self._boundaries.popleft()
            self._spans.pop(self._steps.popleft(), None)
        if self._config.dump_freq and step % self._config.dump_freq == 0:
            self.dump(step)

    def dump(self, step: int) -> None:
        """Write the kept steps as a Chrome trace, except the newest finished step,
        whose kernels may still be running."""
        from torch.profiler._cuspy.observers import base

        steps = list(self._steps)[:-2]
        # Present on torch versions with CuspyConfig.graph_lanes.
        lane_resolver = getattr(base, "default_graph_lane_resolver", None)
        device = torch.cuda.current_device()
        events: list[dict[str, Any]] = []
        lane_names: dict[int, str] = {}
        # A graph node's annotation and lane are the same on every replay.
        node_info: dict[int, tuple[dict[str, Any], int | None]] = {}
        for s in steps:
            for node, start, end, stream in self._spans.get(s, []):
                start_us = self._observer.convert_time_array(start) / 1e3
                end_us = self._observer.convert_time_array(end) / 1e3
                for n, ts_us, te_us, st in zip(
                    node.tolist(), start_us.tolist(), end_us.tolist(), stream.tolist()
                ):
                    args: dict[str, Any] = {"step": s, "stream": st}
                    lane = st
                    if n:
                        if n not in node_info:
                            node_lane = None
                            if lane_resolver is not None and (
                                resolved := lane_resolver(n)
                            ):
                                node_lane, lane_names[node_lane] = resolved
                            node_info[n] = (
                                {
                                    "graph id": n >> 32,
                                    "graph node id": n & 0xFFFFFFFF,
                                    **(graph_annotations.annotation_for(n) or {}),
                                },
                                node_lane,
                            )
                        node_args, node_lane = node_info[n]
                        args.update(node_args)
                        if node_lane is not None:
                            lane = node_lane
                    events.append(
                        {
                            "ph": "X",
                            "cat": "kernel",
                            "name": self._kernel_names.get(n, "kernel"),
                            "pid": device,
                            "tid": lane,
                            "ts": ts_us,
                            "dur": te_us - ts_us,
                            "args": args,
                        }
                    )
        for lane, name in lane_names.items():
            events.append(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": device,
                    "tid": lane,
                    "args": {"name": name},
                }
            )
        step_dir = os.path.join(self._trace_dir, f"iteration_{step}")
        os.makedirs(step_dir, exist_ok=True)
        path = os.path.join(
            step_dir, f"rank{torch.distributed.get_rank()}_trace.json.gz"
        )
        with gzip.open(path, "wt", compresslevel=1) as f:
            json.dump({"traceEvents": events}, f)
        logger.info(
            f"Wrote always-on trace of steps {steps[0] if steps else '-'}-"
            f"{steps[-1] if steps else '-'} to {path}"
        )

    def close(self) -> None:
        self._instantiate_hook.remove()
        self._observer.close()
