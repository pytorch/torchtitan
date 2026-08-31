#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compact CUDA-graph replay streams without changing measured GPU slices.

CUDA graph replay can assign a distinct stream to each captured region, which
makes a PyTorch profiler trace difficult to read. The preferred repair is to
merge a separately captured recording trace, which restores the original
stream identities. When that recording trace is unavailable, this utility
packs replay slices into semantic visualization lanes while preserving their
names, timestamps, durations, and arguments.

The output is for visualization only. Performance analysis must continue to
use the unmodified trace, although interval metrics are unchanged because the
compactor does not alter measured slices.

Example:
    python3 compact_cuda_graph_trace.py input.json.gz -o output.json.gz
    python3 compact_cuda_graph_trace.py trace_directory --merge-pp-ranks
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import heapq
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SLICE_CATEGORIES = {"kernel", "Kernel", "gpu_memcpy", "gpu_memset"}
METADATA_NAMES = {"thread_name", "thread_sort_index"}
GROUP_ORDER = (
    "compute",
    "symmetric_memory_barrier",
    "nccl_all_gather",
    "nccl_reduce_scatter",
    "nccl_all_reduce",
    "nccl_other",
    "memcpy",
    "memset",
)
GROUP_LABELS = {
    "compute": "Compute",
    "symmetric_memory_barrier": "Symmetric-memory barrier",
    "nccl_all_gather": "NCCL all-gather",
    "nccl_reduce_scatter": "NCCL reduce-scatter",
    "nccl_all_reduce": "NCCL all-reduce",
    "nccl_other": "NCCL other",
    "memcpy": "GPU memcpy",
    "memset": "GPU memset",
}
PP_ANNOTATION_PATTERN = re.compile(
    r"^PP:(?P<stage>\d+)(?P<operation>SEND_F|RECV_F|SEND_B|RECV_B|F|B)"
    r"(?P<microbatch>\d+)$"
)
PP_UNSHARD_ANNOTATION_PATTERN = re.compile(r"^PP:(?P<stage>\d+)UNSHARD$")


@dataclass(frozen=True)
class Slice:
    """A measured GPU slice and its position in ``traceEvents``."""

    event_index: int
    pid: int
    old_tid: int
    start: float
    end: float
    group: str


@dataclass(frozen=True)
class Assignment:
    """A slice assignment to a compacted visualization lane."""

    event_index: int
    pid: int
    old_tid: int
    start: float
    end: float
    group: str
    lane: int
    new_tid: int


@dataclass(frozen=True)
class RankedTrace:
    """A compacted trace and its pipeline-rank identity."""

    pp_rank: int
    global_rank: int
    trace: dict[str, Any]


@dataclass(frozen=True)
class PPAnnotationBlock:
    """One PP operation span on a GPU annotation track."""

    stage: int
    operation: str
    microbatch: int | None
    pid: int
    tid: int
    start: float
    end: float

    @property
    def label(self) -> str:
        """Return the concise PP operation label shown on flow arrows."""

        microbatch = "" if self.microbatch is None else self.microbatch
        return f"{self.stage}{self.operation}{microbatch}"


def _load_trace(path: Path) -> dict[str, Any]:
    """Load a JSON or gzip-compressed JSON trace.

    Args:
        path: Trace path ending in ``.json`` or ``.json.gz``.

    Returns:
        Parsed trace object.
    """

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _save_trace(trace: dict[str, Any], path: Path) -> None:
    """Write a JSON or gzip-compressed JSON trace.

    Args:
        trace: Parsed trace object.
        path: Destination ending in ``.json`` or ``.json.gz``.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(trace, handle, separators=(",", ":"))


def _classify_slice(event: dict[str, Any]) -> str:
    """Classify a measured GPU slice into a display group.

    Args:
        event: Chrome-trace event.

    Returns:
        Stable display-group identifier.
    """

    category = event.get("cat")
    name = event.get("name", "").lower()
    if category == "gpu_memcpy":
        return "memcpy"
    if category == "gpu_memset":
        return "memset"
    if "nccl" in name:
        if "allgather" in name:
            return "nccl_all_gather"
        if "reducescatter" in name:
            return "nccl_reduce_scatter"
        if "allreduce" in name:
            return "nccl_all_reduce"
        return "nccl_other"
    if "symmetric_memory" in name and "barrier" in name:
        return "symmetric_memory_barrier"
    return "compute"


def _extract_slices(events: list[dict[str, Any]]) -> list[Slice]:
    """Extract measured GPU slices that have a profiler stream.

    Args:
        events: Trace event list.

    Returns:
        Measured slices in trace-event order.

    Raises:
        ValueError: If a measured slice lacks required scheduling metadata.
    """

    slices = []
    for index, event in enumerate(events):
        if event.get("cat") not in SLICE_CATEGORIES:
            continue
        args = event.get("args", {})
        required = ("pid", "tid", "ts", "dur")
        if (
            any(event.get(field) is None for field in required)
            or args.get("stream") is None
        ):
            raise ValueError(f"GPU slice {index} lacks pid/tid/ts/dur/stream")
        if event["tid"] != args["stream"]:
            raise ValueError(
                f"GPU slice {index} has tid {event['tid']} != stream {args['stream']}"
            )
        start = float(event["ts"])
        duration = float(event["dur"])
        if duration < 0:
            raise ValueError(f"GPU slice {index} has negative duration {duration}")
        slices.append(
            Slice(
                event_index=index,
                pid=int(event["pid"]),
                old_tid=int(event["tid"]),
                start=start,
                end=start + duration,
                group=_classify_slice(event),
            )
        )
    if not slices:
        raise ValueError("trace contains no GPU kernel/memcpy/memset slices")
    return slices


def _pack_group(slices: Iterable[Slice]) -> dict[int, int]:
    """Assign non-overlapping display lanes with interval partitioning.

    Args:
        slices: Slices from one GPU process and semantic group.

    Returns:
        Mapping from trace-event index to zero-based lane number.
    """

    active: list[tuple[float, int]] = []
    free_lanes: list[int] = []
    assignments = {}
    lane_count = 0
    for item in sorted(slices, key=lambda value: (value.start, value.end)):
        while active and active[0][0] <= item.start:
            _, lane = heapq.heappop(active)
            heapq.heappush(free_lanes, lane)
        if free_lanes:
            lane = heapq.heappop(free_lanes)
        else:
            lane = lane_count
            lane_count += 1
        assignments[item.event_index] = lane
        heapq.heappush(active, (item.end, lane))
    return assignments


def _assign_lanes(
    events: list[dict[str, Any]], slices: list[Slice]
) -> list[Assignment]:
    """Create compact lane assignments for all measured slices.

    Args:
        events: Original trace events, used to avoid existing thread IDs.
        slices: Extracted measured GPU slices.

    Returns:
        One assignment per measured slice.
    """

    integer_tids = [
        event["tid"] for event in events if isinstance(event.get("tid"), int)
    ]
    next_tid = max(integer_tids, default=0) + 1000
    by_process_group: dict[tuple[int, str], list[Slice]] = defaultdict(list)
    for item in slices:
        by_process_group[(item.pid, item.group)].append(item)

    assignments = []
    for pid in sorted({item.pid for item in slices}):
        present_groups = {item.group for item in slices if item.pid == pid}
        ordered_groups = [group for group in GROUP_ORDER if group in present_groups]
        ordered_groups.extend(sorted(present_groups.difference(GROUP_ORDER)))
        for group in ordered_groups:
            group_slices = by_process_group[(pid, group)]
            packed = _pack_group(group_slices)
            lane_count = max(packed.values(), default=-1) + 1
            tids = [next_tid + lane for lane in range(lane_count)]
            next_tid += lane_count
            for item in group_slices:
                lane = packed[item.event_index]
                assignments.append(
                    Assignment(
                        event_index=item.event_index,
                        pid=pid,
                        old_tid=item.old_tid,
                        start=item.start,
                        end=item.end,
                        group=group,
                        lane=lane,
                        new_tid=tids[lane],
                    )
                )
    return sorted(assignments, key=lambda value: value.event_index)


def _nearest_lane(
    assignment_starts: dict[tuple[int, int], list[float]],
    assignment_values: dict[tuple[int, int], list[Assignment]],
    pid: int,
    tid: int,
    timestamp: float,
) -> int | None:
    """Find the lane of the nearest slice on an original GPU stream.

    Args:
        assignment_starts: Sorted slice starts by original process/thread.
        assignment_values: Assignments parallel to ``assignment_starts``.
        pid: Original event process ID.
        tid: Original event thread/stream ID.
        timestamp: Event timestamp.

    Returns:
        Compacted thread ID, or ``None`` when the stream has no slices.
    """

    key = (pid, tid)
    starts = assignment_starts.get(key)
    if not starts:
        return None
    values = assignment_values[key]
    position = bisect.bisect_left(starts, timestamp)
    candidates = range(max(0, position - 1), min(len(values), position + 2))
    best = min(
        candidates,
        key=lambda index: 0
        if values[index].start <= timestamp <= values[index].end
        else min(
            abs(timestamp - values[index].start),
            abs(timestamp - values[index].end),
        ),
    )
    return values[best].new_tid


def _annotation_lane_tids(
    events: list[dict[str, Any]], assignments: list[Assignment]
) -> dict[tuple[int, int], int]:
    """Allocate a dedicated annotation track above every compacted lane."""

    integer_tids = [
        event["tid"] for event in events if isinstance(event.get("tid"), int)
    ]
    integer_tids.extend(item.new_tid for item in assignments)
    next_tid = max(integer_tids, default=0) + 1000
    lanes = sorted({(item.pid, item.new_tid) for item in assignments})
    return {lane: next_tid + index for index, lane in enumerate(lanes)}


def _thread_metadata(
    pid: int, tid: int, name: str, sort_index: int
) -> list[dict[str, Any]]:
    """Build Perfetto name and ordering metadata for one thread."""

    return [
        {
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": tid,
            "args": {"name": name},
        },
        {
            "name": "thread_sort_index",
            "ph": "M",
            "pid": pid,
            "tid": tid,
            "args": {"sort_index": sort_index},
        },
    ]


def _lane_metadata(
    assignments: list[Assignment],
    annotation_tids: dict[tuple[int, int], int],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build Perfetto thread metadata for compacted lanes.

    Args:
        assignments: Compacted slice assignments.
        annotation_tids: Dedicated annotation track for each compacted lane.
        events: Remapped events used to identify populated annotation tracks.

    Returns:
        ``thread_name`` and ``thread_sort_index`` metadata events.
    """

    lanes = sorted(
        {(item.pid, item.new_tid, item.group, item.lane) for item in assignments},
        key=lambda value: (value[0], value[1]),
    )
    used_annotation_tids = {
        (event["pid"], event["tid"])
        for event in events
        if event.get("cat") == "gpu_user_annotation"
    }
    metadata = []
    sort_index = 0
    for pid, tid, group, lane in lanes:
        lane_suffix = (
            f" {lane + 1}"
            if sum(
                other.pid == pid and other.group == group and other.lane != lane
                for other in assignments
            )
            else ""
        )
        label = GROUP_LABELS.get(group, group.replace("_", " ").title()) + lane_suffix
        annotation_tid = annotation_tids[(pid, tid)]
        if (pid, annotation_tid) in used_annotation_tids:
            metadata.extend(
                _thread_metadata(
                    pid,
                    annotation_tid,
                    f"{label} annotations",
                    sort_index,
                )
            )
            sort_index += 1
        metadata.extend(_thread_metadata(pid, tid, label, sort_index))
        sort_index += 1
    return metadata


def _slice_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    """Return fields that must remain identical after compaction.

    Args:
        event: Measured GPU slice.

    Returns:
        Immutable signature excluding display placement and the converted
        per-kernel annotation.
    """

    args = dict(event.get("args", {}))
    args.pop("stream", None)
    return (
        event.get("name"),
        event.get("cat"),
        event.get("ph"),
        event.get("pid"),
        event.get("ts"),
        event.get("dur"),
        json.dumps(args, sort_keys=True, separators=(",", ":")),
    )


def _validate_compaction(
    original_events: list[dict[str, Any]],
    compacted_events: list[dict[str, Any]],
    assignments: list[Assignment],
) -> dict[str, Any]:
    """Verify slice identity and non-overlap of every compacted lane.

    Args:
        original_events: Unmodified trace events.
        compacted_events: Post-processed trace events.
        assignments: Expected lane assignments.

    Returns:
        Machine-readable verification summary.

    Raises:
        ValueError: If measured slices changed or compacted lanes overlap.
    """

    original_slices = [
        event for event in original_events if event.get("cat") in SLICE_CATEGORIES
    ]
    compacted_slices = [
        event for event in compacted_events if event.get("cat") in SLICE_CATEGORIES
    ]
    if len(original_slices) != len(compacted_slices):
        raise ValueError("measured GPU slice count changed")
    for index, (original, compacted) in enumerate(
        zip(original_slices, compacted_slices)
    ):
        if _slice_signature(original) != _slice_signature(compacted):
            raise ValueError(f"measured GPU slice {index} changed")

    intervals: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for item in assignments:
        intervals[(item.pid, item.new_tid)].append((item.start, item.end))
    for lane, values in intervals.items():
        previous_end = float("-inf")
        for start, end in sorted(values):
            if start < previous_end:
                raise ValueError(f"compacted lane {lane} contains overlapping slices")
            previous_end = max(previous_end, end)

    original_streams = {(item.pid, item.old_tid) for item in assignments}
    compacted_lanes = {(item.pid, item.new_tid) for item in assignments}
    groups = Counter(item.group for item in assignments)
    group_lanes = Counter((item.pid, item.group, item.new_tid) for item in assignments)
    return {
        "measured_slices": len(assignments),
        "original_streams": len(original_streams),
        "compacted_lanes": len(compacted_lanes),
        "slice_groups": dict(sorted(groups.items())),
        "lanes_by_group": dict(
            sorted(Counter(group for _, group, _ in group_lanes).items())
        ),
    }


def _copy_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy events and their mutable argument dictionaries.

    Args:
        events: Original trace events.

    Returns:
        Events safe to mutate without changing the input trace.
    """

    copied = [dict(event) for event in events]
    for event in copied:
        if isinstance(event.get("args"), dict):
            event["args"] = dict(event["args"])
    return copied


def _index_assignments(
    assignments: list[Assignment],
) -> tuple[
    dict[tuple[int, int], list[float]],
    dict[tuple[int, int], list[Assignment]],
    dict[tuple[int, int, float], int],
]:
    """Index assignments for point and nearest-slice lookup.

    Args:
        assignments: Compacted slice assignments.

    Returns:
        Sorted starts, assignments by old stream, and exact start mappings.
    """

    by_old_stream: dict[tuple[int, int], list[Assignment]] = defaultdict(list)
    exact_points = {}
    for item in assignments:
        by_old_stream[(item.pid, item.old_tid)].append(item)
        exact_points[(item.pid, item.old_tid, item.start)] = item.new_tid
    for values in by_old_stream.values():
        values.sort(key=lambda item: item.start)
    starts = {
        key: [item.start for item in values] for key, values in by_old_stream.items()
    }
    return starts, by_old_stream, exact_points


def _annotation_signature(event: dict[str, Any]) -> str:
    """Identify annotations that differ only in their measured interval.

    Args:
        event: GPU annotation event.

    Returns:
        Stable serialized identity excluding only timestamp and duration.
    """

    identity = {key: value for key, value in event.items() if key not in {"ts", "dur"}}
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _coalesce_gpu_annotations(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Merge duplicate annotation spans created by compacting replay streams.

    PyTorch emits one projected GPU annotation per original CUDA stream. Once
    those streams share a visualization lane, semantically identical spans can
    overlap heavily. Their union carries the same information without obscuring
    the measured GPU slices underneath.

    Args:
        events: Remapped trace events.

    Returns:
        Events with overlapping identical annotations merged, and the number of
        redundant events removed.
    """

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, event in enumerate(events):
        if (
            event.get("cat") == "gpu_user_annotation"
            and event.get("ph") == "X"
            and isinstance(event.get("ts"), (int, float))
            and isinstance(event.get("dur"), (int, float))
            and event["dur"] >= 0
        ):
            grouped[_annotation_signature(event)].append((index, event))

    replacements: dict[int, dict[str, Any]] = {}
    skipped_indices = set()
    removed = 0

    def merge_cluster(cluster: list[tuple[int, dict[str, Any]]]) -> None:
        nonlocal removed
        if len(cluster) == 1:
            return
        replacement_index = min(index for index, _ in cluster)
        start = min(float(event["ts"]) for _, event in cluster)
        end = max(float(event["ts"]) + float(event["dur"]) for _, event in cluster)
        replacement = dict(events[replacement_index])
        replacement["ts"] = start
        replacement["dur"] = end - start
        replacements[replacement_index] = replacement
        skipped_indices.update(
            index for index, _ in cluster if index != replacement_index
        )
        removed += len(cluster) - 1

    for values in grouped.values():
        ordered = sorted(
            values,
            key=lambda value: (
                float(value[1]["ts"]),
                float(value[1]["ts"]) + float(value[1]["dur"]),
                value[0],
            ),
        )
        cluster = [ordered[0]]
        cluster_end = float(ordered[0][1]["ts"]) + float(ordered[0][1]["dur"])
        for value in ordered[1:]:
            event = value[1]
            start = float(event["ts"])
            end = start + float(event["dur"])
            if start <= cluster_end:
                cluster.append(value)
                cluster_end = max(cluster_end, end)
            else:
                merge_cluster(cluster)
                cluster = [value]
                cluster_end = end
        merge_cluster(cluster)

    retained = [
        replacements.get(index, event)
        for index, event in enumerate(events)
        if index not in skipped_indices
    ]
    return retained, removed


def _kernel_annotation_name(event: dict[str, Any]) -> str | None:
    """Return a per-kernel CUDA graph annotation."""

    args = event.get("args")
    if not isinstance(args, dict):
        return None
    name = args.get("name")
    if not isinstance(name, str) or not name:
        return None
    return name


def _kernel_annotation_event(
    name: str,
    pid: int,
    tid: int,
    start: float,
    end: float,
) -> dict[str, Any]:
    """Create a GPU annotation spanning neighboring annotated kernels."""

    return {
        "name": name,
        "cat": "gpu_user_annotation",
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": start,
        "dur": end - start,
        "args": {},
    }


def _convert_kernel_annotations(
    events: list[dict[str, Any]],
    annotation_tids: dict[tuple[int, int], int],
) -> tuple[list[dict[str, Any]], int, int]:
    """Replace per-kernel labels with lane-local annotation spans."""

    by_lane: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("cat") in SLICE_CATEGORIES:
            by_lane[(int(event["pid"]), int(event["tid"]))].append(event)

    annotations = []
    converted = 0
    for (pid, tid), lane_events in sorted(by_lane.items()):
        current_name = None
        current_start = current_end = 0.0
        ordered = sorted(lane_events, key=lambda event: (event["ts"], event["dur"]))
        for event in ordered:
            name = _kernel_annotation_name(event)
            start = float(event["ts"])
            end = start + float(event["dur"])
            if name != current_name:
                if current_name is not None:
                    annotations.append(
                        _kernel_annotation_event(
                            current_name,
                            pid,
                            annotation_tids[(pid, tid)],
                            current_start,
                            current_end,
                        )
                    )
                current_name = name
                current_start = start
            if name is not None:
                converted += 1
                current_end = end
        if current_name is not None:
            annotations.append(
                _kernel_annotation_event(
                    current_name,
                    pid,
                    annotation_tids[(pid, tid)],
                    current_start,
                    current_end,
                )
            )
    return events + annotations, converted, len(annotations)


def _remap_related_events(
    events: list[dict[str, Any]],
    assignment_starts: dict[tuple[int, int], list[float]],
    by_old_stream: dict[tuple[int, int], list[Assignment]],
    exact_points: dict[tuple[int, int, float], int],
    annotation_tids: dict[tuple[int, int], int],
) -> tuple[list[dict[str, Any]], int, int]:
    """Remap GPU flows and annotations, and replace old stream metadata.

    Args:
        events: Events whose measured slices have already been remapped.
        assignment_starts: Sorted slice starts by old process and stream.
        by_old_stream: Assignments by old process and stream.
        exact_points: New lanes keyed by old process, stream, and slice start.
        annotation_tids: Dedicated annotation track for each compacted lane.

    Returns:
        Retained events, mapped flow count, and mapped annotation count.

    Raises:
        ValueError: If any GPU flow endpoint cannot be mapped exactly.
    """

    old_streams = set(by_old_stream)
    flow_events = flow_events_mapped = annotation_events_mapped = 0
    retained = []
    for event in events:
        pid = event.get("pid")
        tid = event.get("tid")
        if not isinstance(pid, int) or not isinstance(tid, int):
            retained.append(event)
            continue
        original_key = (pid, tid)
        is_old_metadata = (
            event.get("ph") == "M"
            and event.get("name") in METADATA_NAMES
            and original_key in old_streams
        )
        if is_old_metadata:
            continue
        if event.get("cat") == "ac2g" and original_key in old_streams:
            flow_events += 1
            mapped = exact_points.get((pid, tid, float(event.get("ts", 0))))
            if mapped is not None:
                event["tid"] = mapped
                flow_events_mapped += 1
        elif event.get("cat") == "gpu_user_annotation" and original_key in old_streams:
            mapped = _nearest_lane(
                assignment_starts,
                by_old_stream,
                pid,
                tid,
                float(event.get("ts", 0)),
            )
            if mapped is not None:
                event["tid"] = annotation_tids[(pid, mapped)]
                annotation_events_mapped += 1
        retained.append(event)
    if flow_events_mapped != flow_events:
        raise ValueError(
            f"mapped {flow_events_mapped}/{flow_events} GPU ac2g flow endpoints"
        )
    return retained, flow_events_mapped, annotation_events_mapped


def compact_trace(
    trace: dict[str, Any],
    *,
    include_local_pp_flows: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compact CUDA graph streams in a parsed PyTorch profiler trace.

    Args:
        trace: Parsed profiler trace.
        include_local_pp_flows: Add same-rank PP dependency arrows.

    Returns:
        Pair of compacted trace and verification summary.

    Raises:
        ValueError: If the trace schema or compaction invariants are invalid.
    """

    original_events = trace.get("traceEvents")
    if not isinstance(original_events, list):
        raise ValueError("traceEvents must be a list")
    events = _copy_events(original_events)
    slices = _extract_slices(original_events)
    assignments = _assign_lanes(original_events, slices)
    assignment_starts, by_old_stream, exact_points = _index_assignments(assignments)
    annotation_tids = _annotation_lane_tids(original_events, assignments)
    for item in assignments:
        events[item.event_index]["tid"] = item.new_tid
        events[item.event_index]["args"]["stream"] = item.new_tid
    retained, flow_events_mapped, annotation_events_mapped = _remap_related_events(
        events,
        assignment_starts,
        by_old_stream,
        exact_points,
        annotation_tids,
    )
    retained, annotation_events_coalesced = _coalesce_gpu_annotations(retained)
    (
        retained,
        kernel_annotations_converted,
        kernel_annotation_blocks,
    ) = _convert_kernel_annotations(retained, annotation_tids)
    retained.extend(_lane_metadata(assignments, annotation_tids, retained))
    compacted = dict(trace)
    compacted["traceEvents"] = retained
    compacted["cuda_graph_stream_compaction"] = {
        "algorithm": "semantic interval partitioning",
        "measurement_policy": (
            "visualization only; measured GPU slices unchanged; overlapping "
            "identical GPU annotations coalesced; per-kernel CUDA graph "
            "annotations converted to lane-local spans"
        ),
        "source": (
            ".claude/skills/cuda_graph_trace_compaction/scripts/"
            "compact_cuda_graph_trace.py"
        ),
    }
    summary = _validate_compaction(original_events, retained, assignments)
    summary["gpu_flow_endpoints_mapped"] = flow_events_mapped
    summary["gpu_annotations_mapped"] = annotation_events_mapped
    summary["gpu_annotations_coalesced"] = annotation_events_coalesced
    summary["kernel_annotations_converted"] = kernel_annotations_converted
    summary["kernel_annotation_blocks"] = kernel_annotation_blocks
    summary["gpu_annotations_retained"] = sum(
        event.get("cat") == "gpu_user_annotation" for event in retained
    )
    summary["cuda_graph_launches"] = sum(
        event.get("name") == "cudaGraphLaunch" for event in retained
    )
    if include_local_pp_flows:
        _add_single_rank_pp_flows(compacted, summary)
    return compacted, summary


def _global_rank(trace: dict[str, Any]) -> int:
    """Read the distributed rank recorded in a Kineto trace."""

    distributed_info = trace.get("distributedInfo")
    if not isinstance(distributed_info, dict):
        raise ValueError("trace is missing distributedInfo")
    rank = distributed_info.get("rank")
    if not isinstance(rank, int):
        raise ValueError("trace distributedInfo is missing an integer rank")
    return rank


def _pp_rank_mapping(traces: list[dict[str, Any]]) -> dict[int, int]:
    """Map global ranks to pipeline ranks using the recorded PP group."""

    global_ranks = [_global_rank(trace) for trace in traces]
    if len(set(global_ranks)) != len(global_ranks):
        raise ValueError("PP traces contain duplicate distributed ranks")
    for trace in traces:
        distributed_info = trace["distributedInfo"]
        for group in distributed_info.get("pg_config", []):
            if (
                group.get("pg_name") != "torchtitan_real_pp"
                and group.get("pg_desc") != "mesh_pp"
            ):
                continue
            ranks = group.get("ranks")
            if isinstance(ranks, list) and all(rank in ranks for rank in global_ranks):
                return {rank: ranks.index(rank) for rank in global_ranks}
    return {rank: index for index, rank in enumerate(sorted(global_ranks))}


def _aligned_trace_copy(
    trace: dict[str, Any], common_base_ns: int
) -> tuple[dict[str, Any], float]:
    """Align one trace to a shared Kineto absolute-time origin."""

    base_ns = trace.get("baseTimeNanoseconds")
    if not isinstance(base_ns, int):
        raise ValueError("PP trace is missing integer baseTimeNanoseconds")
    offset_us = (base_ns - common_base_ns) / 1000.0
    aligned = dict(trace)
    events = _copy_events(trace["traceEvents"])
    for event in events:
        timestamp = event.get("ts")
        if isinstance(timestamp, (int, float)):
            event["ts"] = timestamp + offset_us
    aligned["traceEvents"] = events
    aligned["baseTimeNanoseconds"] = common_base_ns
    return aligned, offset_us


def _ranked_compacted_traces(
    traces: list[dict[str, Any]],
) -> tuple[list[RankedTrace], int, dict[int, float], dict[int, dict[str, Any]]]:
    """Compact and align traces, then attach their PP-rank identities."""

    if len(traces) < 2:
        raise ValueError("merging PP traces requires at least two trace files")
    bases = []
    for trace in traces:
        base = trace.get("baseTimeNanoseconds")
        if not isinstance(base, int):
            raise ValueError("all PP traces must contain integer baseTimeNanoseconds")
        bases.append(base)
    common_base_ns = min(bases)
    pp_ranks = _pp_rank_mapping(traces)
    offsets = {}
    summaries = {}
    ranked = []
    for trace in traces:
        global_rank = _global_rank(trace)
        compacted, summaries[global_rank] = compact_trace(
            trace, include_local_pp_flows=False
        )
        aligned, offsets[global_rank] = _aligned_trace_copy(compacted, common_base_ns)
        ranked.append(RankedTrace(pp_ranks[global_rank], global_rank, aligned))
    return (
        sorted(ranked, key=lambda item: item.pp_rank),
        common_base_ns,
        offsets,
        summaries,
    )


def _ranked_events(
    ranked: RankedTrace, first_pid: int
) -> tuple[list[dict[str, Any]], int, dict[int, int]]:
    """Give one trace unique process IDs and PP-rank-prefixed track names."""

    events = _copy_events(ranked.trace["traceEvents"])
    pids = {event["pid"] for event in events if isinstance(event.get("pid"), int)}
    gpu_pids = {
        event["pid"]
        for event in events
        if event.get("cat") in SLICE_CATEGORIES and isinstance(event.get("pid"), int)
    }
    ordered_pids = sorted(pids, key=lambda pid: (pid in gpu_pids, pid))
    pid_map = {pid: first_pid + index for index, pid in enumerate(ordered_pids)}
    flow_ids = {}
    retained = []
    for event in events:
        old_pid = event.get("pid")
        if isinstance(old_pid, int):
            event["pid"] = pid_map[old_pid]
        if event.get("ph") in {"s", "t", "f"} and "id" in event:
            source_id = json.dumps(event["id"], sort_keys=True, separators=(",", ":"))
            if source_id not in flow_ids:
                flow_ids[source_id] = (ranked.pp_rank + 1) * 1_000_000_000 + len(
                    flow_ids
                )
            event["id"] = flow_ids[source_id]
        if event.get("ph") == "M" and event.get("name") == "process_sort_index":
            continue
        if event.get("ph") == "M" and event.get("name") in {
            "process_name",
            "thread_name",
        }:
            name = event.get("args", {}).get("name")
            if isinstance(name, str):
                event["args"]["name"] = f"PP rank {ranked.pp_rank} | {name}"
        retained.append(event)
    pid_to_pp_rank = {new_pid: ranked.pp_rank for new_pid in pid_map.values()}
    retained.extend(_rank_process_metadata(ranked, ordered_pids, pid_map, gpu_pids))
    return retained, first_pid + len(pid_map), pid_to_pp_rank


def _rank_process_metadata(
    ranked: RankedTrace,
    ordered_pids: list[int],
    pid_map: dict[int, int],
    gpu_pids: set[int],
) -> list[dict[str, Any]]:
    """Order every process by PP rank and identify otherwise unnamed processes."""

    metadata = []
    for index, old_pid in enumerate(ordered_pids):
        pid = pid_map[old_pid]
        role = "GPU" if old_pid in gpu_pids else f"process {old_pid}"
        metadata.extend(
            (
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "args": {
                        "name": (
                            f"PP rank {ranked.pp_rank} "
                            f"(global rank {ranked.global_rank}) | {role}"
                        )
                    },
                },
                {
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": pid,
                    "tid": 0,
                    "args": {"sort_index": ranked.pp_rank * 1000 + index},
                },
            )
        )
    return metadata


def _pp_annotation_key(event: dict[str, Any]) -> tuple[int, str, int] | None:
    """Parse a PP GPU annotation's stage, operation, and microbatch."""

    if event.get("cat") != "gpu_user_annotation" or event.get("ph") != "X":
        return None
    name = event.get("name")
    match = PP_ANNOTATION_PATTERN.fullmatch(name) if isinstance(name, str) else None
    required = (event.get("pid"), event.get("tid"), event.get("ts"), event.get("dur"))
    if match is None or not all(isinstance(value, (int, float)) for value in required):
        return None
    return (
        int(match.group("stage")),
        match.group("operation"),
        int(match.group("microbatch")),
    )


def _named_annotation_lanes(
    events: list[dict[str, Any]],
    lane_names: set[str],
) -> set[tuple[int, int]]:
    """Return annotation tracks with one of the requested display names."""

    lanes = set()
    for event in events:
        if event.get("ph") != "M" or event.get("name") != "thread_name":
            continue
        pid = event.get("pid")
        tid = event.get("tid")
        name = event.get("args", {}).get("name")
        if not (
            isinstance(pid, int) and isinstance(tid, int) and isinstance(name, str)
        ):
            continue
        lane_name = name.rsplit(" | ", maxsplit=1)[-1]
        if lane_name in lane_names:
            lanes.add((pid, tid))
    return lanes


def _main_compute_annotation_lanes(
    events: list[dict[str, Any]],
) -> set[tuple[int, int]]:
    """Return the primary compute annotation track for each GPU process."""

    return _named_annotation_lanes(
        events, {"Compute annotations", "Compute 1 annotations"}
    )


def _unshard_annotation_lanes(
    events: list[dict[str, Any]],
) -> set[tuple[int, int]]:
    """Return the first two all-gather annotation tracks for each process."""

    return _named_annotation_lanes(
        events,
        {
            "NCCL all-gather annotations",
            "NCCL all-gather 1 annotations",
            "NCCL all-gather 2 annotations",
        },
    )


def _annotation_block(
    key: tuple[int, str, int], events: list[dict[str, Any]]
) -> PPAnnotationBlock:
    """Build one PP block from overlapping annotations on one lane."""

    representative = min(events, key=lambda event: (event["ts"], event["tid"]))
    stage, operation, microbatch = key
    return PPAnnotationBlock(
        stage=stage,
        operation=operation,
        microbatch=microbatch,
        pid=int(representative["pid"]),
        tid=int(representative["tid"]),
        start=min(float(event["ts"]) for event in events),
        end=max(float(event["ts"]) + float(event["dur"]) for event in events),
    )


def _compute_annotation_blocks(
    key: tuple[int, str, int],
    events: list[dict[str, Any]],
    main_compute_lanes: set[tuple[int, int]],
) -> list[PPAnnotationBlock]:
    """Group parallel spans and represent them on the primary compute lane."""

    ordered = sorted(events, key=lambda event: (event["ts"], event["dur"]))
    clusters = []
    cluster = [ordered[0]]
    cluster_end = float(ordered[0]["ts"]) + float(ordered[0]["dur"])

    def append_cluster(values: list[dict[str, Any]]) -> None:
        main_compute_values = [
            event
            for event in values
            if (int(event["pid"]), int(event["tid"])) in main_compute_lanes
        ]
        if main_compute_values:
            clusters.append(_annotation_block(key, main_compute_values))

    for event in ordered[1:]:
        start = float(event["ts"])
        end = start + float(event["dur"])
        if start <= cluster_end:
            cluster.append(event)
            cluster_end = max(cluster_end, end)
        else:
            append_cluster(cluster)
            cluster = [event]
            cluster_end = end
    append_cluster(cluster)
    return clusters


def _pp_annotation_blocks(events: list[dict[str, Any]]) -> list[PPAnnotationBlock]:
    """Extract logical PP blocks from lane-local GPU annotations."""

    main_compute_lanes = _main_compute_annotation_lanes(events)
    grouped: dict[tuple[int, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        key = _pp_annotation_key(event)
        if key is not None:
            grouped[(int(event["pid"]), *key)].append(event)
    blocks = []
    for (_, stage, operation, microbatch), values in sorted(grouped.items()):
        key = (stage, operation, microbatch)
        if operation in {"F", "B"}:
            blocks.extend(_compute_annotation_blocks(key, values, main_compute_lanes))
        else:
            blocks.extend(_annotation_block(key, [event]) for event in values)
    return blocks


def _unshard_annotation_blocks(
    events: list[dict[str, Any]],
) -> list[PPAnnotationBlock]:
    """Extract PP unshard blocks from the first two all-gather lanes."""

    unshard_lanes = _unshard_annotation_lanes(events)
    blocks = []
    for event in events:
        pid = event.get("pid")
        tid = event.get("tid")
        name = event.get("name")
        match = (
            PP_UNSHARD_ANNOTATION_PATTERN.fullmatch(name)
            if isinstance(name, str)
            else None
        )
        required = (pid, tid, event.get("ts"), event.get("dur"))
        if (
            event.get("cat") != "gpu_user_annotation"
            or event.get("ph") != "X"
            or match is None
            or not all(isinstance(value, (int, float)) for value in required)
            or (int(pid), int(tid)) not in unshard_lanes
        ):
            continue
        start = float(event["ts"])
        blocks.append(
            PPAnnotationBlock(
                stage=int(match.group("stage")),
                operation="UNSHARD",
                microbatch=None,
                pid=int(pid),
                tid=int(tid),
                start=start,
                end=start + float(event["dur"]),
            )
        )
    return sorted(blocks, key=lambda block: (block.pid, block.start, block.stage))


def _pair_nearest_blocks(
    sources: list[PPAnnotationBlock], destinations: list[PPAnnotationBlock]
) -> list[tuple[PPAnnotationBlock, PPAnnotationBlock]]:
    """Pair repeated PP blocks by nearest midpoint."""

    candidates = sorted(
        (
            abs((source.start + source.end) - (destination.start + destination.end)),
            source_index,
            destination_index,
        )
        for source_index, source in enumerate(sources)
        for destination_index, destination in enumerate(destinations)
    )
    used_sources = set()
    used_destinations = set()
    pairs = []
    for _, source_index, destination_index in candidates:
        if source_index in used_sources or destination_index in used_destinations:
            continue
        used_sources.add(source_index)
        used_destinations.add(destination_index)
        pairs.append((sources[source_index], destinations[destination_index]))
    return pairs


def _annotation_flow_pair(
    source: PPAnnotationBlock,
    destination: PPAnnotationBlock,
    pid_to_pp_rank: dict[int, int],
    category: str,
    flow_id: int,
    *,
    strictly_forward: bool = False,
) -> list[dict[str, Any]]:
    """Build a flow whose endpoints bind to PP GPU annotation blocks."""

    destination_timestamp = max(source.start, destination.start)
    if strictly_forward and destination_timestamp == source.start:
        destination_timestamp = math.nextafter(source.start, math.inf)
    if destination_timestamp > destination.end:
        return []
    name = f"{source.label} -> {destination.label}"
    args = {
        "source_pp_rank": pid_to_pp_rank[source.pid],
        "destination_pp_rank": pid_to_pp_rank[destination.pid],
        "source_stage": source.stage,
        "destination_stage": destination.stage,
        "source_operation": source.operation,
        "destination_operation": destination.operation,
        "microbatch": (
            source.microbatch
            if source.microbatch is not None
            else destination.microbatch
        ),
    }
    return [
        {
            "name": name,
            "cat": category,
            "ph": "s",
            "bp": "e",
            "pid": source.pid,
            "tid": source.tid,
            "ts": source.start,
            "id": flow_id,
            "args": args,
        },
        {
            "name": name,
            "cat": category,
            "ph": "f",
            "bp": "e",
            "pid": destination.pid,
            "tid": destination.tid,
            "ts": destination_timestamp,
            "id": flow_id,
            "args": args,
        },
    ]


def _send_recv_flow_events(
    blocks: list[PPAnnotationBlock],
    pid_to_pp_rank: dict[int, int],
    first_flow_id: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Connect matching send and receive annotation blocks across PP ranks."""

    grouped: dict[tuple[int, str, int], list[PPAnnotationBlock]] = defaultdict(list)
    for block in blocks:
        grouped[(block.stage, block.operation, block.microbatch)].append(block)
    flows = []
    matched_recv_keys = set()
    unmatched = 0
    next_flow_id = first_flow_id
    for (stage, operation, microbatch), sends in sorted(grouped.items()):
        if operation not in {"SEND_F", "SEND_B"}:
            continue
        peer_stage = stage + 1 if operation == "SEND_F" else stage - 1
        recv_key = (peer_stage, f"RECV_{operation[-1]}", microbatch)
        recvs = grouped.get(recv_key, [])
        matched_recv_keys.add(recv_key)
        pairs = _pair_nearest_blocks(sends, recvs)
        unmatched += len(sends) + len(recvs) - 2 * len(pairs)
        for send, recv in pairs:
            pair_flow = _annotation_flow_pair(
                send,
                recv,
                pid_to_pp_rank,
                "pp_send_recv",
                next_flow_id,
                strictly_forward=True,
            )
            flows.extend(pair_flow)
            unmatched += 2 if not pair_flow else 0
            next_flow_id += 1
    unmatched += sum(
        len(values)
        for key, values in grouped.items()
        if key[1].startswith("RECV_") and key not in matched_recv_keys
    )
    return flows, len(flows) // 2, unmatched, next_flow_id


def _local_dependency_flow_events(
    blocks: list[PPAnnotationBlock],
    pid_to_pp_rank: dict[int, int],
    first_flow_id: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Connect receive, compute, and send annotation blocks within each stage."""

    grouped: dict[tuple[int, int, str, int], list[PPAnnotationBlock]] = defaultdict(
        list
    )
    for block in blocks:
        grouped[(block.pid, block.stage, block.operation, block.microbatch)].append(
            block
        )
    flows = []
    unmatched = 0
    next_flow_id = first_flow_id
    for (pid, stage, operation, microbatch), computes in sorted(grouped.items()):
        if operation not in {"F", "B"}:
            continue
        recv = grouped.get((pid, stage, f"RECV_{operation}", microbatch), [])
        send = grouped.get((pid, stage, f"SEND_{operation}", microbatch), [])
        for sources, destinations in ((recv, computes), (computes, send)):
            if not sources or not destinations:
                continue
            pairs = _pair_nearest_blocks(sources, destinations)
            unmatched += len(sources) + len(destinations) - 2 * len(pairs)
            for source, destination in pairs:
                pair_flow = _annotation_flow_pair(
                    source,
                    destination,
                    pid_to_pp_rank,
                    "pp_compute_dependency",
                    next_flow_id,
                )
                flows.extend(pair_flow)
                unmatched += 2 if not pair_flow else 0
                next_flow_id += 1
    return flows, len(flows) // 2, unmatched, next_flow_id


def _unshard_dependency_flow_events(
    unshards: list[PPAnnotationBlock],
    blocks: list[PPAnnotationBlock],
    pid_to_pp_rank: dict[int, int],
    first_flow_id: int,
) -> tuple[list[dict[str, Any]], int, int]:
    """Connect each unshard to the first same-stage compute that follows it."""

    computes: dict[tuple[int, int], list[PPAnnotationBlock]] = defaultdict(list)
    for block in blocks:
        if block.operation in {"F", "B"}:
            computes[(block.pid, block.stage)].append(block)
    for values in computes.values():
        values.sort(key=lambda block: (block.start, block.end))

    flows = []
    unmatched = 0
    next_flow_id = first_flow_id
    for unshard in unshards:
        destination = next(
            (
                block
                for block in computes.get((unshard.pid, unshard.stage), [])
                if block.start >= unshard.start
            ),
            None,
        )
        if destination is None:
            unmatched += 1
            continue
        pair_flow = _annotation_flow_pair(
            unshard,
            destination,
            pid_to_pp_rank,
            "pp_unshard_dependency",
            next_flow_id,
        )
        flows.extend(pair_flow)
        if not pair_flow:
            unmatched += 1
        next_flow_id += 1
    return flows, len(flows) // 2, unmatched


def _rank_local_flow_events(
    blocks: list[PPAnnotationBlock],
    unshards: list[PPAnnotationBlock],
    pid_to_pp_rank: dict[int, int],
    first_flow_id: int,
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    """Build compute and unshard dependency flows within one PP rank."""

    local, local_pairs, local_unmatched, next_flow_id = _local_dependency_flow_events(
        blocks, pid_to_pp_rank, first_flow_id
    )
    unshard, unshard_pairs, unshard_unmatched = _unshard_dependency_flow_events(
        unshards, blocks, pid_to_pp_rank, next_flow_id
    )
    return (
        local + unshard,
        local_pairs,
        local_unmatched,
        unshard_pairs,
        unshard_unmatched,
    )


def _single_trace_pp_rank(trace: dict[str, Any]) -> int:
    """Return the trace's PP rank, or zero when rank metadata is absent."""

    try:
        global_rank = _global_rank(trace)
        return _pp_rank_mapping([trace])[global_rank]
    except ValueError:
        return 0


def _add_single_rank_pp_flows(trace: dict[str, Any], summary: dict[str, Any]) -> None:
    """Add rank-local PP dependency arrows to a compacted trace."""

    events = trace["traceEvents"]
    pp_rank = _single_trace_pp_rank(trace)
    pid_to_pp_rank = {
        int(event["pid"]): pp_rank
        for event in events
        if isinstance(event.get("pid"), int)
    }
    blocks = _pp_annotation_blocks(events)
    unshards = _unshard_annotation_blocks(events)
    (
        flows,
        local_pairs,
        local_unmatched,
        unshard_pairs,
        unshard_unmatched,
    ) = _rank_local_flow_events(blocks, unshards, pid_to_pp_rank, 1_000_000_000_000)
    events.extend(flows)
    flow_summary = {
        "pp_rank": pp_rank,
        "pp_compute_dependency_pairs": local_pairs,
        "pp_compute_dependency_unmatched": local_unmatched,
        "pp_unshard_compute_pairs": unshard_pairs,
        "pp_unshard_compute_unmatched": unshard_unmatched,
    }
    trace["cuda_graph_pp_local_flows"] = flow_summary
    summary.update(flow_summary)


def _communication_flow_events(
    events: list[dict[str, Any]], pid_to_pp_rank: dict[int, int]
) -> tuple[list[dict[str, Any]], int, int, int, int, int, int]:
    """Build cross-rank and rank-local PP annotation flows."""

    blocks = _pp_annotation_blocks(events)
    unshards = _unshard_annotation_blocks(events)
    (
        send_recv,
        send_recv_pairs,
        send_recv_unmatched,
        next_flow_id,
    ) = _send_recv_flow_events(blocks, pid_to_pp_rank, 1_000_000_000_000)
    (
        local,
        local_pairs,
        local_unmatched,
        unshard_pairs,
        unshard_unmatched,
    ) = _rank_local_flow_events(blocks, unshards, pid_to_pp_rank, next_flow_id)
    return (
        send_recv + local,
        send_recv_pairs,
        send_recv_unmatched,
        local_pairs,
        local_unmatched,
        unshard_pairs,
        unshard_unmatched,
    )


def merge_pp_traces(
    traces: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compact and merge PP-rank traces into one aligned Perfetto timeline."""

    ranked, common_base_ns, offsets, compaction = _ranked_compacted_traces(traces)
    events = []
    pid_to_pp_rank = {}
    next_pid = 1
    for item in ranked:
        rank_events, next_pid, rank_pid_map = _ranked_events(item, next_pid)
        events.extend(rank_events)
        pid_to_pp_rank.update(rank_pid_map)
    (
        flows,
        flow_pairs,
        unmatched,
        local_pairs,
        local_unmatched,
        unshard_pairs,
        unshard_unmatched,
    ) = _communication_flow_events(events, pid_to_pp_rank)
    events.extend(flows)
    merged = dict(ranked[0].trace)
    merged["traceEvents"] = events
    merged["baseTimeNanoseconds"] = common_base_ns
    merged["traceName"] = "Merged pipeline-parallel CUDA graph trace"
    merged["distributedInfo"] = {
        "backend": ranked[0].trace.get("distributedInfo", {}).get("backend"),
        "ranks": [item.global_rank for item in ranked],
        "pp_rank_count": len(ranked),
    }
    summary = {
        "time_alignment": "baseTimeNanoseconds",
        "time_offsets_us": offsets,
        "pp_ranks": [
            {"pp_rank": item.pp_rank, "global_rank": item.global_rank}
            for item in ranked
        ],
        "pp_send_recv_pairs": flow_pairs,
        "pp_send_recv_unmatched": unmatched,
        "pp_compute_dependency_pairs": local_pairs,
        "pp_compute_dependency_unmatched": local_unmatched,
        "pp_unshard_compute_pairs": unshard_pairs,
        "pp_unshard_compute_unmatched": unshard_unmatched,
        "per_rank_compaction": compaction,
    }
    merged["cuda_graph_pp_merge"] = summary
    return merged, summary


def _default_output(input_path: Path) -> Path:
    """Derive a compacted trace path from an input path.

    Args:
        input_path: Source trace path.

    Returns:
        Sibling path ending in ``_compacted.json[.gz]``.
    """

    name = input_path.name
    if name.endswith(".json.gz"):
        name = name[: -len(".json.gz")] + "_compacted.json.gz"
    elif name.endswith(".json"):
        name = name[: -len(".json")] + "_compacted.json"
    else:
        name += "_compacted.json.gz"
    return input_path.with_name(name)


def _rank_trace_paths(input_path: Path) -> list[Path]:
    """Find unprocessed rank traces in a directory or beside one trace."""

    directory = input_path if input_path.is_dir() else input_path.parent
    paths = set(directory.glob("rank*_trace.json.gz"))
    paths.update(directory.glob("rank*_trace.json"))
    rank_pattern = re.compile(r"rank(?P<rank>\d+)_trace\.json(?:\.gz)?$")

    def rank(path: Path) -> int:
        match = rank_pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid PP trace filename: {path.name}")
        return int(match.group("rank"))

    ranked_paths = sorted(paths, key=rank)
    if len(ranked_paths) < 2:
        raise ValueError(f"found fewer than two rank traces in {directory}")
    return ranked_paths


def _default_merged_output(input_path: Path) -> Path:
    """Return the default merged output beside the input rank traces."""

    directory = input_path if input_path.is_dir() else input_path.parent
    return directory / "pp_traces_merged_compacted.json.gz"


def main() -> None:
    """Run CUDA graph stream compaction and print verification results."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="PyTorch trace, or a rank-trace directory with --merge-pp-ranks",
    )
    parser.add_argument("-o", "--output", type=Path, help="Compacted trace path")
    parser.add_argument(
        "--merge-pp-ranks",
        action="store_true",
        help="Compact and merge sibling rank*_trace.json[.gz] PP traces",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output"
    )
    args = parser.parse_args()

    output = args.output or (
        _default_merged_output(args.input)
        if args.merge_pp_ranks
        else _default_output(args.input)
    )
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists: {output}; pass --overwrite to replace it")
    if args.merge_pp_ranks:
        try:
            input_paths = _rank_trace_paths(args.input)
            compacted, summary = merge_pp_traces(
                [_load_trace(path) for path in input_paths]
            )
        except ValueError as error:
            parser.error(str(error))
        inputs: str | list[str] = [str(path) for path in input_paths]
    else:
        if args.input.is_dir():
            parser.error("input must be a trace file without --merge-pp-ranks")
        compacted, summary = compact_trace(_load_trace(args.input))
        inputs = str(args.input)
    _save_trace(compacted, output)
    print(json.dumps({"input": inputs, "output": str(output), **summary}, indent=2))


if __name__ == "__main__":
    main()
