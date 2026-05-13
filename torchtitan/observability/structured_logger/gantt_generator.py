# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Merge per-rank trace JSONL into a Chrome Trace JSON for Perfetto.

TODO: not tuned for thousands of ranks or long runs. If that starts
to hurt, cap record count with a CLI arg, lower peak memory, or
assign tids online in the formatter.
"""

import heapq
import json
import os
from collections import defaultdict
from glob import glob
from typing import Any

from torchtitan.observability.structured_logger.structured_logging import LogType
from torchtitan.tools.logging import logger


def generate_gantt_trace(log_dir: str, output_path: str) -> dict:
    """Merge per-rank JSONL into a Chrome Trace JSON for Perfetto.

    Open the result in https://ui.perfetto.dev or ``chrome://tracing``.

    The problem this solves: every async endpoint call runs in a fresh
    :class:`asyncio.Task` with an auto-generated name (``Task-1``,
    ``Task-3``, ...). A naive trace builder would give each task its
    own Perfetto track, so a 100-step run ends up with hundreds of
    parallel tracks per actor -- unreadable. This builder groups
    records by task so all spans that share a task (including nested
    ones) render on one track. Non-overlapping tasks reuse a track;
    truly concurrent tasks (e.g. via ``asyncio.gather``) get their own.

    Example::

        # Actor with two nested spans per endpoint call:
        @endpoint
        async def generate(self, prompts):
            with log_trace_span("rl_generate"):
                with log_trace_span("rl_engine_steps"):
                    return await self.engine.step(prompts)

        # After 5 sequential calls, the JSONL has 10 paired records
        # across 5 task names (Task-1, Task-3, ...). Build the trace:
        generate_gantt_trace("outputs/structured_logs/", "outputs/gantt.json")

        # Perfetto shows the generator as ONE track with 5 stacked
        # (rl_generate / rl_engine_steps) pairs along its timeline.
        # If two calls had run via asyncio.gather, they would instead
        # appear on two parallel tracks.

    Args:
        log_dir: Directory containing per-rank ``*.jsonl`` trace files.
        output_path: Path to write the merged Chrome Trace JSON.

    Returns:
        The Chrome Trace dict (``{"traceEvents": [...]}``). Also
        written to ``output_path``.
    """
    records = load_all_records(log_dir)
    if not records:
        logger.info(f"No records found in {log_dir}")
        return {"traceEvents": []}

    sources = sorted({r["_source_file"] for r in records})
    source_to_pid = {s: i for i, s in enumerate(sources)}

    paired, instants = _collect_paired_and_instants(records, source_to_pid)
    tid_by_source_and_task = _assign_tids_per_source(paired)
    events = _emit_chrome_events(
        paired=paired,
        instants=instants,
        source_to_pid=source_to_pid,
        tid_by_source_and_task=tid_by_source_and_task,
    )

    trace = {"traceEvents": events}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2)

    logger.info(f"Chrome Trace: {output_path}")
    logger.info(f"  {len(events)} events from {len(sources)} sources")
    logger.info("  View in: chrome://tracing or https://ui.perfetto.dev")
    return trace


def _assign_tids_per_source(
    paired: list[dict],
) -> dict[tuple[str, str | None], int]:
    """Assign a Perfetto ``tid`` to every paired span (mutates ``s["tid"]``).

    Task-level interval scheduling:

    1. Group spans by ``(source, task_name)``. Each asyncio task has
       one group; SPMD code groups under ``task_name=None``.
    2. Compute each group's time range ``[min(start_ts), max(end_ts)]``.
    3. Slot-pack the task ranges within each source via min-heap
       interval scheduling. Non-overlapping task ranges reuse a slot.
    4. Every span inherits its task's slot.

    Nested spans within a task share a tid -- Perfetto renders the
    nesting as a stacked flamegraph on one track. Concurrent tasks
    whose ranges overlap get different tids and render as parallel
    tracks. Sequential tasks reuse a single tid.

    Example (one source, four spans across three tasks)::

        span         start  end    task
        outer        100    500    Task-1
        inner        150    200    Task-1   # nested, same task
        weight_sync  600    800    Task-2   # after Task-1 ends
        log_shard    650    700    Task-3   # overlaps Task-2

        task ranges:  Task-1 [100, 500]
                      Task-2 [600, 800]
                      Task-3 [650, 700]

        slot packing (tasks sorted by start):
            Task-1 -> slot 0              # first, new slot
            Task-2 -> slot 0  (reused)    # 500 <= 600, slot 0 free
            Task-3 -> slot 1              # Task-2 still busy, new slot

        assigned tids:
            outer       -> 0
            inner       -> 0   (same task as outer -> Perfetto stacks)
            weight_sync -> 0
            log_shard   -> 1

    Args:
        paired: Paired span dicts from ``_collect_paired_and_instants``.
            Mutated in place: each span gets a ``"tid"`` key.

    Returns:
        ``(source, task_name) -> tid`` mapping. Used by
        ``_resolve_instant_tid`` so instants (which have no paired
        start) land on their enclosing task's track.
    """
    by_source: dict[str, list[dict]] = defaultdict(list)
    for p in paired:
        by_source[p["source"]].append(p)

    tid_by_source_and_task: dict[tuple[str, str | None], int] = {}

    for source, spans in by_source.items():
        # Compute each task's [min_start, max_end] range.
        task_range: dict[str | None, list[int]] = {}
        for s in spans:
            tn = s.get("task_name")
            ts, end = s["start_ts"], s["end_ts"]
            r = task_range.get(tn)
            if r is None:
                task_range[tn] = [ts, end]
            else:
                if ts < r[0]:
                    r[0] = ts
                if end > r[1]:
                    r[1] = end

        # Slot-pack task ranges (min-heap interval scheduling, sorted
        # by task start time).
        busy: list[tuple[int, int]] = []
        next_slot = 0
        for tn, (start, end) in sorted(task_range.items(), key=lambda x: x[1][0]):
            if busy and busy[0][0] <= start:
                _, slot = heapq.heappop(busy)
            else:
                slot = next_slot
                next_slot += 1
            heapq.heappush(busy, (end, slot))
            tid_by_source_and_task[(source, tn)] = slot

        # Every span inherits its task's tid.
        for s in spans:
            s["tid"] = tid_by_source_and_task[(source, s.get("task_name"))]

    return tid_by_source_and_task


def _resolve_instant_tid(
    *,
    source: str,
    task_name: str | None,
    tid_by_source_and_task: dict[tuple[str, str | None], int],
) -> int:
    """Pick the Perfetto tid for an instant event (no paired span).

    An instant lands on the same track as its enclosing task if that
    task has paired spans. Otherwise it falls back to tid 0 (the
    per-process main track).

    Args:
        source: The instant's source (one JSONL file = one source).
        task_name: The instant's task name, or ``None`` if emitted
            outside any asyncio task.
        tid_by_source_and_task: Map produced by ``_assign_tids_per_source``.

    Returns:
        The Perfetto ``tid`` for the instant.
    """
    if (source, task_name) in tid_by_source_and_task:
        return tid_by_source_and_task[(source, task_name)]
    return 0


def load_all_records(log_dir: str) -> list[dict]:
    """Load all JSONL records from a ``structured_logs/`` directory.

    Args:
        log_dir: Directory containing ``*.jsonl`` files (one per rank).

    Returns:
        All records across all files, each annotated with a
        ``"_source_file"`` key (the filename minus ``.jsonl``) so the
        caller can group records by process. Files are loaded in sorted
        name order; records within a file keep file order.
    """
    records = []
    for path in sorted(glob(os.path.join(log_dir, "*.jsonl"))):
        source = os.path.basename(path)
        # Strip the trailing .jsonl extension for a readable source label.
        source_name = source.rsplit(".", 1)[0] if "." in source else source
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["_source_file"] = source_name
                    records.append(r)
    return records


def _collect_paired_and_instants(
    records: list[dict], source_to_pid: dict[str, int]
) -> tuple[list[dict], list[dict]]:
    """Pair ``_start`` / ``_end`` via a per-(source, task_name) LIFO stack.

    Each asyncio task name gets its own stack, so nested spans within a
    task pair correctly regardless of what other tasks on the same
    process are doing. SPMD code has ``task_name=None`` throughout and
    pairs on a single stack per source.

    Instants include ``log_trace_instant`` markers, ``log_trace_scalar``
    metric values, and ``_error`` records.

    Example -- 5 records for one task, one source::

        # input (one JSONL line each, in order)
        {"log_type_name": "step_start",    "time_us": 100, "task_name": "Task-1"}
        {"log_type_name": "fwd_bwd_start", "time_us": 110, "task_name": "Task-1"}
        {"log_type_name": "fwd_bwd_end",   "time_us": 500, "task_name": "Task-1", "value": 0.39}
        {"log_type_name": "metric_value",  "time_us": 520, "event_name": "loss", "value": 2.5}
        {"log_type_name": "step_end",      "time_us": 600, "task_name": "Task-1", "value": 0.5}

        # output
        paired = [
            {"display_name": "fwd_bwd", "start_ts": 110, "end_ts": 500, ...},
            {"display_name": "step",    "start_ts": 100, "end_ts": 600, ...},
        ]
        instants = [
            {"name": "loss=2.5000", "time_us": 520, ...},
        ]

    ``fwd_bwd`` appears first in ``paired`` because LIFO pops the inner
    span (``fwd_bwd``) when ``fwd_bwd_end`` arrives; ``step`` pops later
    when ``step_end`` arrives.

    Args:
        records: All JSONL records across all sources, in the order
            produced by ``load_all_records``.
        source_to_pid: Map from ``_source_file`` name to the Perfetto
            pid assigned to that source.

    Returns:
        ``(paired, instants)`` -- paired spans in end-order (inner first,
        outer last) and instant events preserving record order.
    """
    paired: list[dict[str, Any]] = []
    instants: list[dict[str, Any]] = []
    # LIFO stack per (source, task_name). None task_name maps to one
    # stack per source (the SPMD path).
    pending: dict[tuple[str, str | None], list[dict]] = defaultdict(list)

    for r in records:
        event_type = r.get("log_type_name", "")
        log_type = r.get("log_type", "")
        time_us = r.get("time_us") or (r.get("time_ms") or 0) * 1000
        pid = source_to_pid[r["_source_file"]]
        rank = r.get("rank", 0)
        step = r.get("step")
        task_name = r.get("task_name")
        source = r["_source_file"]
        caller = r.get("caller")
        key = (source, task_name)

        # Point-in-time records (log_trace_instant, log_trace_scalar) carry
        # log_type=instant. Branch on intent before suffix matching so that
        # instants whose names happen to end in "_start" (binary_start,
        # training_start) don't get wrongly routed to the span-pair stack.
        if log_type == str(LogType.INSTANT):
            if event_type == "metric_value":
                event_name = r.get("event_name", "metric")
                value = r.get("value") or 0
                display_name = f"{event_name}={value:.4f}"
            else:
                display_name = event_type
            instants.append(
                {
                    "name": display_name,
                    "time_us": time_us,
                    "pid": pid,
                    "source": source,
                    "task_name": task_name,
                    "step": step,
                    "caller": caller,
                }
            )
        elif event_type.endswith("_start"):
            type_name = event_type.removesuffix("_start")
            display_name = r.get("event_name") or type_name
            pending[key].append(
                {
                    "ts": time_us,
                    "step": step,
                    "display_name": display_name,
                    "pid": pid,
                    "rank": rank,
                    "task_name": task_name,
                    "source": source,
                    "caller": caller,
                }
            )

        elif event_type.endswith("_end"):
            type_name = event_type.removesuffix("_end")
            duration_ms = r.get("value", 0)
            duration_us = (duration_ms or 0) * 1000
            stack = pending.get(key)
            start = stack.pop() if stack else None
            if start is not None:
                start_ts = start["ts"]
                end_ts = start_ts + duration_us
            else:
                # Orphan _end (e.g. truncated JSONL). Derive start from the
                # _end's own timestamp and the recorded duration so the span
                # renders in its real time window instead of after its end.
                end_ts = time_us
                start_ts = end_ts - duration_us
            paired.append(
                {
                    "pid": pid,
                    "rank": (start or {}).get("rank", rank),
                    "task_name": (start or {}).get("task_name", task_name),
                    "source": (start or {}).get("source", source),
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "display_name": (start or {}).get("display_name", type_name),
                    "step": step,
                    "duration_ms": duration_ms,
                    "caller": (start or {}).get("caller", caller),
                }
            )

        elif event_type.endswith("_error"):
            type_name = event_type.removesuffix("_error")
            instants.append(
                {
                    "name": f"ERROR: {type_name}",
                    "time_us": time_us,
                    "pid": pid,
                    "source": source,
                    "task_name": task_name,
                    "step": step,
                    "caller": caller,
                }
            )

        else:
            # Default: unrecognized record type renders as a bare instant on
            # the enclosing task's track.
            instants.append(
                {
                    "name": event_type,
                    "time_us": time_us,
                    "pid": pid,
                    "source": source,
                    "task_name": task_name,
                    "step": step,
                    "caller": caller,
                }
            )

    return paired, instants


def _emit_chrome_events(
    *,
    paired: list[dict],
    instants: list[dict],
    source_to_pid: dict[str, int],
    tid_by_source_and_task: dict[tuple[str, str | None], int],
) -> list[dict]:
    """Build the Chrome Trace event list (one Perfetto process per source).

    Emits three kinds of events:

    - ``"M"`` (metadata) ``process_name`` -- one per source, labels the
      process track in Perfetto.
    - ``"X"`` (complete) -- one per paired span, with ``ts`` / ``dur``
      for rendering as a bar.
    - ``"i"`` (instant) -- one per instant event, rendered as a
      vertical marker on its track.

    Args:
        paired: Spans from ``_collect_paired_and_instants`` with ``tid``
            already assigned by ``_assign_tids_per_source``.
        instants: Instant events from ``_collect_paired_and_instants``.
        source_to_pid: Map from source name to Perfetto pid.
        tid_by_source_and_task: Map used to place instants on their
            enclosing task's track.

    Returns:
        A flat list of Chrome Trace event dicts ready to serialize.
    """
    events: list[dict[str, Any]] = []

    for source, pid in source_to_pid.items():
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": source},
            }
        )

    for p in paired:
        duration_ms = p["duration_ms"] or 0
        args: dict = {
            **({"step": p["step"]} if p["step"] is not None else {}),
            "duration_ms": f"{duration_ms:.2f}" if duration_ms else "0.00",
        }
        if p.get("caller"):
            args["caller"] = p["caller"]
        events.append(
            {
                "name": p["display_name"],
                "ph": "X",
                "ts": p["start_ts"],
                "dur": p["end_ts"] - p["start_ts"],
                "pid": p["pid"],
                "tid": p.get("tid", 0),
                "args": args,
            }
        )

    for i in instants:
        tid = _resolve_instant_tid(
            source=i["source"],
            task_name=i["task_name"],
            tid_by_source_and_task=tid_by_source_and_task,
        )
        args = {**({"step": i["step"]} if i["step"] is not None else {})}
        if i.get("caller"):
            args["caller"] = i["caller"]
        events.append(
            {
                "name": i["name"],
                "ph": "i",
                "ts": i["time_us"],
                "pid": i["pid"],
                "tid": tid,
                "s": "t",
                "args": args,
            }
        )

    return events
