# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Merge per-rank trace JSONL into a Chrome Trace JSON for Perfetto.

TODO: not tuned for thousands of ranks, or for long runs rendered without
``last_steps``.
"""

import heapq
import json

import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from glob import glob
from typing import Any, BinaryIO

from torchtitan.observability.structured_logger.structured_logging import LogType


logger = logging.getLogger(__name__)

# asyncio names a task "Task-<n>" unless its creator passes `name=`.
_UNNAMED_TASK_RE = re.compile(r"^Task-\d+$")
# `TraceJsonlHandler` names each file "{source}.global_rank_{rank}.{date}-{time}-{suffix}.jsonl".
_FILE_NAME_RE = re.compile(
    r"^(?P<logger_source>.+)\.global_rank_(?P<rank>\d+)\.\d{8}-\d{6}-\w+$"
)
# Each file's last step is read from its last 64 KiB (about 80 records).
_TAIL_BYTES = 64 * 1024
# With a window, each file is read from 1 s before it (see _read_records).
_WINDOW_MARGIN_US = 1_000_000


def generate_gantt_trace(
    log_dir: str,
    output_path: str,
    *,
    last_steps: int | None = None,
    files_modified_after_us: int | None = None,
    max_rows_per_label: int | None = 8,
) -> dict:
    """Merge per-rank JSONL into a Chrome Trace JSON for Perfetto.

    Open the result in https://ui.perfetto.dev or ``chrome://tracing``. Each JSONL file is
    one process, named "{source} rank {rank}". All spans of one asyncio task share one row,
    so nested spans stack. Tasks with the same label share rows: a task reuses a row that
    is free when it starts.

        task                           label            rows (max_rows_per_label=8)
        "trainer"                      trainer          trainer
        "rollout_worker_0" .. "_7",    rollout_worker   rollout_worker_0 .. rollout_worker_7
          one task per row
        "rollout_worker_0" .. "_31",   rollout_worker   rollout_worker row 0 .. rollout_worker row 7
          some rows hold several
        unnamed "Task-812" whose       generate         generate row 0 .. generate row 7
          first span is `generate`
        none (SPMD code, a thread)     thread 4242      thread 4242

    Each span and instant names its task in its args (``"task_name"``), so hovering a
    shared row shows which task a bar belongs to.

    Processes are sorted by name, numbers as numbers; in a process, labels with one row
    come first, then labels by name. A label cut by ``max_rows_per_label`` says so on its
    last row: "generate row 7 (+56 hidden rows)". That row also shows the hidden tasks'
    errors, but not their spans. Markers and metrics carry no task name: they go on
    their thread's row if spans ran there outside any task, else on the process's own
    track, which Perfetto names after the process.

    Example::

        # The last 10 steps of a 1000-step RL run, the controller process in Perfetto:
        generate_gantt_trace(
            "outputs/rl/structured_logs", "outputs/rl/gantt.json", last_steps=10
        )
        #   rl_controller rank 0
        #     batcher             take_finalized | batcher_add_group | ...
        #     trainer             sync_log_step | train_step | metrics_log | ...
        #     rollout_worker_0    rollout_group | rollout_record | ...
        #     ...

    Args:
        log_dir: Directory containing per-rank ``*.jsonl`` trace files.
        output_path: Path to write the merged Chrome Trace JSON.
        last_steps: Render from the first record of step ``max_step - last_steps + 1`` to
            the end. A span that started earlier is cut at that time; its ``duration_ms``
            keeps the full length. Only that part of each file is read, so the cost
            does not grow with the run. None, logs without any step, or a run with at
            most ``last_steps`` steps (a resumed run too) render everything, setup
            included.
        files_modified_after_us: Skip files last written before this time, in epoch
            microseconds, e.g. an earlier run's files in a reused dump folder.
        max_rows_per_label: Rows to show per label; the spans of tasks on other rows are
            left out and counted in the log. 8 shows how busy a label is and keeps a
            generator with 768 requests in flight on one screen. None shows every row.

    Returns:
        The Chrome Trace dict (``{"traceEvents": [...]}``). Also
        written to ``output_path``.
    """
    start_time = time.perf_counter()
    if last_steps is not None and last_steps < 1:
        raise ValueError(f"last_steps must be None or >= 1, got {last_steps}")
    if max_rows_per_label is not None and max_rows_per_label < 1:
        raise ValueError(
            f"max_rows_per_label must be None or >= 1, got {max_rows_per_label}"
        )
    paths = sorted(glob(os.path.join(log_dir, "*.jsonl")))
    if files_modified_after_us is not None:
        paths = [
            path
            for path in paths
            if os.stat(path).st_mtime_ns // 1000 >= files_modified_after_us
        ]
    window_start_us = (
        None if last_steps is None else _find_window_start_us(paths, last_steps)
    )
    paired, instants = _collect_paired_and_instants(
        _read_records(paths, window_start_us=window_start_us),
        window_start_us=window_start_us,
    )

    process_names = _process_names({event["source"] for event in [*paired, *instants]})
    sources = sorted(
        process_names,
        key=lambda source: (_natural_key(process_names[source]), source),
    )
    source_to_pid = {source: pid for pid, source in enumerate(sources)}

    row_by_task, row_names, hidden_tasks = _assign_rows(
        paired, max_rows_per_label=max_rows_per_label
    )
    events = _emit_chrome_events(
        paired=paired,
        instants=instants,
        source_to_pid=source_to_pid,
        process_names=process_names,
        row_by_task=row_by_task,
        row_names=row_names,
        hidden_tasks=hidden_tasks,
    )

    trace = {"traceEvents": events}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trace, f)

    logger.info(f"Chrome Trace: {output_path}")
    logger.info(
        f"  {len(events)} events from {len(sources)} sources, "
        f"{time.perf_counter() - start_time:.1f} s"
    )
    logger.info("  View in: chrome://tracing or https://ui.perfetto.dev")
    return trace


def _assign_rows(
    paired: list[dict], *, max_rows_per_label: int | None
) -> tuple[
    dict[tuple[str, str], int], dict[tuple[str, int], str], set[tuple[str, str]]
]:
    """Give each task a row of its process, as ``generate_gantt_trace`` describes.

    Within a label, a task takes the lowest-numbered row free when it starts (see
    ``_pack_rows``). Rows past ``max_rows_per_label`` are hidden: their tasks map to the
    label's last shown row, so their instants (in practice, errors) show there.

    Example (one process, ``max_rows_per_label=2``)::

        task       first span   time range
        trainer    train_step   [0, 100]
        Task-1     generate     [0, 50]
        Task-2     generate     [10, 40]
        Task-3     generate     [45, 90]    # reuses Task-2's row, free since 40
        Task-4     generate     [46, 60]    # both rows busy: a third row, hidden

        row_by_task  = {"trainer": 0, "Task-1": 1, "Task-2": 2, "Task-3": 2, "Task-4": 2}
        row_names    = {0: "trainer", 1: "generate row 0", 2: "generate row 1 (+1 hidden rows)"}
        hidden_tasks = {"Task-4"}
        # The real keys also hold the source: (source, "Task-1") and (source, 1).

    Returns:
        ``row_by_task``: ``(source, task_name) -> row``, where rows count from 0 in each
        process, top to bottom. ``row_names``: ``(source, row) -> row name``.
        ``hidden_tasks``: the ``(source, task_name)`` whose spans are left out.
    """
    spans_by_source: dict[str, list[dict]] = defaultdict(list)
    for span in paired:
        spans_by_source[span["source"]].append(span)

    row_by_task: dict[tuple[str, str], int] = {}
    row_names: dict[tuple[str, int], str] = {}
    hidden_tasks: set[tuple[str, str]] = set()
    num_hidden_spans = 0
    for source, spans in spans_by_source.items():
        # Each task's time range, span count, and first span: the earliest, and of spans
        # that start together the outermost (latest end). An unnamed task is labeled by it.
        task_ranges: dict[str, tuple[int, int]] = {}
        num_spans_by_task: Counter[str] = Counter()
        first_span_by_task: dict[str, dict] = {}
        for span in spans:
            task_name = span["task_name"]
            start, end = task_ranges.get(task_name, (span["start_ts"], span["end_ts"]))
            task_ranges[task_name] = (
                min(start, span["start_ts"]),
                max(end, span["end_ts"]),
            )
            num_spans_by_task[task_name] += 1
            first_span = first_span_by_task.get(task_name)
            if first_span is None or (span["start_ts"], -span["end_ts"]) < (
                first_span["start_ts"],
                -first_span["end_ts"],
            ):
                first_span_by_task[task_name] = span

        tasks_by_label: dict[str, list[str]] = defaultdict(list)
        for task_name in task_ranges:
            if _UNNAMED_TASK_RE.match(task_name):
                label = first_span_by_task[task_name]["display_name"]
            else:
                # "rollout_worker_3" -> "rollout_worker"
                label = re.sub(r"_\d+$", "", task_name)
            tasks_by_label[label].append(task_name)
        rows_by_label = {
            label: _pack_rows(task_names, task_ranges)
            for label, task_names in tasks_by_label.items()
        }

        next_row = 0
        # One-row labels first, then by name, so every rank of a role lists its rows the
        # same way however busy each one was.
        for label, rows in sorted(
            rows_by_label.items(), key=lambda item: (len(item[1]) > 1, item[0])
        ):
            shown_rows = rows[:max_rows_per_label]
            # Task names only if every shown row holds one named task. Otherwise
            # "<label> row <i>", which can't be read as one task's name.
            rows_hold_one_named_task = all(
                len(row_tasks) == 1 and not _UNNAMED_TASK_RE.match(row_tasks[0])
                for row_tasks in shown_rows
            )
            for row_index, row_tasks in enumerate(shown_rows):
                if rows_hold_one_named_task:
                    row_name = row_tasks[0]
                elif len(shown_rows) > 1:
                    row_name = f"{label} row {row_index}"
                else:
                    row_name = label
                if row_index == len(shown_rows) - 1 and len(rows) > len(shown_rows):
                    row_name += f" (+{len(rows) - len(shown_rows)} hidden rows)"
                row_names[(source, next_row)] = row_name
                for task_name in row_tasks:
                    row_by_task[(source, task_name)] = next_row
                next_row += 1
            for row_tasks in rows[len(shown_rows) :]:
                for task_name in row_tasks:
                    row_by_task[(source, task_name)] = next_row - 1
                    hidden_tasks.add((source, task_name))
                    num_hidden_spans += num_spans_by_task[task_name]

    if hidden_tasks:
        logger.info(
            f"Gantt left out {num_hidden_spans} spans of {len(hidden_tasks)} tasks to "
            f"show at most {max_rows_per_label} rows per label; pass "
            "max_rows_per_label=None to show them"
        )
    return row_by_task, row_names, hidden_tasks


def _pack_rows(
    task_names: list[str], task_ranges: dict[str, tuple[int, int]]
) -> list[list[str]]:
    """Pack tasks into rows: each takes the lowest-numbered row free when it starts.

    The lowest row, not the one freed first, so a row cap keeps the most tasks. Tasks that
    start together go in name order, numbers compared as numbers ("rollout_worker_2"
    before "rollout_worker_10").

    Example::

        _pack_rows(["a", "b", "c"], {"a": (0, 100), "b": (0, 90), "c": (95, 110)})
        # -> [["a"], ["b", "c"]]   # c starts after b ends, before a ends
    """
    rows: list[list[str]] = []
    busy_rows: list[tuple[int, int]] = []  # min-heap of (end, row)
    free_rows: list[int] = []  # min-heap of row indices
    for task_name in sorted(
        task_names, key=lambda name: (task_ranges[name][0], _natural_key(name))
    ):
        start, end = task_ranges[task_name]
        while busy_rows and busy_rows[0][0] <= start:
            heapq.heappush(free_rows, heapq.heappop(busy_rows)[1])
        if free_rows:
            row = heapq.heappop(free_rows)
        else:
            row = len(rows)
            rows.append([])
        rows[row].append(task_name)
        heapq.heappush(busy_rows, (end, row))
    return rows


def _natural_key(name: str) -> list[int | str]:
    """Sort key that compares numbers as numbers: "rank 2" before "rank 10"."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _process_names(sources: set[str]) -> dict[str, str]:
    """Name each process "{source} rank {rank}" from its file name.

    Two files that would get the same name add their file's random suffix. A file
    not named by ``TraceJsonlHandler`` keeps its name.

    Example::

        _process_names({
            "rl_controller.global_rank_0.20261106-091716-VJYQ56",
            "rl_generator.global_rank_0.20261106-091742-9NEAFG",
            "rl_generator.global_rank_0.20261106-091742-MU4RKP",
            "trace",
        })
        # -> {"rl_controller.global_rank_0.20261106-091716-VJYQ56": "rl_controller rank 0",
        #     "rl_generator.global_rank_0.20261106-091742-9NEAFG": "rl_generator rank 0 (9NEAFG)",
        #     "rl_generator.global_rank_0.20261106-091742-MU4RKP": "rl_generator rank 0 (MU4RKP)",
        #     "trace": "trace"}
    """
    short_names: dict[str, str] = {}
    for source in sources:
        match = _FILE_NAME_RE.match(source)
        short_names[source] = (
            f"{match['logger_source']} rank {match['rank']}" if match else source
        )
    num_files_by_name = Counter(short_names.values())
    return {
        source: (
            f"{name} ({source.rsplit('-', 1)[-1]})"
            if num_files_by_name[name] > 1
            else name
        )
        for source, name in short_names.items()
    }


def _read_records(paths: list[str], *, window_start_us: int | None) -> Iterator[dict]:
    """Yield the records of ``paths``, one file after another, from just before
    ``window_start_us`` (None: from the start).

    Each record gets a ``"_source_file"`` key (the filename minus ``.jsonl``), so the
    caller can group records by process. An unfinished last line (a process killed
    mid-write) is skipped with a warning.
    """
    for path in paths:
        source = os.path.basename(path).removesuffix(".jsonl")
        with open(path, "rb") as file:
            size = os.fstat(file.fileno()).st_size
            if window_start_us is not None:
                # Start 1 s early: a span that started just before the window then pairs
                # with its start record. Without it, its start is end - duration_ms,
                # which lands after its own child when the end record was stamped late.
                _seek_first_line(
                    file,
                    lambda record: record["time_us"],
                    window_start_us - _WINDOW_MARGIN_US,
                )
            while (record := _next_record(file)) is not None:
                record["_source_file"] = source
                yield record
            if file.tell() < size:
                logger.warning(f"Gantt skipped the unfinished last line of {path}")


def _find_window_start_us(paths: list[str], last_steps: int) -> int | None:
    """Return when the last ``last_steps`` steps start: the earliest first record of
    those steps over all files. None (render everything) if no file ends with a stepped
    record, or if the run has at most ``last_steps`` steps, counted from its first step
    (a run resumed from a checkpoint starts after step 1).

    The last step comes from the end of each file. The run's first step and each file's
    first record of the window come from binary searches, so a 1000-step run costs the
    same as a 10-step one.

    Example::

        # steps by line (from line 0)   file A: [None, 1, 1, 2, 2, 3]   file B: [1, 2, 3, 3]
        _find_window_start_us([A, B], last_steps=2)
        # -> last step 3, window from step 2: min(time_us of A line 3, B line 1)
    """
    tail_records = []
    for path in paths:
        with open(path, "rb") as file:
            _seek_line(file, os.fstat(file.fileno()).st_size - _TAIL_BYTES)
            while (record := _next_record(file)) is not None:
                if record.get("step") is not None:
                    tail_records.append(record)
    if not tail_records:
        return None

    first_step = max(record["step"] for record in tail_records) - last_steps + 1
    # The run's first step over all files; the tails' steps count too, so the min
    # always has one.
    run_first_steps = [record["step"] for record in tail_records]
    for path in paths:
        with open(path, "rb") as file:
            # Steps are >= 0, so this finds the file's first stepped record.
            _seek_first_line(file, lambda record: record.get("step"), 0)
            if (record := _next_record(file)) is not None:
                run_first_steps.append(record["step"])
    if first_step <= min(run_first_steps):
        return None
    # The tails' records of the window count too, so the min below always has one.
    first_times_us = [
        record["time_us"] for record in tail_records if record["step"] >= first_step
    ]
    for path in paths:
        with open(path, "rb") as file:
            _seek_first_line(file, lambda record: record.get("step"), first_step)
            record = _next_record(file)
            if record is not None:
                first_times_us.append(record["time_us"])
    return min(first_times_us)


def _seek_first_line(
    file: BinaryIO, key: Callable[[dict], int | None], target: int
) -> None:
    """Move ``file`` to its first line whose ``key`` is >= ``target``, by binary search.

    Keys must rise through the file, as ``time_us`` does. ``step`` does too, with two
    exceptions:
    - A record without a step counts as below ``target``. A process logs those only
      before its first ``set_step``; after it, ``get_step`` falls back to that step.
    - A task keeps the step it was created in (asyncio copies the step ContextVar), so
      an older step can follow a newer one. Such a record can only move the result
      later, and the caller takes the earliest result over all files.

    Example::

        # steps by line (from line 0): [None, 1, 1, 2, 2, 3]
        _seek_first_line(file, lambda record: record.get("step"), 2)  # -> at line 3
    """
    # TODO: probe the largest step in a block of records, so that a file with many
    # records of older tasks can't hide the newer ones.
    low, high = 0, os.fstat(file.fileno()).st_size
    while low < high:
        middle = (low + high) // 2
        _seek_line(file, middle)
        record = _next_record(file)
        value = None if record is None else key(record)
        # The end of the file counts as >= target; a record without the key, as below.
        if record is None or (value is not None and value >= target):
            high = middle
        else:
            low = middle + 1
    _seek_line(file, low)


def _seek_line(file: BinaryIO, offset: int) -> None:
    """Move ``file`` to the first line that starts at or after byte ``offset``."""
    if offset <= 0:
        file.seek(0)
    else:
        # Byte offset - 1 is a newline when a line starts at offset.
        file.seek(offset - 1)
        file.readline()


def _next_record(file: BinaryIO) -> dict | None:
    """Read the next record; None at the end of the file or at an unfinished last
    line, which stays unread."""
    while line := file.readline():
        if not line.endswith(b"\n"):
            file.seek(-len(line), os.SEEK_CUR)
            return None
        if line.strip():
            return json.loads(line)
    return None


def _collect_paired_and_instants(
    records: Iterator[dict], *, window_start_us: int | None
) -> tuple[list[dict], list[dict]]:
    """Pair ``_start`` / ``_end`` via a per-(source, task_name, span type) LIFO stack.

    Each asyncio task name gets its own stacks, so nested spans within a
    task pair correctly regardless of what other tasks on the same
    process are doing. Records outside any asyncio task (SPMD code, plain
    threads) pair on stacks per native thread, as task ``"thread {tid}"``.
    A span runs from its ``_start`` record's ``time_us`` to its ``_end``
    record's; an ``_end`` without a ``_start`` (cut off by the window)
    starts ``duration_ms`` before its end.

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

        # window_start_us=510: fwd_bwd is dropped (it ended at 500); step is
        # kept with start_ts=510 and duration_ms=0.5; the loss instant is kept.

    ``paired`` is in ``_end``-record order: ``fwd_bwd`` ends first.

    Args:
        records: All JSONL records across all sources, in the order
            produced by ``_read_records``.
        window_start_us: Drop spans that end before this time and instants
            before it; a span that starts before it starts at it. None keeps
            everything.

    Returns:
        ``(paired, instants)`` -- paired spans in ``_end``-record order and
        instant events in record order.
    """
    # -inf keeps every record.
    min_time_us = -math.inf if window_start_us is None else window_start_us
    paired: list[dict[str, Any]] = []
    instants: list[dict[str, Any]] = []
    # LIFO stack per (source, task_name, span type).
    pending: dict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for record in records:
        event_type = record.get("log_type_name", "")
        log_type = record.get("log_type", "")
        time_us = record["time_us"]
        step = record.get("step")
        task_name = record.get("task_name") or f"thread {record.get('tid')}"
        source = record["_source_file"]
        caller = record.get("caller")

        # Point-in-time records (log_trace_instant, log_trace_scalar) carry
        # log_type=instant. Branch on intent before suffix matching so that
        # instants whose names happen to end in "_start" (structured_logger_started,
        # training_start) don't get wrongly routed to the span-pair stack.
        if log_type == str(LogType.INSTANT):
            if event_type == "metric_value":
                event_name = record.get("event_name", "metric")
                value = record.get("value") or 0
                display_name = f"{event_name}={value:.4f}"
            else:
                display_name = event_type
            instants.append(
                {
                    "name": display_name,
                    "time_us": time_us,
                    "source": source,
                    "task_name": task_name,
                    "step": step,
                    "caller": caller,
                }
            )
        elif event_type.endswith("_start"):
            type_name = event_type.removesuffix("_start")
            display_name = record.get("event_name") or type_name
            pending[(source, task_name, type_name)].append(
                {"ts": time_us, "display_name": display_name, "caller": caller}
            )

        elif event_type.endswith("_end"):
            type_name = event_type.removesuffix("_end")
            duration_ms = record.get("value", 0)
            stack = pending.get((source, task_name, type_name))
            start = stack.pop() if stack else None
            # The end record's own time, not start + duration_ms: a record is stamped a
            # moment after its span's timer reads, so start + duration_ms can overrun the
            # parent span. Records of one task are stamped in order, so children nest.
            end_ts = time_us
            if start is not None:
                start_ts = start["ts"]
            else:
                start_ts = end_ts - (duration_ms or 0) * 1000
            if end_ts >= min_time_us:
                paired.append(
                    {
                        "task_name": task_name,
                        "source": source,
                        "start_ts": max(start_ts, min_time_us),
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
                    "source": source,
                    "task_name": task_name,
                    "step": step,
                    "caller": caller,
                }
            )

    instants = [instant for instant in instants if instant["time_us"] >= min_time_us]
    return paired, instants


def _emit_chrome_events(
    *,
    paired: list[dict],
    instants: list[dict],
    source_to_pid: dict[str, int],
    process_names: dict[str, str],
    row_by_task: dict[tuple[str, str], int],
    row_names: dict[tuple[str, int], str],
    hidden_tasks: set[tuple[str, str]],
) -> list[dict]:
    """Build the Chrome Trace event list (one Perfetto process per source).

    Emits three kinds of events:

    - ``"M"`` (metadata) -- per source, ``process_name`` and
      ``process_sort_index``; per row, ``thread_name``. Without the sort
      index, Perfetto orders processes by slice count. Rows sort by tid,
      which follows row order.
    - ``"X"`` (complete) -- one per paired span on its task's row, with
      ``ts`` / ``dur`` for rendering as a bar. Spans of hidden tasks are
      skipped.
    - ``"i"`` (instant) -- one per instant event, rendered as a
      vertical marker on its task's row, or on the process's own track if
      its task has no span.

    Row ``r`` gets tid ``len(source_to_pid) + r``, above every pid. Perfetto
    merges tid 0 into the row whose tid equals the pid, so two rows would share
    one track.

        3 processes (pids 0, 1, 2), each with rows 0, 1 -> tids 3, 4 in each

    Args:
        paired: Spans from ``_collect_paired_and_instants``.
        instants: Instant events from ``_collect_paired_and_instants``.
        source_to_pid: Map from source name to Perfetto pid.
        process_names: Map from source name to process name.
        row_by_task: Map from ``(source, task_name)`` to the task's row.
        row_names: Map from ``(source, row)`` to the row's name.
        hidden_tasks: Tasks on rows past the cap; their spans are skipped, their
            instants go on the row in ``row_by_task``.

    Returns:
        A flat list of Chrome Trace event dicts ready to serialize.
    """
    events: list[dict[str, Any]] = []

    first_tid = len(source_to_pid)

    for source, pid in source_to_pid.items():
        events.append(
            {
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": pid,
                "args": {"name": process_names[source]},
            }
        )
        events.append(
            {
                "name": "process_sort_index",
                "ph": "M",
                "pid": pid,
                "tid": pid,
                "args": {"sort_index": pid},
            }
        )

    for (source, row), row_name in row_names.items():
        events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "pid": source_to_pid[source],
                "tid": first_tid + row,
                "args": {"name": row_name},
            }
        )

    for span in paired:
        task = (span["source"], span["task_name"])
        if task in hidden_tasks:
            continue
        duration_ms = span["duration_ms"] or 0
        args: dict = {
            "task_name": span["task_name"],
            **({"step": span["step"]} if span["step"] is not None else {}),
            "duration_ms": f"{duration_ms:.2f}" if duration_ms else "0.00",
        }
        if span.get("caller"):
            args["caller"] = span["caller"]
        events.append(
            {
                "name": span["display_name"],
                "ph": "X",
                "ts": span["start_ts"],
                "dur": span["end_ts"] - span["start_ts"],
                "pid": source_to_pid[span["source"]],
                "tid": first_tid + row_by_task[task],
                "args": args,
            }
        )

    for instant in instants:
        args = {
            "task_name": instant["task_name"],
            **({"step": instant["step"]} if instant["step"] is not None else {}),
        }
        if instant.get("caller"):
            args["caller"] = instant["caller"]
        event = {
            "name": instant["name"],
            "ph": "i",
            "ts": instant["time_us"],
            "pid": source_to_pid[instant["source"]],
            "args": args,
        }
        row = row_by_task.get((instant["source"], instant["task_name"]))
        if row is None:
            # The process's own track; no row uses tid == pid.
            # TODO: a named row for instants with no task: Perfetto names this track
            # after the process. Not done: it needs a check in the Perfetto UI, and the
            # README explains the track meanwhile.
            event.update(s="p", tid=event["pid"])
        else:
            event.update(s="t", tid=first_tid + row)
        events.append(event)

    return events
