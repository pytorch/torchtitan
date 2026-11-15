# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU time and memory of one process, optionally with its child processes, read from /proc."""

from __future__ import annotations

import collections
import os
from pathlib import Path

from torchtitan.rl.observability.metrics.types import Max, Metric, Min, Sum


class ProcessStats:
    """Reads this process's CPU seconds since the last read, its anonymous memory, and its host's
    available memory.

        children=False   this process only
        children=True    plus every live descendant (e.g. a rollout worker's agent subprocesses)
                         and, for CPU, every descendant already reaped inside the tree
        left the tree    a descendant whose parent exited first, or that double-forked: counted
                         while in the tree; once it leaves, its CPU so far comes out of the next
                         reads (e.g. 50 s counted, then 0 s reported until the tree does 50 s more)
        never counted    a container's processes (children of dockerd)

    Example: ProcessStats("rollout_worker", children=True).metrics() on two workers, reduced
    to the sum (the job), the max (the largest worker) and the min (the fullest host):
        process/rollout_worker/cpu_seconds/sum              79.2   CPU-s since the last read
        process/rollout_worker/rss_anon_gib/sum             52.8
        process/rollout_worker/rss_anon_gib/max             27.1
        process/rollout_worker/host_mem_available_gib/min   512.0
    """

    def __init__(self, role: str, *, children: bool) -> None:
        self._role = role
        self._children = children
        # `_read` keeps the larger of this and its new reading, so start from 0.
        self._total_cpu_seconds = 0.0
        self._total_cpu_seconds, _ = self._read()

    def metrics(self) -> list[Metric]:
        total_cpu_seconds, live_descendants = self._read()
        cpu_since_last_read = total_cpu_seconds - self._total_cpu_seconds
        self._total_cpu_seconds = total_cpu_seconds
        anon_gib = sum(map(_anonymous_kib, (os.getpid(), *live_descendants))) / 2**20
        prefix = f"process/{self._role}"
        return [
            Metric(f"{prefix}/cpu_seconds", Sum(cpu_since_last_read)),
            Metric(f"{prefix}/rss_anon_gib", Sum(anon_gib)),
            Metric(f"{prefix}/rss_anon_gib", Max(anon_gib)),
            Metric(
                f"{prefix}/host_mem_available_gib", Min(_mem_available_kib() / 2**20)
            ),
        ]

    def _read(self) -> tuple[float, list[int]]:
        """Total CPU seconds so far, and the live descendants' pids.

        A reaped child's CPU moves into its parent's counters, so:
        1. Parents are read before children: this process (`os.times`), then the descendants
           top-down. A child reaped between its parent's read and its own is missed once,
           never counted twice.
        2. The total keeps the largest reading, so the missed time shows up at a later read,
           not as a negative delta.

        Example: a child that ends at 1.2 s of CPU.
            read 1   the child is live at 1.0 s                       total counts 1.0 s
            read 2   reads the parent; the child is reaped before      total stays at 1.0 s
                     read 2 reaches it
            read 3   the parent's counters hold the child's 1.2 s     total grows by 0.2 s
        """
        if not self._children:
            own = os.times()
            return max(self._total_cpu_seconds, own.user + own.system), []
        # Find the tree first; then read the counters, parents before children.
        descendants = _descendant_start_times(os.getpid())
        own = os.times()
        seconds = own.user + own.system + own.children_user + own.children_system
        live_descendants = []
        for pid, start_time in descendants.items():
            stat = _read_stat(pid)
            # None: the process exited. Another start time: its pid now names a new process.
            if stat is not None and stat[_START_TIME] == start_time:
                live_descendants.append(pid)
                seconds += (
                    sum(int(field) for field in stat[_CPU_TIMES]) / _TICKS_PER_SECOND
                )
        # TODO: per-process deltas, or the cgroup's cpu.stat, would keep the CPU of a descendant
        #   that left the tree; this max takes it back out of the next reads.
        return max(self._total_cpu_seconds, seconds), live_descendants


# Fields of /proc/<pid>/stat after "(comm)", 0-based: ppid (field 4), utime, stime, cutime,
# cstime (fields 14-17) and starttime (field 22).
_PPID, _CPU_TIMES, _START_TIME = 1, slice(11, 15), 19
_TICKS_PER_SECOND = os.sysconf("SC_CLK_TCK")


def _descendant_start_times(pid: int) -> dict[int, str]:
    """Every live process below `pid`, parents before children, with its start time.

    Example: 100 -> 101 -> 102, and 100 -> 103: keys in the order 101, 103, 102
    """
    start_times, children = {}, collections.defaultdict(list)
    for entry in os.scandir("/proc"):
        if entry.name.isdigit():
            stat = _read_stat(int(entry.name))
            if stat is not None:
                start_times[int(entry.name)] = stat[_START_TIME]
                children[int(stat[_PPID])].append(int(entry.name))
    found, parents = {}, collections.deque([pid])
    while parents:
        for child in children[parents.popleft()]:
            found[child] = start_times[child]
            parents.append(child)
    return found


def _read_stat(pid: int) -> list[str] | None:
    """/proc/<pid>/stat's fields after "(comm)"; None if the process has exited."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # comm may hold spaces and parentheses, so split after its last ")".
    return stat.rsplit(")", 1)[1].split()


def _anonymous_kib(pid: int) -> int:
    """`Anonymous:` of /proc/<pid>/smaps_rollup, in KiB. 0 when the process
    - has exited (a zombie's read raises ProcessLookupError), or
    - hides it (PermissionError: it called prctl(PR_SET_DUMPABLE, 0), or runs a setuid binary).
    `status`'s RssAnon is cheaper but can read low before Linux 6.16: each CPU reports its
    share of the counter late."""
    try:
        rollup = Path(f"/proc/{pid}/smaps_rollup").read_text()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return 0
    for line in rollup.splitlines():
        if line.startswith("Anonymous:"):
            return int(line.split()[1])
    raise RuntimeError(f"/proc/{pid}/smaps_rollup has no Anonymous line: {rollup!r}")


def _mem_available_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("/proc/meminfo has no MemAvailable line")
