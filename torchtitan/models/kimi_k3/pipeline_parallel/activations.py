# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where a pipeline rank keeps what its backward needs."""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist
from torch.distributed.pipelining.schedules import _ComputationType
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.activation_storage import (
    ActivationStorage,
    ChunkKey,
    RemoteBackend,
)

logger = logging.getLogger(__name__)

_KIND = {
    _ComputationType.FORWARD: "F",
    _ComputationType.FULL_BACKWARD: "B",
    _ComputationType.BACKWARD_INPUT: "B",
    _ComputationType.BACKWARD_WEIGHT: "W",
}

# A pool span holds the most bytes its source parks at once, plus room for fragmentation.
_SLACK = 1.1

ActionKey = tuple[str, int, int]
Item = tuple[int, int]


@dataclass(kw_only=True, slots=True)
class PPMemoryConfig:
    """How a pipeline rank stores the tensors its backward reads."""

    cpu_offload: Literal["none", "all", "planned"] = "none"
    """``all`` offloads every tensor autograd saves outside the model's last layer to host
    memory (``CheckpointPolicy.MUST_CPU_OFFLOAD``); ``planned`` does so in the first step, then
    offloads what a plan over every rank's profiled timeline picks. The rank store's blocks stay
    on the device."""

    balance: bool = False
    """With ``cpu_offload="planned"``, the plan parks saves in pools on lighter pipeline ranks
    through mooncake's transfer engine before it offloads them to host memory."""

    target_gib: float | None = None
    """The peak each rank is planned down to; unset is the mean of the ranks' profiled peaks."""

    host_gbps: float = 25.0
    """Host link bandwidth of one rank that the plan budgets, in GB/s."""

    peer_gbps: float = 25.0
    """Rank-to-rank bandwidth that the plan budgets, in GB/s."""

    remote_protocol: Literal["tcp", "rdma"] = "tcp"
    """mooncake transport: ``tcp`` keeps a pool in the destination's pinned host memory,
    ``rdma`` in its device memory."""

    remote_device_names: str = ""
    """RDMA devices passed to mooncake's transfer engine, comma separated."""

    staging_mib: int = 256
    """Registered buffer a rank copies through to its pool on another rank."""

    cpu_offload_budget_gib: float | None = None
    """Host memory a rank holds offloaded saves in; a save past it stays on the device."""

    cpu_offload_prefetch_n_layers: int = 1
    """Backward reads offloaded saves back this many layers ahead."""

    min_tensor_mib: int = 1
    """Saves smaller than this stay on the device."""


def compute_actions(order: list[Any]) -> list[ActionKey]:
    return [
        (_KIND[a.computation_type], a.stage_index, a.microbatch_index)
        for a in order
        if a is not None and a.computation_type in _KIND
    ]


class BackwardPrefetch:
    """Starts reading back what a stage's backward needs first when the rank's previous compute
    action begins."""

    def __init__(self, storage: ActivationStorage, actions: list[ActionKey]) -> None:
        self._storage = storage
        self._next_backward = {
            key: following[1:]
            for key, following in zip(actions, actions[1:])
            if following[0] == "B"
        }

    def begin(self, key: ActionKey) -> None:
        item = self._next_backward.get(key)
        if item is not None:
            self._storage.prefetch_first(*item)

    def end(self, key: ActionKey) -> None:
        pass


@dataclass
class RankProfile:
    """One profiled step of a rank: each compute action's peak and duration, and what could move."""

    peaks: list[int]
    seconds: list[float]
    items: dict[Item, int]


def _max_live(entries: list[tuple[float, float, int]]) -> int:
    events = sorted(
        [(t0, n) for t0, _, n in entries] + [(t1, -n) for _, t1, n in entries]
    )
    live = best = 0
    for _, n in events:
        live += n
        best = max(best, live)
    return best


class MemoryPlan:
    """Which stage micro-batches each rank moves, to host or to a lighter rank's pool, and when
    each comes back: a move lowers the rank's profiled timeline from its copy out to its copy
    back, and the plan lowers the highest peak first until every rank is under the target.
    """

    def __init__(
        self,
        orders: dict[int, list[Any]],
        profiles: dict[int, RankProfile],
        *,
        target_bytes: int | None,
        offload: bool,
        balance: bool,
        host_bps: float,
        peer_bps: float,
    ) -> None:
        self.backend: dict[tuple[int, int, int], str] = {}
        self.windows: dict[tuple[int, int, int], tuple[int, int]] = {}
        self.due: dict[int, dict[ActionKey, list[Item]]] = defaultdict(dict)
        self.dests: dict[int, int] = {}
        self.spans: dict[int, int] = {}
        ranks = sorted(profiles)
        actions = {r: compute_actions(orders[r]) for r in ranks}
        position = {r: {key: i for i, key in enumerate(actions[r])} for r in ranks}
        timeline = {r: list(profiles[r].peaks) for r in ranks}
        starts: dict[int, list[float]] = {}
        for r in ranks:
            t, begin = 0.0, []
            for seconds in profiles[r].seconds:
                begin.append(t)
                t += seconds
            starts[r] = begin + [t]
        self.profiled = {r: max(timeline[r], default=0) for r in ranks}
        self.target = (
            target_bytes
            if target_bytes is not None
            else sum(self.profiled.values()) // max(len(ranks), 1)
        )
        budget = {r: 0.8 * starts[r][-1] * host_bps if offload else 0.0 for r in ranks}
        parked: dict[int, list[tuple[float, float, int]]] = defaultdict(list)

        def window(r: int, item: Item, bps: float) -> tuple[int, int] | None:
            f = position[r].get(("F", *item))
            b = position[r].get(("B", *item))
            if f is None or b is None:
                return None
            seconds = profiles[r].items[item] / bps
            # the copy out has finished by the middle of the first action that gains its bytes
            out = next(
                (
                    k
                    for k in range(f + 1, b)
                    if (starts[r][k] + starts[r][k + 1]) / 2
                    >= starts[r][f + 1] + seconds
                ),
                None,
            )
            back = next(
                (
                    k
                    for k in range(b - 1, f, -1)
                    if starts[r][b] - starts[r][k] >= seconds
                ),
                None,
            )
            if out is None or back is None or out >= back:
                return None
            return out, back

        def pool(d: int) -> int:
            return sum(self.spans[src] for src, dst in self.dests.items() if dst == d)

        def peak(r: int) -> int:
            return max(timeline[r], default=0) + pool(r)

        blocked: set[int] = set()
        while True:
            open_ranks = [r for r in ranks if r not in blocked]
            if not open_ranks:
                break
            r = max(open_ranks, key=lambda x: (peak(x), -x))
            if peak(r) <= self.target:
                break
            k = max(range(len(timeline[r])), key=lambda i: (timeline[r][i], -i))
            best = None
            items = sorted(
                profiles[r].items.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
            )
            for item, nbytes in items:
                if (r, *item) in self.backend or nbytes <= 0:
                    continue
                if balance and r not in self.dests.values():
                    span = window(r, item, peer_bps)
                    if span is not None and span[0] <= k < span[1]:
                        entry = (starts[r][span[0]], starts[r][span[1]], nbytes)
                        grown = int(_max_live(parked[r] + [entry]) * _SLACK)
                        grown -= self.spans.get(r, 0)
                        fixed = self.dests.get(r)
                        for d in sorted(ranks, key=lambda x: (peak(x), x)):
                            if d == r or d in self.dests or fixed not in (None, d):
                                continue
                            if peak(d) + grown <= self.target:
                                best = ("remote", item, nbytes, span, d, entry)
                                break
                if best is None and offload and budget[r] >= 2 * nbytes:
                    span = window(r, item, host_bps)
                    if span is not None and span[0] <= k < span[1]:
                        best = ("host", item, nbytes, span, None, None)
                if best is not None:
                    break
            if best is None:
                blocked.add(r)
                continue
            kind, item, nbytes, (out, back), d, entry = best
            self.backend[(r, *item)] = kind
            self.windows[(r, *item)] = (out, back)
            for i in range(out, back):
                timeline[r][i] -= nbytes
            self.due[r].setdefault(actions[r][back], []).append(item)
            if kind == "remote":
                assert d is not None and entry is not None
                self.dests[r] = d
                parked[r].append(entry)
                self.spans[r] = int(_max_live(parked[r]) * _SLACK)
            else:
                budget[r] -= 2 * nbytes
        self.timelines = timeline
        self.peaks = {r: peak(r) for r in ranks}

    def summary(self, rank: int) -> str:
        moved = [key for key in self.backend if key[0] == rank]
        host = sum(1 for key in moved if self.backend[key] == "host")
        return (
            f"rank {rank}: {len(moved)} stage micro-batch(es) moved ({host} to host, "
            f"{len(moved) - host} to rank {self.dests.get(rank, '-')}), peak "
            f"{self.profiled[rank] / 2**30:.2f} -> {self.peaks[rank] / 2**30:.2f} GiB, "
            f"target {self.target / 2**30:.2f} GiB"
        )


class PPMemoryController(BackwardPrefetch):
    """Profiles a rank's first training step with every eligible save offloaded, then plans
    its moves with the other ranks once that step has returned and applies them from the next
    step on."""

    def __init__(
        self,
        config: PPMemoryConfig,
        storage: ActivationStorage,
        *,
        group: dist.ProcessGroup | None,
        rank: int,
        orders: dict[int, list[Any]],
        device: torch.device,
        skip_layers: Collection[int] = (),
    ) -> None:
        super().__init__(storage, compute_actions(orders[rank]))
        self._config = config
        self._group = group
        self._rank = rank
        self._orders = orders
        # pyrefly: ignore [read-only]
        self._device = device
        self._skip = skip_layers
        self._actions = compute_actions(orders[rank])
        self._peaks: list[int] = []
        self._off_device = 0
        self._marks: list[Any] = []
        self.plan: MemoryPlan | None = None
        storage.counting = True

    def attach(self, schedule: Any) -> None:
        step = schedule.step

        # The plan's collective must follow every send of the profiled step; only step's return does.
        def step_then_plan(*args: Any, **kwargs: Any) -> Any:
            out = step(*args, **kwargs)
            if self.plan is None and len(self._peaks) == len(self._actions):
                self._build()
            return out

        schedule.step = step_then_plan

    def policy(self, tensor: torch.Tensor, chunk: ChunkKey) -> CheckpointPolicy:
        assert self.plan is not None
        if chunk[2] in self._skip:
            return CheckpointPolicy.PREFER_SAVE
        if (self._rank, chunk[0], chunk[1]) in self.plan.backend:
            return CheckpointPolicy.MUST_CPU_OFFLOAD
        return CheckpointPolicy.PREFER_SAVE

    def route(self, chunk: ChunkKey) -> str:
        assert self.plan is not None
        return self.plan.backend[(self._rank, chunk[0], chunk[1])]

    def _mark(self) -> Any:
        if self._device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def begin(self, key: ActionKey) -> None:
        if self.plan is not None:
            for stage, mb in self.plan.due[self._rank].get(key, []):
                self._storage.prefetch_first(stage, mb)
            return
        super().begin(key)
        if len(self._peaks) == len(self._actions):
            return
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)
        self._off_device = self._storage.off_device_bytes
        self._marks.append(self._mark())

    def end(self, key: ActionKey) -> None:
        if self.plan is not None or len(self._peaks) == len(self._actions):
            return
        peak = 0
        if self._device.type == "cuda":
            # The profiled step moved every eligible save; add back what was off the device.
            off = max(self._off_device, self._storage.off_device_bytes)
            peak = torch.cuda.max_memory_allocated(self._device) + off
        self._peaks.append(peak)
        self._marks.append(self._mark())

    def _seconds(self) -> list[float]:
        marks = self._marks
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            return [
                marks[2 * i].elapsed_time(marks[2 * i + 1]) / 1e3
                for i in range(len(self._actions))
            ]
        return [marks[2 * i + 1] - marks[2 * i] for i in range(len(self._actions))]

    def _resident(self, items: dict[Item, int]) -> list[int]:
        # Without device statistics the timeline is the saves held from each forward to its backward.
        position = {key: i for i, key in enumerate(self._actions)}
        timeline = [0] * len(self._actions)
        for (stage, mb), nbytes in items.items():
            f, b = position.get(("F", stage, mb)), position.get(("B", stage, mb))
            if f is None or b is None:
                continue
            for i in range(f, b + 1):
                timeline[i] += nbytes
        return timeline

    def _build(self) -> None:
        storage = self._storage
        items: Counter[Item] = Counter()
        for (stage, mb, layer), nbytes in storage.chunk_bytes.items():
            if layer not in self._skip:
                items[(stage, mb)] += nbytes
        peaks = self._peaks if self._device.type == "cuda" else self._resident(items)
        profile = RankProfile(peaks=peaks, seconds=self._seconds(), items=dict(items))
        gathered: list[RankProfile | None] = [None] * dist.get_world_size(self._group)
        dist.all_gather_object(gathered, profile, group=self._group)
        config = self._config
        plan = MemoryPlan(
            self._orders,
            {r: p for r, p in enumerate(gathered) if p is not None},
            target_bytes=None
            if config.target_gib is None
            else int(config.target_gib * 2**30),
            offload=True,
            balance=config.balance,
            host_bps=config.host_gbps * 1e9,
            peer_bps=config.peer_gbps * 1e9,
        )
        if config.balance and plan.dests:
            storage.add_backend(
                "remote",
                RemoteBackend(
                    self._group,
                    dests=plan.dests,
                    spans=plan.spans,
                    staging_bytes=config.staging_mib << 20,
                    device=self._device,
                    protocol=config.remote_protocol,
                    device_names=config.remote_device_names,
                ),
            )
        storage.counting = False
        self.plan = plan
        storage.set_policy(self.policy, self.route)
        logger.info("Kimi K3 pipeline memory plan, %s", plan.summary(self._rank))
