# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


r"""
Paged stash runtime for CUDA-graphable MoE.

A port of Megatron-Core 0.19.0's paged stashing
(``megatron.core.transformer.moe.paged_stash``), shared by TorchTitan's two
ways of training: the eager trainer (``eager.py``) and GraphTrainer's traced
graphs (``experiments/graph_trainer/paged_stash_memory_policy.py``). The frontends
decide *which* saved activations are paged and *where* the stash and reload
points are; this module is everything after that decision, and is the same for
both.

Why Paged Stashing
------------------

When a MoE token dispatcher runs with a *static* capacity factor, as a CUDA
graph requires, every permuted activation is allocated for the worst case
while only ``sum(tokens_per_expert)`` rows hold data. With CUDA graphs on,
every layer that saves such an activation also needs its own capacity-padded
buffer resident in the graph pool, so the un-paged cost is
``num_layers x padded``, and with pipeline parallelism it is held once per
in-flight microbatch.

Paged stashing reclaims that gap: once the forward consumers are done, the live
rows are copied into a shared pool of fixed-size pages and the padded tensor is
freed; backward pages it back in. The pool is shared by every layer -- one per
``(dtype, hidden_size)`` -- and sized from measured live rows rather than the
padded worst case, which turns ``O(layers x worst case)`` into
``O(worst case + actual usage)``.

What lives here
---------------

* ``PagedStashBuffer``: Megatron's page buffer and circular freelists.
* ``PagedStashManager``: the process-wide state -- buffers, flags, the side
  stream, the ``begin -> capture -> captured`` lifecycle (measure, allocate,
  page), page-granular sizing, the stash records, and Megatron's protocol for
  joining stashes and prefetching reloads one schedule entry ahead.
* ``infer_slot_dims``: how a tensor's bytes read as token rows.
* The paged-layer schedule and schedule-replay sizing over PyTorch pipeline
  ``_Action`` lists.
* ``PagedStashRunner``: overflow detection and recovery around a step, in
  ``deferred`` (non-blocking) or ``blocking`` (Megatron's rerun) mode.
The Triton kernels (``kernels.py``) and the ``PagedStashBuffer`` layout are
copied from Megatron-Core unchanged, so the page allocator, the pinned-host
spill path, and the overflow protocol behave identically. The settings are in
``config.py``.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, TYPE_CHECKING, TypeVar

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import (
    _Action,
    BACKWARD_INPUT,
    FORWARD,
    FULL_BACKWARD,
)
from torch.distributed.tensor import DTensor

from torchtitan.distributed.deepep.paged_stash.config import OverflowCheckMode
from torchtitan.distributed.deepep.paged_stash.kernels import (
    GLOBAL_BLOCK_SIZE,
    paged_stash_copy_kernel,
    paged_stash_pop_kernel,
)

if TYPE_CHECKING:
    from torchtitan.components.optim.optimizer import OptimizersContainer


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


# ===========================================================================
# Paged stash buffers and stash slots
# ===========================================================================
#
# ``PagedStashBuffer`` is a structural port of Megatron-Core 0.19.0's
# ``megatron.core.transformer.moe.paged_stash.PagedStashBuffer``: the same
# ``[num_pages * page_size, hidden]`` layout, the same two-element
# ``[cuda, host]`` freelist state, and the same preallocated reset values so
# ``reset()`` never allocates (required inside a CUDA graph).


# Megatron pages columnwise MXFP8 scale-inverse tensors at 1/32 the token
# granularity of their data tensor. GraphTrainer's MoE saves plain tensors, so
# the manager tracks a per-slot ``token_divisor`` instead of a boolean, with 1
# meaning "one row per token". Kept as a named constant so an MXFP8 GraphTrainer
# MoE can reuse the Megatron value directly.
SCALE_INV_BLOCK_SIZE = 32


def _num_pages(num_tokens: int, page_size: int) -> int:
    """Pages needed to hold ``num_tokens`` rows."""
    return (num_tokens + page_size - 1) // page_size


class PagedStashBuffer:
    """A paged stash buffer with page-level memory management.

    Supports both CUDA and an optional pinned host buffer for overflow fallback.
    Buffers are organized as ``[num_pages * page_size, hidden_size]`` and use
    per-buffer circular freelists tracked as two-element state:
    ``[0] = CUDA``, ``[1] = host``.
    """

    def __init__(
        self,
        num_tokens: int,
        hidden_size: int,
        page_size: int,
        device: torch.device,
        overflow: torch.Tensor,
        host_spill: torch.Tensor,
        dtype: torch.dtype,
        num_tokens_host: int = 0,
    ) -> None:
        self.hidden_size = hidden_size
        self.page_size = page_size
        # pyrefly: ignore [read-only]
        self.device = device
        self.dtype = dtype
        self.overflow = overflow  # GPU flag (shared across all buffers)
        self.host_spill = host_spill  # GPU flag (shared across all buffers)

        # CUDA buffer
        self.num_cuda_pages = _num_pages(num_tokens, page_size)
        self.total_cuda_tokens = self.num_cuda_pages * page_size
        self.cuda_buffer = torch.empty(
            (self.total_cuda_tokens, hidden_size), dtype=dtype, device=device
        )

        # Host buffer (pinned), optional
        self.num_host_pages = _num_pages(num_tokens_host, page_size)
        self.total_host_tokens = self.num_host_pages * page_size
        if self.num_host_pages > 0:
            self.host_buffer: torch.Tensor | None = torch.empty(
                (self.total_host_tokens, hidden_size),
                dtype=dtype,
                device="cpu",
                pin_memory=True,
            )
        else:
            self.host_buffer = None

        # Freelist state: shape (2,), index 0 = CUDA, 1 = host. Device memory so
        # the Triton kernels can read/update it without a host round trip.
        self.free_list_head = torch.zeros(2, dtype=torch.int64, device=device)
        self.free_list_tail = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages], dtype=torch.int64, device=device
        )
        self.free_list_capacity = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages], dtype=torch.int64, device=device
        )

        # Freelist arrays (device memory): page IDs for each buffer.
        self.free_list_cuda = torch.arange(
            self.num_cuda_pages, dtype=torch.int64, device=device
        )
        if self.num_host_pages > 0:
            self.free_list_host = torch.arange(
                self.num_host_pages, dtype=torch.int64, device=device
            )
        else:
            self.free_list_host = torch.empty(0, dtype=torch.int64, device=device)

        # Preallocated reset values (CUDA graph safe: no allocation in reset()).
        self._reset_tail = torch.tensor(
            [self.num_cuda_pages, self.num_host_pages], dtype=torch.int64, device=device
        )
        self._reset_free_list_cuda = torch.arange(
            self.num_cuda_pages, dtype=torch.int64, device=device
        )
        if self.num_host_pages > 0:
            self._reset_free_list_host: torch.Tensor | None = torch.arange(
                self.num_host_pages, dtype=torch.int64, device=device
            )
        else:
            self._reset_free_list_host = None

    def reset(self) -> None:
        """Reset both CUDA and host freelists (CUDA graph safe: no allocations)."""
        self.free_list_cuda.copy_(self._reset_free_list_cuda)
        self.free_list_head.zero_()
        self.free_list_tail.copy_(self._reset_tail)
        if self._reset_free_list_host is not None:
            self.free_list_host.copy_(self._reset_free_list_host)

    def nbytes(self) -> int:
        """Bytes of CUDA page memory held by this buffer."""
        return self.cuda_buffer.nelement() * self.cuda_buffer.element_size()

    def launch_copy(
        self,
        source: torch.Tensor,
        num_tokens: torch.Tensor,
        page_record: torch.Tensor,
        spilled: torch.Tensor,
        new_free_list_head: torch.Tensor,
        *,
        hidden_size: int,
        num_blocks: int,
    ) -> None:
        """Launch ``paged_stash_copy_kernel`` from ``source`` into free pages."""
        paged_stash_copy_kernel[(num_blocks,)](
            source,
            self.cuda_buffer,
            self._host_or_cuda_buffer(),
            num_tokens,
            self.free_list_cuda,
            self.free_list_host,
            self.free_list_head,
            self.free_list_tail,
            self.free_list_capacity,
            page_record,
            self.overflow,
            self.host_spill,
            spilled,
            new_free_list_head,
            PAGE_SIZE=self.page_size,  # pyrefly: ignore [bad-argument-type]
            HIDDEN_SIZE=hidden_size,  # pyrefly: ignore [bad-argument-type]
            BLOCK_SIZE=GLOBAL_BLOCK_SIZE,  # pyrefly: ignore [bad-argument-type]
            # pyrefly: ignore [bad-argument-type]
            HAS_HOST_BUFFER=1 if self.host_buffer is not None else 0,
        )

    def launch_pop(
        self,
        dest: torch.Tensor,
        num_tokens: torch.Tensor,
        page_record: torch.Tensor,
        spilled: torch.Tensor,
        new_free_list_tail: torch.Tensor,
        *,
        hidden_size: int,
        num_blocks: int,
    ) -> None:
        """Launch ``paged_stash_pop_kernel`` from the recorded pages into ``dest``."""
        paged_stash_pop_kernel[(num_blocks,)](
            self.cuda_buffer,
            self._host_or_cuda_buffer(),
            dest,
            num_tokens,
            page_record,
            spilled,
            self.overflow,
            self.free_list_cuda,
            self.free_list_host,
            self.free_list_tail,
            self.free_list_capacity,
            new_free_list_tail,
            PAGE_SIZE=self.page_size,  # pyrefly: ignore [bad-argument-type]
            HIDDEN_SIZE=hidden_size,  # pyrefly: ignore [bad-argument-type]
            BLOCK_SIZE=GLOBAL_BLOCK_SIZE,  # pyrefly: ignore [bad-argument-type]
        )

    def _host_or_cuda_buffer(self) -> torch.Tensor:
        # The kernels take a host pointer unconditionally; without a host buffer
        # it is never dereferenced, so any valid pointer will do.
        return self.host_buffer if self.host_buffer is not None else self.cuda_buffer

    def __repr__(self) -> str:
        return (
            f"PagedStashBuffer(num_cuda_pages={self.num_cuda_pages}, "
            f"num_host_pages={self.num_host_pages}, "
            f"page_size={self.page_size}, hidden_size={self.hidden_size}, "
            f"device={self.device}, dtype={self.dtype})"
        )


@dataclass(slots=True)
class StashSlot:
    """Static, per-graph-node description of one stashed activation.

    The FX pass assigns each tagged node a slot id and bakes it into the
    ``paged_stash`` op call, so the runtime can look up everything it needs
    without inspecting the tensor. ``max_num_tokens`` is the padded
    (capacity-factor sized) leading dimension; ``token_divisor`` is 1 for plain
    activations and ``SCALE_INV_BLOCK_SIZE`` for MXFP8 columnwise scale
    inverses, matching Megatron's ``is_columnwise_scale_inv`` handling.
    """

    slot_id: int
    dtype: torch.dtype
    hidden_size: int
    max_num_tokens: int
    token_divisor: int = 1
    layer_id: int = -1
    node_name: str = ""
    # Pipeline stage that owns this slot. Sizing needs it because a stashed
    # activation is live exactly as long as its stage's microbatch is in flight,
    # and different stages hold different numbers of microbatches.
    stage_index: int = 0

    @property
    def key(self) -> tuple[torch.dtype, int]:
        return (self.dtype, self.hidden_size)

    @property
    def max_stash_tokens(self) -> int:
        return self.max_num_tokens // self.token_divisor


# Megatron stores 1-byte float8 activations in a uint8 page buffer because
# Triton cannot address float8_e4m3fn/float8_e8m0fnu element types directly.
_BUFFER_DTYPE_OVERRIDE = {
    torch.float8_e4m3fn: torch.uint8,
    torch.float8_e8m0fnu: torch.uint8,
}


def buffer_dtype_for(dtype: torch.dtype) -> torch.dtype:
    """Storage dtype used for a stash buffer holding ``dtype`` activations."""
    return _BUFFER_DTYPE_OVERRIDE.get(dtype, dtype)


# ===========================================================================
# The process-wide stash manager
# ===========================================================================
#
# ``PagedStashManager`` replaces Megatron's manager. The frontends decide where
# activations are stashed and reloaded -- GraphTrainer bakes ops into its traced
# graphs, the eager trainer drives the same calls from module and autograd
# hooks -- and the manager keeps what spans those points: the buffers, the
# shared overflow/host-spill flags, the side stream, the measure-step token
# accounting, the paged-layer schedule, and every stash from its launch until
# the reload that consumes it.


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    """One paged MoE layer's forward or backward for one microbatch.

    Megatron's schedule entry: it encodes ``(vp_stage, layer, microbatch)`` as
    one integer and negates it for the backward.
    """

    forward: bool
    stage_index: int
    layer_id: int
    microbatch: int


@dataclass(slots=True, eq=False)
class StashRecord:
    """One stash of one slot, held from the stash until the reload that
    consumes it. Megatron's ``PagedTensor``.

    ``handle`` is one int64 tensor: ``[:max_pages]`` records which page backs
    each ``page_size``-row chunk of the activation, and the last element is the
    spill flag. ``resident`` is the activation itself when it stayed resident
    because its backward runs next. ``live_rows`` is a copy of just its live
    rows, kept instead of paging on the measure step and while paging is
    disabled for a rerun. ``measured`` is the live token count read then,
    released from the live-page counters by the reload. ``reloaded`` is the
    tensor a reload is filling, once one was launched -- by a prefetch or by
    the reload itself.
    """

    slot: StashSlot
    microbatch: int
    handle: torch.Tensor
    num_tokens: torch.Tensor
    shape: list[int]
    dtype: torch.dtype
    resident: torch.Tensor | None = None
    live_rows: torch.Tensor | None = None
    measured: int | None = None
    reloaded: torch.Tensor | None = None


class PagedStashManager:
    """Process-wide owner of paged stash buffers, streams, and overflow flags.

    Lifecycle mirrors Megatron's ``begin -> capture -> captured`` state machine:

    ``begin``
        Buffers do not exist. ``stash`` runs in measure mode: it keeps the
        activation alive and records its real (device-side) token count,
        exactly like Megatron's capture iteration, which reads
        ``num_tokens_tensor.item()`` before any buffer exists.
    ``capture``
        One full measured step has been observed. ``allocate_stash_buffers``
        turns the recorded peaks into real page buffers.
    ``captured``
        Steady state. ``stash`` pages activations through the buffers.

    TorchTitan differs from Megatron in *where* the peak comes from: Megatron
    accumulates it while walking its virtual-pipeline schedule, while here the
    measure step records each slot's live pages, and GraphTrainer additionally
    replays its GraphPP schedule over them (see ``replay_schedule_page_peak``).
    """

    _INSTANCE: PagedStashManager | None = None

    @classmethod
    def get_instance(cls) -> PagedStashManager:
        if cls._INSTANCE is None:
            cls._INSTANCE = PagedStashManager()
        return cls._INSTANCE

    @classmethod
    def reset_instance(cls) -> None:
        """Drop the manager, if there is one, so the next ``get_instance``
        starts a fresh one.

        Its page buffers are freed here rather than with the last reference to
        it: the ``deferred`` runner's optimizer hooks hold one for as long as
        the optimizers live. As on the fallback path, this must run after the
        CUDA graphs that captured the buffers' pointers are torn down.
        """
        manager = cls._INSTANCE
        if manager is None:
            return
        cls._INSTANCE = None
        if manager.stash_buffers:
            manager.release_stash_buffers()

    def __init__(self) -> None:
        self.enabled = False
        self.status = "begin"  # begin, capture, captured
        # Set for a ``blocking`` rerun, which runs with paging disabled as
        # Megatron's does: ``stash`` keeps activations resident instead of
        # paging them.
        self.paging_disabled = False
        self.device: torch.device | None = None

        self._stream: torch.cuda.Stream | None = None

        self.page_size = 64
        self.buffer_size_factor_cuda = 1.10
        self.buffer_size_factor_cpu = 0.0
        # Pipeline stage whose graph is currently being rewritten, stamped onto
        # each slot GraphTrainer's pass registers. GraphPP sets it per stage;
        # the non-PP joint graph leaves it at 0.
        self.current_stage_index = 0
        # Pages per (dtype, hidden_size) that the GraphPP schedule says must be
        # resident at the pipeline's peak. Empty without GraphPP.
        self.schedule_estimate: dict[tuple[torch.dtype, int], int] = {}

        # (dtype, hidden_size) -> buffer, shared by every slot with that key.
        self.stash_buffers: dict[
            tuple[torch.dtype, int], PagedStashBuffer
        ] | None = None
        # Set whenever buffers are (re)allocated: the Triton kernels must be
        # JIT-compiled outside CUDA graph capture, and a fresh set of buffers
        # can introduce constexpr combinations that were never compiled.
        self.needs_kernel_warmup = False
        self.overflow: torch.Tensor | None = None
        self.host_spill: torch.Tensor | None = None

        # slot_id -> StashSlot, registered by a frontend.
        self.slots: dict[int, StashSlot] = {}
        self._next_slot_id = 0

        # Measure-step accounting, mirroring Megatron's
        # temp_/max_tokens_across_vp_stages pair but in whole pages, since every
        # individual stash rounds up to a page boundary: ``_live_pages`` is the
        # currently stashed page count per (dtype, hidden_size), ``_peak_pages``
        # its running peak.
        self._live_pages: dict[tuple[torch.dtype, int], int] = {}
        self._peak_pages: dict[tuple[torch.dtype, int], int] = {}

        # Largest live token count observed for a *single* stash of each slot.
        # Replaying the GraphPP schedule over these reproduces Megatron's
        # per-schedule-entry accumulation without modeling the virtual pipeline
        # by hand.
        self._slot_tokens: dict[int, int] = {}

        # The schedule, from which ``next_entries`` decides when a stash is
        # skipped and which reloads are prefetched -- Megatron's
        # ``_pp_schedule``. Either GraphTrainer's GraphPP actions, which
        # ``paged_layer_schedule`` expands per paged layer (None without
        # pipeline parallelism), or a schedule the eager trainer recorded while
        # the measure step ran. ``_schedule_cache`` holds the expansion per set
        # of paged layers, which is only known once the graph pass has run.
        self.skip_immediate_backward = False
        self._pp_actions: list[_Action] | None = None
        self._recorded_schedule: tuple[
            list[ScheduleEntry], dict[ScheduleEntry, int]
        ] | None = None
        self._schedule_cache: (
            tuple[
                dict[int, tuple[int, ...]],
                list[ScheduleEntry],
                dict[ScheduleEntry, int],
            ]
            | None
        ) = None

        # Every stash of a slot, in stash order, until its reload: Megatron's
        # `paged_tensors_to_reload`. Records are per slot because with
        # pipeline parallelism a slot is stashed once per in-flight microbatch
        # and reloaded in stash order, so the count of a slot's stashes so far
        # is the microbatch its next stash belongs to. A stage's backward runs
        # the microbatch its latest reload was for.
        self._records: dict[int, deque[StashRecord]] = defaultdict(deque)
        self._num_stashed: dict[int, int] = defaultdict(int)
        self._backward_microbatch: dict[int, int] = {}
        # Copies launched but not yet joined, with what they read: Megatron's
        # `paged_tensors_stash_in_progress`. Holding the activations here is
        # what keeps them alive until a later layer's group start, possibly in
        # a later pipeline action, as Megatron's `_original_tensor` does.
        self._stashes_in_flight: list[tuple[torch.Event, tuple[torch.Tensor, ...]]] = []

        self.step = 0

    @property
    def stream(self) -> torch.cuda.Stream:
        """Side stream for both the stash and the reload copies.

        One stream, as in Megatron: the freelists are unsynchronized mutable
        state, so packing and unpacking must never run concurrently.
        """
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

    # ------------------------------------------------------------------
    # Slot registration
    # ------------------------------------------------------------------
    def next_slot_id(self) -> int:
        """Allocate a slot id. Ids are process-global so stage graphs and
        frontends that register slots independently never collide."""
        slot_id = self._next_slot_id
        self._next_slot_id += 1
        return slot_id

    def register_slot(self, slot: StashSlot) -> None:
        existing = self.slots.get(slot.slot_id)
        if existing is not None and existing != slot:
            raise ValueError(
                f"paged stash slot id {slot.slot_id} reused with different "
                f"metadata: {existing} vs {slot}"
            )
        self.slots[slot.slot_id] = slot

    # ------------------------------------------------------------------
    # Measure-step accounting
    # ------------------------------------------------------------------
    def record_stash(self, slot: StashSlot, num_tokens: int) -> None:
        """Record one measured stash and update the per-key live/peak counters."""
        key = slot.key
        self._live_pages[key] = self._live_pages.get(key, 0) + _num_pages(
            num_tokens, self.page_size
        )
        self._peak_pages[key] = max(self._peak_pages.get(key, 0), self._live_pages[key])
        self._slot_tokens[slot.slot_id] = max(
            self._slot_tokens.get(slot.slot_id, 0), num_tokens
        )

    def record_reload(self, slot: StashSlot, num_tokens: int) -> None:
        """Release one measured stash from the live counters."""
        if slot.key in self._live_pages:
            self._live_pages[slot.key] -= _num_pages(num_tokens, self.page_size)

    # ------------------------------------------------------------------
    # The schedule: skipping, joining, and prefetching
    # ------------------------------------------------------------------
    def set_pipeline_actions(self, actions: list[_Action] | None) -> None:
        """Install this rank's GraphPP actions, or None without pipeline
        parallelism."""
        if actions != self._pp_actions:
            self._pp_actions = actions
            self._schedule_cache = None

    def set_recorded_schedule(self, schedule: list[ScheduleEntry] | None) -> None:
        """Install a schedule recorded while a step ran, as Megatron records
        ``_pp_schedule`` during its capture iteration. It takes precedence over
        pipeline actions."""
        if schedule is None:
            self._recorded_schedule = None
            return
        self._recorded_schedule = (
            list(schedule),
            {entry: i for i, entry in enumerate(schedule)},
        )

    def _paged_layers(self) -> dict[int, tuple[int, ...]]:
        """Each stage's paged layers, in forward order."""
        layers: dict[int, set[int]] = defaultdict(set)
        for slot in self.slots.values():
            layers[slot.stage_index].add(slot.layer_id)
        return {stage: tuple(sorted(ids)) for stage, ids in layers.items()}

    def _schedule(
        self, microbatch: int
    ) -> tuple[list[ScheduleEntry], dict[ScheduleEntry, int]]:
        """This rank's schedule and each entry's position in it."""
        if self._recorded_schedule is not None:
            return self._recorded_schedule
        layers = self._paged_layers()
        if self._pp_actions is None:
            # Without pipeline parallelism a microbatch runs its whole forward,
            # then its whole backward.
            actions = [
                _Action(0, FORWARD, microbatch),
                _Action(0, FULL_BACKWARD, microbatch),
            ]
            schedule = paged_layer_schedule(actions, layers)
            return schedule, {entry: i for i, entry in enumerate(schedule)}
        if self._schedule_cache is None or self._schedule_cache[0] != layers:
            schedule = paged_layer_schedule(self._pp_actions, layers)
            self._schedule_cache = (
                layers,
                schedule,
                {entry: i for i, entry in enumerate(schedule)},
            )
            logger.info(
                "Paged stash: this rank's schedule has %d paged-layer entries "
                "over stages %s",
                len(schedule),
                sorted(layers),
            )
        return self._schedule_cache[1], self._schedule_cache[2]

    def next_entries(self, entry: ScheduleEntry, count: int = 1) -> list[ScheduleEntry]:
        """The ``count`` schedule entries that follow ``entry`` on this rank.

        The answer depends on ``entry``'s place in the schedule, and a CUDA
        graph replays whatever its capture decided. Both frontends capture the
        whole step -- the eager trainer a whole pipeline step, GraphPP every
        pipeline action at its own place -- so each entry replays where it was
        captured and the schedule-dependent answers hold. A stage graph
        captured on its own would replay one place's skips and prefetches for
        every microbatch; neither frontend captures one.
        """
        schedule, index = self._schedule(entry.microbatch)
        position = index.get(entry)
        if position is None:
            return []
        return schedule[position + 1 : position + 1 + count]

    def stash_entry(self, slot: StashSlot) -> ScheduleEntry:
        """The forward entry the next stash of ``slot`` belongs to."""
        return ScheduleEntry(
            True, slot.stage_index, slot.layer_id, self._num_stashed[slot.slot_id]
        )

    def backward_entry(self, slot: StashSlot) -> ScheduleEntry | None:
        """The entry of the running backward of ``slot``'s layer.

        Its microbatch is that of the stage's latest reload, not of ``slot``'s:
        a backward's prefetch runs once the expert MLP's backward has produced
        its input gradient, which reads some of the layer's activations but not
        necessarily all -- a weight gradient may read the rest later.
        """
        microbatch = self._backward_microbatch.get(slot.stage_index)
        if microbatch is None:
            return None
        return ScheduleEntry(False, slot.stage_index, slot.layer_id, microbatch)

    def should_skip_stash(self, entry: ScheduleEntry) -> bool:
        """Whether the next entry is ``entry``'s own backward.

        Megatron's ``current_schedule_layer != -next_schedule_layer`` test: if
        so, the activation would be written to the stash and read straight
        back with no bubble in between, so it stays resident instead
        (``remove_paged_tensor_from_stash``). Only a stage's last paged layer
        can pass it, and only on a forward its backward follows directly --
        every forward without pipeline parallelism, the last stage's in 1F1B.
        """
        if not self.skip_immediate_backward:
            return False
        own_backward = ScheduleEntry(
            False, entry.stage_index, entry.layer_id, entry.microbatch
        )
        return self.next_entries(entry) == [own_backward]

    def add_stash_record(self, slot_id: int, record: StashRecord) -> None:
        self._num_stashed[slot_id] += 1
        self._records[slot_id].append(record)

    def take_stash_record(self, slot: StashSlot, handle: torch.Tensor) -> StashRecord:
        """Pop the oldest stash of ``slot``, which the reload of ``handle``
        consumes."""
        queue = self._records.get(slot.slot_id)
        if not queue or queue[0].handle.data_ptr() != handle.data_ptr():
            raise RuntimeError(
                f"paged stash: the reload of slot {slot.slot_id} "
                f"({slot.node_name}) is not for its oldest stash. Every slot's "
                "stashes must be reloaded in stash order."
            )
        return self.take_record(queue[0])

    def take_record(self, record: StashRecord) -> StashRecord:
        """Remove ``record`` from its slot's pending stashes as its reload
        begins, and note that its stage's backward is now on its microbatch."""
        queue = self._records.get(record.slot.slot_id)
        if queue and queue[0] is record:
            queue.popleft()
        elif queue is not None and record in queue:
            queue.remove(record)
        self._backward_microbatch[record.slot.stage_index] = record.microbatch
        return record

    def add_stash_in_flight(
        self, event: torch.Event, retained: tuple[torch.Tensor, ...]
    ) -> None:
        """Hold what a launched copy reads until ``join_stashes``."""
        self._stashes_in_flight.append((event, retained))

    def join_stashes(self) -> None:
        """Megatron's ``wait_for_stash_to_complete``.

        Orders the compute stream after every copy launched so far and drops
        the activations they read, which frees the padded tensors. Runs at
        each paged layer's group start and backward start, whichever comes
        first, as in Megatron.
        """
        if not self._stashes_in_flight:
            return
        # The copies run in launch order on one stream, so the last one's
        # event covers them all.
        last_event = self._stashes_in_flight[-1][0]
        torch.cuda.current_stream(self.device).wait_event(last_event)
        self._stashes_in_flight.clear()

    def prefetch_after(self, entry: ScheduleEntry, count: int) -> None:
        """Start the reloads of every backward among the ``count`` schedule
        entries after ``entry``.

        Megatron's ``reload_paged_tensors(next_schedule_layer)``. The reloaded
        tensors wait in their stash records until backward reads them, which may
        be in another pipeline action; a stash that stayed resident has nothing
        to reload.
        """
        if count <= 0 or not self.active:
            return
        for upcoming in self.next_entries(entry, count):
            if upcoming.forward:
                continue
            for slot in self.slots.values():
                if (slot.stage_index, slot.layer_id) != (
                    upcoming.stage_index,
                    upcoming.layer_id,
                ):
                    continue
                queue = self._records.get(slot.slot_id)
                if not queue:
                    continue
                record = queue[0]
                if (
                    record.microbatch != upcoming.microbatch
                    or record.resident is not None
                    or record.reloaded is not None
                ):
                    continue
                record.reloaded = _launch_reload(
                    slot, record.handle, record.num_tokens, record.shape, record.dtype
                )

    # ------------------------------------------------------------------
    # Stash and reload
    # ------------------------------------------------------------------
    def stash(
        self,
        slot: StashSlot,
        activation: torch.Tensor,
        num_tokens: torch.Tensor,
        *,
        handle: torch.Tensor,
        prefetch: int = 0,
    ) -> StashRecord:
        """Stash ``activation``'s live rows, recording the pages in ``handle``.

        ``handle`` must be an ``int64`` tensor of ``max_pages + 1`` elements.
        Once the copy completes it holds the page id backing each
        ``page_size``-token chunk, plus a trailing flag recording whether the
        stash landed in the pinned host buffer.

        The copy runs on the side stream, and the manager keeps ``activation``
        alive until the next ``join_stashes``. When the schedule runs this
        layer's backward next, nothing is copied and the activation stays
        resident instead (see ``should_skip_stash``). While not paging -- the
        measure step and a ``blocking`` rerun -- the handle is zeroed, the real
        token count is read off the device and recorded for sizing, and only
        the live rows are kept, as Megatron's capture iteration does: a step
        that does not page needs no more memory than one that does. Unlike
        Megatron, a rerun measures too, so the buffers reallocated after an
        overflow fit the step that overflowed them.

        ``prefetch`` is nonzero on a layer's last stash, where Megatron's group
        commit runs: once the layer's copies are launched, the reloads of every
        backward among the next ``prefetch`` schedule entries start.
        """
        shape, dtype = list(activation.shape), activation.dtype
        if not self.active:
            self.device = activation.device
            count = live_rows = None
            if self.measuring or self.paging_disabled:
                count = int(num_tokens.reshape(-1)[0].item()) // slot.token_divisor
                self.record_stash(slot, count)
                live_rows = activation.reshape(-1, slot.hidden_size)[:count].clone()
            handle.zero_()
            record = StashRecord(
                slot,
                self._num_stashed[slot.slot_id],
                handle,
                num_tokens,
                shape,
                dtype,
                resident=activation if live_rows is None else None,
                live_rows=live_rows,
                measured=count,
            )
            self.add_stash_record(slot.slot_id, record)
            return record

        _check_slot_dims(activation, slot)
        entry = self.stash_entry(slot)
        record = StashRecord(slot, entry.microbatch, handle, num_tokens, shape, dtype)
        if self.should_skip_stash(entry):
            # Megatron's `remove_paged_tensor_from_stash`: no copy, no pages, the
            # activation just stays where it is. Zero the handle so nothing
            # reads a stale page record.
            handle.zero_()
            record.resident = activation
        else:
            _launch_stash(slot, handle, activation, num_tokens)
        self.add_stash_record(slot.slot_id, record)
        self.prefetch_after(entry, prefetch)
        return record

    def materialize(self, record: StashRecord) -> torch.Tensor:
        """The activation ``record`` stashed, for backward to read.

        Hands back the resident activation, the live rows a step that did not
        page kept, padded back to the activation's shape, the tensor a prefetch
        already started filling, or pops the pages now if nothing was
        prefetched. A reload is joined by ``wait_for_copy`` before backward
        reads it. As with a reload, the rows past the live ones are not
        restored: nothing reads them.
        """
        if self.active:
            # Megatron's backward start: copies still in flight -- the last
            # forward's, when no group start has run since -- complete first.
            self.join_stashes()
        if record.measured is not None:
            # Without the release the measured "peak" would be the sum of every
            # stash in the step rather than the largest set alive at one time,
            # which with pipeline parallelism over-sizes the buffers by the
            # microbatch count.
            self.record_reload(record.slot, record.measured)
            record.measured = None
        if record.resident is not None:
            return record.resident
        if record.live_rows is not None:
            live_rows = record.live_rows
            record.live_rows = None
            record.reloaded = torch.empty(
                record.shape, dtype=record.dtype, device=live_rows.device
            )
            record.reloaded.view(-1, record.slot.hidden_size)[
                : live_rows.shape[0]
            ].copy_(live_rows)
        if record.reloaded is None:
            record.reloaded = _launch_reload(
                record.slot,
                record.handle,
                record.num_tokens,
                record.shape,
                record.dtype,
            )
        return record.reloaded

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------
    def stage_page_footprint(self) -> dict[int, dict[tuple[torch.dtype, int], int]]:
        """Pages one in-flight microbatch of each pipeline stage keeps stashed.

        Keyed by stage, then by ``(dtype, hidden_size)``. GraphTrainer replays
        its GraphPP schedule over this table to find the pipeline-wide peak,
        which is the quantity Megatron reconstructs by hand from its
        virtual-pipeline order.
        """
        footprint: dict[int, dict[tuple[torch.dtype, int], int]] = {}
        for slot_id, tokens in self._slot_tokens.items():
            slot = self.slots.get(slot_id)
            if slot is None:
                continue
            stage = footprint.setdefault(slot.stage_index, {})
            stage[slot.key] = stage.get(slot.key, 0) + _num_pages(
                tokens, self.page_size
            )
        return footprint

    def allocate_stash_buffers(self) -> None:
        """Allocate page buffers from the measured peak and the schedule estimate.

        Sizing follows Megatron -- peak concurrent usage per ``(dtype,
        hidden_size)`` scaled by ``buffer_size_factor_cuda``, plus an optional
        pinned host buffer scaled by ``..._cpu`` -- with two changes:

        * Sizing is in **pages**, not tokens. Megatron accumulates raw token
          counts, which under-allocates whenever an individual stash is small
          relative to ``page_size``: four concurrent 16-token stashes need four
          pages, but their 64-token sum fits in one.
        * The measured peak is combined with a **schedule estimate** when one
          is installed: GraphTrainer replays its GraphPP schedule over the
          per-stage footprint, so buffers are sized for the steady-state
          pipeline depth even if the measure step was shallower. The eager
          trainer's measure step runs the real schedule, so its measured peak
          already is that depth.
        """
        if self.device is None:
            raise RuntimeError("paged stash: device is unset; nothing was stashed.")

        if self.overflow is None:
            self.overflow = torch.zeros(1, dtype=torch.int64, device=self.device)
        else:
            self.overflow.zero_()
        if self.host_spill is None:
            self.host_spill = torch.zeros(1, dtype=torch.int64, device=self.device)
        else:
            self.host_spill.zero_()

        if not self._peak_pages:
            logger.info(
                "Paged stash: no activations were stashed during the measure step; "
                "skipping stash buffer allocation."
            )
            self.stash_buffers = {}
            return

        self.needs_kernel_warmup = True
        self.stash_buffers = {}
        total_bytes = 0
        for key, measured_pages in self._peak_pages.items():
            dtype, hidden_size = key
            schedule_pages = self.schedule_estimate.get(key, 0)
            required_pages = max(measured_pages, schedule_pages)
            logger.debug(
                "  paged stash sizing %s/%d: measured=%d pages, schedule=%d pages",
                dtype,
                hidden_size,
                measured_pages,
                schedule_pages,
            )

            buffer = PagedStashBuffer(
                int(required_pages * self.buffer_size_factor_cuda) * self.page_size,
                hidden_size,
                self.page_size,
                self.device,
                self.overflow,
                self.host_spill,
                buffer_dtype_for(dtype),
                num_tokens_host=(
                    int(required_pages * self.buffer_size_factor_cpu) * self.page_size
                ),
            )
            self.stash_buffers[key] = buffer
            total_bytes += buffer.nbytes()
            message = (
                f"Paged stash: allocated cuda {tuple(buffer.cuda_buffer.shape)} "
                f"pages={buffer.num_cuda_pages}"
            )
            if buffer.host_buffer is not None:
                message += f" host {tuple(buffer.host_buffer.shape)}"
            message += f" dtype={buffer.dtype} ({dtype})"
            logger.info(message)

        logger.info(
            "Paged stash: %d buffer(s), %.2f MiB of CUDA pages "
            "(page_size=%d, factor=%.2f)",
            len(self.stash_buffers),
            total_bytes / (1024 * 1024),
            self.page_size,
            self.buffer_size_factor_cuda,
        )

    def release_stash_buffers(self) -> None:
        """Drop the large page buffers, keeping the small shared flags.

        Called on the fallback path, matching Megatron's
        ``PagedStashManager.release_stash_buffers``, and by ``reset_instance``;
        both after the CUDA graphs that reference these pointers are torn down.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.stash_buffers = None
        logger.info("Paged stash: released stash page buffers.")

    def get_buffer(self, dtype: torch.dtype, hidden_size: int) -> PagedStashBuffer:
        if self.stash_buffers is None:
            raise RuntimeError("paged stash: buffers have not been allocated yet.")
        try:
            return self.stash_buffers[(dtype, hidden_size)]
        except KeyError:
            raise RuntimeError(
                f"paged stash: no buffer for dtype={dtype} hidden_size={hidden_size}. "
                f"Known: {list(self.stash_buffers)}"
            ) from None

    def reset_buffers(self) -> None:
        """Restore every freelist and clear the spill flag, allocating nothing.

        The overflow flag is sticky: an overflow is fatal, and only the
        ``blocking`` rerun clears it. The deferred
        check reads a copy taken on an earlier step, and the CPU can run several
        steps ahead of that copy; if each step cleared the flag, a later copy
        could overwrite an unread one and lose the overflow.
        """
        assert self.stash_buffers is not None
        for buffer in self.stash_buffers.values():
            buffer.reset()
        assert self.overflow is not None and self.host_spill is not None
        self.host_spill.zero_()

    # ------------------------------------------------------------------
    # Per-step lifecycle
    # ------------------------------------------------------------------
    def begin_step(self) -> None:
        """Advance the state machine and reset per-step state.

        Mirrors Megatron's ``paged_stash_reset``, which runs at the top of every
        ``forward_backward_func`` call.
        """
        if not self.enabled:
            return
        self.step += 1
        self._records.clear()
        self._num_stashed.clear()
        self._backward_microbatch.clear()
        self._stashes_in_flight.clear()
        self._live_pages = {key: 0 for key in self._live_pages}
        if self.paging_disabled:
            return

        if self.status == "begin":
            self.status = "capture"
        elif self.status == "capture":
            self.status = "captured"
            self.allocate_stash_buffers()
        elif self.status == "captured" and self.stash_buffers is None:
            # Buffers were released by a fallback; reallocate from the same
            # measured peaks.
            self.allocate_stash_buffers()

        if self.status == "captured" and self.stash_buffers:
            self.reset_buffers()

    @property
    def measuring(self) -> bool:
        """True on the measure step, which records token counts for sizing."""
        return self.enabled and not self.paging_disabled and self.status == "capture"

    @property
    def active(self) -> bool:
        """True when stashes page through real buffers."""
        return (
            self.enabled
            and not self.paging_disabled
            and self.status == "captured"
            and bool(self.stash_buffers)
        )


def _flag_is_set(flag: torch.Tensor | None) -> torch.Tensor:
    manager = PagedStashManager.get_instance()
    if not manager.enabled or flag is None:
        device = manager.device or torch.device("cuda")
        return torch.zeros(1, dtype=torch.bool, device=device)
    return flag.ne(0)


def check_paged_stash_overflow() -> torch.Tensor:
    """Device-side flag: did any stash fail to find a free page this step?"""
    return _flag_is_set(PagedStashManager.get_instance().overflow)


def check_paged_stash_host_spill() -> torch.Tensor:
    """Device-side flag: did any activation spill to pinned host memory?"""
    return _flag_is_set(PagedStashManager.get_instance().host_spill)


# ===========================================================================
# Stash and reload primitives
# ===========================================================================
#
# Megatron's ``PagedTensor.offload_to_stash`` / ``reload_from_stash``. All
# stash and reload copies run on a single side stream: Megatron shares one
# stream between pack and unpack because the freelist is unsynchronized mutable
# state, and that invariant is preserved here. Destinations are allocated on the
# compute stream and filled on the side stream, matching both Megatron's
# ``reload_paged_tensors`` and upstream ``ao::reload``.


# Megatron caps the copy grid at 2048 programs; the kernels stride over tokens
# so this only bounds parallelism, not correctness.
_MAX_BLOCKS = 2048

# --- Wait registry: data_ptr() -> (completion event, device, retained inputs).
#
# Same shape as ao's registry, plus the ``retain`` tuple, which exists because
# the reload kernels run on a side stream while their *inputs* are owned by the
# compute stream. A frontend drops them as soon as the reload is issued -- FX
# emits a ``del`` for a value right after its last graph use -- so without an
# extra reference the caching allocator would hand ``page_record`` / ``spilled``
# / ``num_tokens`` to the next compute-stream allocation while the pop kernel is
# still reading them. Megatron never hits this because its ``PagedTensor`` keeps
# those tensors alive for the whole iteration; retaining them until the
# matching ``wait_for_copy`` reproduces that lifetime. A stash's inputs are held the same way, by the
# manager, until a group start or a backward joins the copy.
_wait_registry: dict[
    int, tuple[torch.Event, torch.device, tuple[torch.Tensor, ...]]
] = {}


def _register_wait(
    tensor: torch.Tensor,
    device: torch.device,
    retain: tuple[torch.Tensor, ...] = (),
) -> torch.Event:
    """Record the event ``tensor``'s pending copy will signal, and what it reads.

    ``tensor`` is retained alongside ``retain`` -- not for its own sake, but to
    keep the key valid. The registry is addressed by device pointer, which is
    only unique while that address stays claimed: drop the last reference and
    the caching allocator hands the same address to the next allocation, whose
    registration then overwrites a live entry and releases its retained inputs
    while the copy kernel is still reading them. Holding the keyed tensor makes
    "a key is the address of a tensor the registry keeps alive" an invariant
    rather than an assumption.
    """
    key = tensor.data_ptr()
    if key in _wait_registry:
        raise RuntimeError(
            f"paged stash: a wait is already registered at address {key:#x}. "
            "Every reload must be paired with exactly one wait_for_copy before "
            "the next use of that buffer."
        )
    event = torch.Event()
    _wait_registry[key] = (event, device, retain + (tensor,))
    return event


def wait_for_copy(tensor: torch.Tensor) -> None:
    """Join the compute stream to ``tensor``'s pending copy.

    Once the compute stream is ordered after the copy kernel, the inputs the
    registry retained for it may go back to the allocator. A no-op in measure
    mode, where nothing was copied.
    """
    registered = _wait_registry.pop(tensor.data_ptr(), None)
    if registered is None:
        return
    completion_event, device, _retained = registered
    torch.cuda.current_stream(device).wait_event(completion_event)


def _check_slot_dims(activation: torch.Tensor, slot: StashSlot) -> int:
    """Validate that ``activation`` really has the slot's row layout.

    The kernels address the activation as ``view(-1, hidden_size)`` and index a
    prefix of at most ``max_stash_tokens`` rows. If the slot's row layout does not
    describe this tensor, both of those are silently wrong and the copy walks
    off the end of the buffer -- an illegal access far from its cause. Element
    counts alone are ambiguous, so check them here where the real tensor is in
    hand and the numbers can be named in the error.
    """
    numel = activation.numel()
    if numel % slot.hidden_size:
        raise RuntimeError(
            f"paged stash slot {slot.slot_id} ({slot.node_name}): tensor with "
            f"{numel} elements is not a whole number of {slot.hidden_size}-wide "
            "rows; the slot's row layout does not describe this tensor."
        )
    rows = numel // slot.hidden_size
    if rows > slot.max_stash_tokens:
        raise RuntimeError(
            f"paged stash slot {slot.slot_id} ({slot.node_name}): tensor has "
            f"{rows} rows of {slot.hidden_size} but the slot was sized for at "
            f"most {slot.max_stash_tokens} (max_num_tokens="
            f"{slot.max_num_tokens}, token_divisor={slot.token_divisor}). "
            "Paging it would write past the page record."
        )
    return rows


def _num_tokens_arg(num_tokens: torch.Tensor, slot: StashSlot) -> torch.Tensor:
    """Shape the device-side token count the way the Triton kernels expect.

    Megatron reshapes a 0-dim count to ``(1,)`` and divides by
    ``SCALE_INV_BLOCK_SIZE`` for columnwise scale inverses; ``token_divisor``
    generalizes that second step.
    """
    if num_tokens.dim() == 0:
        num_tokens = num_tokens.reshape(1)
    if slot.token_divisor != 1:
        num_tokens = num_tokens // slot.token_divisor
    return num_tokens


def _handle_views(
    handle: torch.Tensor, max_pages: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the handle into its page-record and spill-flag views.

    Megatron keeps these as two tensors on the ``PagedTensor``. One tensor is
    used here so a stash has a single handle to record into -- GraphTrainer's
    forward stash needs one mutated base for GraphPP to save -- and so backward
    carries one saved value per activation instead of two.
    """
    return handle[:max_pages], handle[max_pages : max_pages + 1]


def _launch_stash(
    slot: StashSlot,
    handle: torch.Tensor,
    activation: torch.Tensor,
    num_tokens: torch.Tensor,
) -> None:
    """Copy ``activation``'s live rows into pages on the side stream, recording
    them in ``handle``; the manager holds what the copy reads until it is
    joined."""
    manager = PagedStashManager.get_instance()
    device = activation.device
    buffer = manager.get_buffer(slot.dtype, slot.hidden_size)
    current_stream = torch.cuda.current_stream(device)
    page_record, spilled = _handle_views(handle, handle.numel() - 1)

    # Derive the token count *before* the stream join. For a scale slot
    # ``token_divisor != 1``, so this enqueues a division on the compute
    # stream, and ``wait_stream`` only covers work already enqueued when it
    # runs -- ordering it afterwards lets the copy kernel read the count before
    # the division has written it. Data slots divide by 1 and launch nothing,
    # which is why only the MXFP8 scale slots ever raced.
    tokens = _num_tokens_arg(num_tokens, slot)

    stream = manager.stream
    stream.wait_stream(current_stream)
    completion_event = torch.Event()
    with torch.cuda.stream(stream):
        # Freelist scratch never leaves the side stream, so it must be
        # allocated here: allocating on the compute stream and then freeing it
        # at op exit would let the allocator hand the block to another
        # compute-stream tensor while the copy kernel is still reading it.
        new_free_list_head = torch.empty(2, dtype=torch.int64, device=device)
        handle.zero_()
        source = activation.contiguous().view(-1, slot.hidden_size)
        buffer.launch_copy(
            source.view(buffer.cuda_buffer.dtype),
            tokens,
            page_record,
            spilled,
            new_free_list_head,
            hidden_size=slot.hidden_size,
            num_blocks=min(slot.max_stash_tokens, _MAX_BLOCKS),
        )
        buffer.free_list_head.copy_(new_free_list_head)
        stream.record_event(completion_event)
    manager.add_stash_in_flight(completion_event, (activation, num_tokens, tokens))


def _launch_reload(
    slot: StashSlot,
    handle: torch.Tensor,
    num_tokens: torch.Tensor,
    size: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pop ``handle``'s pages into a new tensor on the side stream.

    The destination is allocated on the compute stream (so the caching
    allocator owns it there) and filled on the side stream, matching both
    Megatron's ``reload_paged_tensors`` and upstream ``ao::reload``;
    ``wait_for_copy`` joins the copy.
    """
    manager = PagedStashManager.get_instance()
    device = handle.device
    buffer = manager.get_buffer(slot.dtype, slot.hidden_size)
    current_stream = torch.cuda.current_stream(device)
    page_record, spilled = _handle_views(handle, handle.numel() - 1)

    result = torch.empty(size, dtype=dtype, device=device)

    # Before the join, for the reason given in ``_launch_stash``.
    tokens = _num_tokens_arg(num_tokens, slot)

    stream = manager.stream
    stream.wait_stream(current_stream)

    # ``handle`` and the token count are dead in the graph the moment the
    # reload returns, but the pop kernel may not have run yet.
    completion_event = _register_wait(
        result, device, retain=(handle, page_record, spilled, num_tokens, tokens)
    )

    with torch.cuda.stream(stream):
        # Side-stream-local scratch; see the note in ``_launch_stash``.
        new_free_list_tail = torch.empty(2, dtype=torch.int64, device=device)
        buffer.launch_pop(
            result.view(-1, slot.hidden_size).view(buffer.cuda_buffer.dtype),
            tokens,
            page_record,
            spilled,
            new_free_list_tail,
            hidden_size=slot.hidden_size,
            num_blocks=min(slot.max_stash_tokens, _MAX_BLOCKS),
        )
        buffer.free_list_tail.copy_(new_free_list_tail)
        stream.record_event(completion_event)

    return result


def warmup_paged_stash_kernels() -> None:
    """JIT-compile both Triton kernels for every live constexpr combination.

    Triton compiles on first launch, which is illegal inside CUDA graph
    capture. Megatron never hits this because its stash runs for real during the
    eager warmup iterations that precede capture; GraphTrainer's warmup step is
    the *measure* step, which launches no kernels at all, so the first real
    launch would land inside the capture. Priming the cache here removes that
    hazard.

    Each launch passes ``num_tokens=0``, so the kernels compile and then exit
    without touching pages or freelists.
    """
    manager = PagedStashManager.get_instance()
    if not manager.needs_kernel_warmup or not manager.stash_buffers:
        return

    device = manager.device
    zero_tokens = torch.zeros(1, dtype=torch.int64, device=device)
    for buffer in manager.stash_buffers.values():
        scratch = torch.zeros(
            (buffer.page_size, buffer.hidden_size), dtype=buffer.dtype, device=device
        )
        page_record = torch.zeros(1, dtype=torch.int64, device=device)
        spilled = torch.zeros(1, dtype=torch.int64, device=device)
        new_free_list_ends = torch.zeros(2, dtype=torch.int64, device=device)
        for launch in (buffer.launch_copy, buffer.launch_pop):
            launch(
                scratch,
                zero_tokens,
                page_record,
                spilled,
                new_free_list_ends,
                hidden_size=buffer.hidden_size,
                num_blocks=1,
            )

    # The warmup launches are no-ops, but reset anyway so the step starts from
    # a pristine freelist regardless of kernel-side bookkeeping.
    manager.reset_buffers()
    torch.cuda.synchronize(device)
    manager.needs_kernel_warmup = False
    logger.info(
        "Paged stash: compiled Triton kernels for %d buffer configuration(s).",
        len(manager.stash_buffers),
    )


# ===========================================================================
# Which activations the stash can hold
# ===========================================================================
#
# An activation is pageable when its leading dimension is the *capacity-padded*
# permuted-token dimension of a static-capacity token dispatch and only the
# first ``sum(tokens_per_expert)`` rows of it are live. That padded-vs-live gap
# is exactly what paged stashing reclaims. The frontends apply the same rule:
# GraphTrainer to traced node values, the eager trainer to the tensors autograd
# saves.


# Default module-FQN pattern of the paged-stash region: the routed experts, so
# the region holds the dispatcher's output and everything the expert MLP saves.
DEFAULT_MODULE_FQN = "layers.*.moe.routed_experts"

# Paging a tensor costs one full round trip through the stash, so tiny tensors
# are never worth it. Same rationale as GraphTrainer cpu_offload's
# ``_MIN_OFFLOAD_BYTES``.
MIN_STASH_BYTES = 1 << 20  # 1 MiB


def worth_paging(nbytes: int, token_divisor: int) -> bool:
    """Whether a tensor of ``nbytes`` with ``token_divisor`` is worth a round
    trip.

    A block-scale buffer is 1/32 of the tensor it describes and is only ever
    paged alongside it -- Transformer Engine marks the pair together and
    Megatron stashes a layer's tensors as one list. Judging it on its own bytes
    would reject it for being small when the round trip it shares is already
    paid for, so it is weighed by the data it scales.
    """
    return nbytes * token_divisor >= MIN_STASH_BYTES


def infer_slot_dims(val: torch.Tensor, padded: int) -> tuple[int, int] | None:
    """``(token_divisor, hidden_size)`` -- how the stash addresses this tensor as
    rows -- or ``None`` if it cannot be paged.

    Megatron does not require a saved tensor to be two-dimensional. It flattens
    it and divides the element count by the padded row count -- ``max_num_tokens``
    for data, ``max_num_tokens // 32`` for a columnwise scale inverse -- and the
    copy kernel then addresses a prefix of those rows. That is what lets it page
    Transformer Engine's scale buffers, which are flat 1-D arrays with no token
    axis to test.

    Data tensors are held to the stricter rule: their leading dimension must be
    the token count. A tensor whose rows are not tokens has its live values
    strided through the buffer rather than sitting in a prefix, which is
    precisely the case the copy kernel cannot express.

    Scale inverses are recognized by dtype. Megatron learns that a saved tensor
    is one from the ``grouped_tensor_scale_inv`` flag Transformer Engine stamps
    on it; nothing marks ours, but e8m0 is only ever used for MXFP8 block
    scales, so the dtype carries the same information.
    """
    if val.dtype is torch.float8_e8m0fnu:
        # 32 tokens share one scale row, so the row count is padded // 32.
        if padded % SCALE_INV_BLOCK_SIZE:
            return None
        divisor, rows = SCALE_INV_BLOCK_SIZE, padded // SCALE_INV_BLOCK_SIZE
        if val.dim() > 1 and int(val.shape[0]) != rows:
            # A shaped buffer whose rows are not 32-token blocks may be stored
            # column by column, each column's live tokens strided through it
            # rather than sitting in a prefix, and paging a prefix of that
            # would restore the wrong values. A flat buffer has no rows to say
            # so and gets the token reading, as Transformer Engine's flat
            # buffers do in Megatron. The frontends also try any view that
            # covers the same bytes, so a flat view of it can still read as
            # token rows.
            return None
    else:
        if int(val.shape[0]) != padded:
            return None
        divisor, rows = 1, padded
    if rows <= 0 or val.numel() % rows:
        return None
    return divisor, val.numel() // rows


# ===========================================================================
# The paged-layer schedule and schedule-replay sizing
# ===========================================================================
#
# Megatron sizes its stash buffers by reconstructing its own virtual-pipeline
# order: ``PagedStashManager`` builds a ``_pp_schedule`` list while a capture
# iteration runs, encodes ``(vp_stage, layer, microbatch)`` triples into integers,
# and accumulates per-tensor token counts over the entries that are live at the
# same time. The peak of that accumulation is the buffer size.
#
# A PyTorch pipeline schedule is already a concrete list of ``_Action`` s per
# rank, so the same accumulation can be run directly against it: walk the
# actions, add a stage's stash footprint when a microbatch enters forward,
# subtract it when the backward that consumes that stage's saved values runs, and
# keep the peak.
#
# A stashed activation becomes live at ``FORWARD(stage, microbatch)`` and is
# released by the backward that consumes the stage's saved values:
# ``FULL_BACKWARD`` or, for split schedules, ``BACKWARD_INPUT``.
# ``BACKWARD_WEIGHT`` consumes a different set of live values and does not release
# the stash, so it is not counted.


PageCounts = dict[tuple[torch.dtype, int], int]


def _iter_compute_actions(action: _Action):
    """Yield ``action`` itself, or its sub-actions for a multiplexed action."""
    if action.sub_actions:
        yield from action.sub_actions
    else:
        yield action


def replay_schedule_page_peak(
    actions: list[_Action],
    stage_footprint: dict[int, PageCounts],
) -> PageCounts:
    """Peak stashed pages per ``(dtype, hidden_size)`` for one rank's schedule.

    Args:
        actions: The rank's ordered pipeline actions.
        stage_footprint: Pages one in-flight microbatch of each stage keeps
            stashed, from ``PagedStashManager.stage_page_footprint``.

    Returns:
        Peak concurrently stashed pages per key.
    """
    live: PageCounts = {}
    peak: PageCounts = {}

    def _apply(stage_index: int, sign: int) -> None:
        counts = stage_footprint.get(stage_index)
        if not counts:
            return
        for key, pages in counts.items():
            live[key] = live.get(key, 0) + sign * pages
            if sign > 0:
                peak[key] = max(peak.get(key, 0), live[key])

    for action in actions:
        for sub in _iter_compute_actions(action):
            kind = sub.computation_type
            if kind is FORWARD:
                _apply(sub.stage_index, +1)
            elif kind in (FULL_BACKWARD, BACKWARD_INPUT):
                _apply(sub.stage_index, -1)
    return peak


def max_live_stash_generations(actions: list[_Action]) -> int:
    """Peak number of concurrently stashed microbatches for one rank's schedule.

    Reported for logging; the actual sizing uses ``replay_schedule_page_peak``,
    which is per-stage and therefore tighter than this rank-wide count.
    """
    live = 0
    peak = 0
    for action in actions:
        for sub in _iter_compute_actions(action):
            kind = sub.computation_type
            if kind is FORWARD:
                live += 1
                peak = max(peak, live)
            elif kind in (FULL_BACKWARD, BACKWARD_INPUT):
                live = max(0, live - 1)
    return peak


def paged_layer_schedule(
    actions: list[_Action], layers: Mapping[int, Sequence[int]]
) -> list[ScheduleEntry]:
    """Megatron's ``_pp_schedule`` for one rank: every paged layer's forward
    and backward, in the order the rank runs them.

    Megatron records it while its capture iteration runs, one entry per MoE
    layer of every microbatch's forward and backward, and decides from the
    entry after the current one whether to skip a stash (its own backward) and
    which reloads to start (a backward). A pipeline action list already is the
    execution order, so each compute action of a stage with paged layers
    expands into those layers -- ascending for a forward, descending for a
    backward. Stages without paged layers add nothing, as dense layers add
    nothing to Megatron's; sends, receives and unshards stash nothing, and
    neither does a ``BACKWARD_WEIGHT``, which consumes a different set of saved
    values -- the reason :func:`replay_schedule_page_peak` ignores it too.

    Args:
        actions: The rank's ordered pipeline actions.
        layers: Each stage's paged layers, in forward order.

    Returns:
        The rank's schedule entries, in execution order.

    Raises:
        ValueError: If a paged stage's forward has no backward that consumes
            its saved values, e.g. because ``actions`` name their backwards
            otherwise. Sizing would then keep those stashes live to the end
            of the step, and nothing would be skipped or prefetched.
    """
    schedule: list[ScheduleEntry] = []
    for action in actions:
        for sub in _iter_compute_actions(action):
            stage_layers = layers.get(sub.stage_index)
            if not stage_layers:
                continue
            if sub.computation_type is FORWARD:
                forward, order = True, stage_layers
            elif sub.computation_type in (FULL_BACKWARD, BACKWARD_INPUT):
                forward, order = False, list(reversed(stage_layers))
            else:
                continue
            microbatch = sub.microbatch_index
            assert microbatch is not None, f"{sub} runs no microbatch"
            schedule.extend(
                ScheduleEntry(forward, sub.stage_index, layer, microbatch)
                for layer in order
            )
    forwards = {(e.stage_index, e.microbatch) for e in schedule if e.forward}
    backwards = {(e.stage_index, e.microbatch) for e in schedule if not e.forward}
    if forwards != backwards:
        raise ValueError(
            "Paged stash: every forward of a paged stage needs a FULL_BACKWARD "
            "or BACKWARD_INPUT of the same microbatch in the schedule; "
            f"unmatched (stage, microbatch): {sorted(forwards ^ backwards)}"
        )
    return schedule


# ===========================================================================
# Overflow detection and recovery around one forward/backward step
# ===========================================================================
#
# Port of Megatron-Core's ``PagedStashRunner``. The detection half is identical:
# the Triton kernels raise a device-side ``overflow`` flag when neither the CUDA
# nor the pinned-host freelist can satisfy a stash, and both kernels then become
# no-ops so nothing is corrupted further.
#
# **Reading the verdict.** Megatron combines the flags across ranks with one
# all-reduce and calls ``.item()`` on them at every step boundary. That is a
# device sync on a value that is almost always zero: it forfeits CPU run-ahead
# and turns each step's CPU work into a GPU idle bubble. ``deferred``, the
# default, instead copies each rank's own flags into pinned host memory
# asynchronously and reads the copy on a later step once ``Event.query()`` says
# it landed, so no step syncs. ``blocking`` keeps Megatron's all-reduce and
# synchronous read, since every rank must agree to rerun the step.
#
# **Recovering.** ``blocking`` follows Megatron's ``PagedStashRunner``: discard
# the step's gradients, drop the CUDA graph, release the pages, and rerun the
# step once with paging disabled, which cannot overflow; the next step
# reallocates the buffers and pages again. Megatron disables paging by switching
# off ``moe_paged_stash``, a flag its eager forward reads. Here the manager
# disables it: every stash keeps its activation resident, as in the measure
# step, and is measured as in the measure step, so the reallocated buffers fit
# the step that overflowed them. Megatron reallocates the same size, so once
# routing has outgrown its capture iteration, every step overflows and reruns
# until it changes back. Megatron also reruns a step whose dispatch
# dropped tokens, with the capacity factor cleared so the dispatch sizes itself;
# here the capacity is part of what is captured -- GraphTrainer's compiled graph,
# the eager trainer's CUDA graph -- so a drop is reported instead.
#
# Overflow means backward read activations that were never written back, so the
# step's gradients are invalid. Neither mode ever applies them:
#
# ``deferred`` (default)
#     No per-step sync. Overflow is fatal and rank-local: a rank raises when it
#     reads its own verdict, typically two steps after the overflow, and the
#     other ranks exit when the job is torn down. The overflow flag is sticky --
#     ``reset_buffers`` leaves it set -- so however far the CPU runs ahead of the
#     copies, a later one still carries it, and a blocking read at shutdown
#     catches an overflow on the final steps, which no later step reads.
#
#     The steps until the raise are already enqueued, so they are skipped on
#     device instead, the way AMP skips a step whose gradients hold an inf.
#     Every optimizer step all-reduces the flag and hands it to the optimizer as
#     ``found_inf``, which makes fused Adam and AdamW leave the parameters and
#     their state untouched; the expert biases the step's MoE load-balancing
#     hook moved are put back. All ranks skip, not just the one that
#     overflowed, because FSDP's reduce-scatter and EP's all-to-all carry its
#     gradients to every rank. A checkpoint saved meanwhile is therefore the
#     last state before the overflow, with the step count, learning-rate
#     schedule and data position advanced, as AMP's skipped steps leave them.
#     Optimizers without ``found_inf`` support cannot skip a step on device, so
#     they are only accepted in ``blocking`` mode.
#
#     TODO: a pop after an overflow copies nothing, so backward reads
#     uninitialized rows. When those make the gradient norm non-finite, the
#     trainer's non-finite guard ends the job with a device-side assert before
#     this check raises, and the user sees that error instead of the paged
#     stash one. Zeroing the skipped rows would keep the step finite until the
#     overflow error is raised.
# ``blocking``
#     The verdict is known before the step returns, so an overflowing step is
#     rerun with paging disabled. Recovers in process and works with any
#     optimizer; costs one device sync per step.


@dataclass(frozen=True, slots=True)
class OverbudgetProbe:
    """A token dispatcher's device-side "dropped tokens this step" flag.

    Reported with the verdict as the ``overbudget`` slot of Megatron's
    ``check_moe_overflow``, which Megatron fills from
    ``token_dispatcher.check_over_budget()``. ``reset`` clears the flag at the
    start of every attempt, so each verdict covers exactly one step.
    """

    read: Callable[[], torch.Tensor | None]
    reset: Callable[[], None]


def skips_step_on_found_inf(optimizer: torch.optim.Optimizer) -> bool:
    """Whether ``optimizer.step()`` leaves the parameters and their state
    untouched when AMP's ``found_inf`` attribute is 1, as ``GradScaler``
    relies on: true for fused Adam and AdamW.

    Checked per param group because torchtitan sets ``fused`` there, which
    leaves ``_step_supports_amp_scaling`` -- set only for a constructor-level
    ``fused=True`` -- unset.
    """
    return isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)) and all(
        group["fused"] for group in optimizer.param_groups
    )


def moe_expert_biases(model_parts: Sequence[nn.Module]) -> list[torch.Tensor]:
    """The MoE expert biases, as local tensors that write through to them.

    Besides the parameters, the optimizer step moves these:
    ``register_moe_load_balancing_hook`` adds to them in a step pre-hook. They
    are found the way that hook finds them, MTP decoder blocks included.
    """
    biases = []
    for model_part in model_parts:
        layer_containers = [model_part.get_submodule("layers")]
        mtp_layers = getattr(model_part, "mtp_layers", None)
        if mtp_layers is not None:
            layer_containers.append(mtp_layers)
        for layers in layer_containers:
            for block in layers.children():
                if not getattr(block, "moe_enabled", False):
                    continue
                bias = cast(nn.Module, block.moe).expert_bias_E
                if bias is not None:
                    assert isinstance(bias, torch.Tensor)
                    # A buffer, so no gradient flows through ``to_local``.
                    biases.append(
                        bias.to_local() if isinstance(bias, DTensor) else bias
                    )
    return biases


class PagedStashRunner:
    """Run one forward/backward step with paged-stash overflow recovery.

    Sits exactly where Megatron's ``PagedStashRunner`` sits: wrapped around the
    call that runs a whole schedule of microbatches, so a single reset and a
    single overflow verdict cover the microbatches that share the stash buffers.

    Args:
        optimizers: The trainer's optimizers. In ``deferred`` mode the runner
            hooks their step to skip it while a stash overflow is flagged, so
            every one of them must be able to skip a step on device (see
            ``skips_step_on_found_inf``). ``blocking`` mode reruns an
            overflowing step instead and leaves them alone.
        expert_biases: Tensors the optimizer step updates besides the
            parameters (see ``moe_expert_biases``). A skipped step puts them
            back.
        overbudget_probes: Token-drop flags of the model's dispatchers, reported
            with the verdict.
        max_attempts: Attempts per step in ``blocking`` mode, including the
            first. Megatron asserts on a third attempt; the same bound applies
            here.
        process_group: Group for the flag all-reduce. Defaults to WORLD, which
            is what Megatron uses.
        overflow_check: ``deferred`` for the non-blocking pinned-mirror read,
            which skips the optimizer steps after an overflow and fails the job;
            ``blocking`` for Megatron's synchronous per-step read, which reruns
            an overflowing step.
        reset_cuda_graphs: Called before a ``blocking`` rerun to drop every CUDA
            graph that captured the stash buffers' pointers, so the rerun can
            run eagerly and the next step can capture again. Megatron's
            ``reset_cuda_graph``.
        config_path: Where ``PagedStashConfig`` lives in the job config, for
            messages that name its fields.
        disable_hint: How to turn paged stashing off, for the overflow error.
    """

    def __init__(
        self,
        optimizers: OptimizersContainer,
        *,
        expert_biases: Sequence[torch.Tensor] = (),
        overbudget_probes: Sequence[OverbudgetProbe] = (),
        max_attempts: int = 2,
        process_group: torch.distributed.ProcessGroup | None = None,
        overflow_check: OverflowCheckMode = "deferred",
        reset_cuda_graphs: Callable[[], None] | None = None,
        config_path: str = "paged_stash",
        disable_hint: str = "paged_stash = None",
    ) -> None:
        if overflow_check not in ("deferred", "blocking"):
            raise ValueError(
                "paged stash: overflow_check must be 'deferred' or 'blocking', "
                f"got {overflow_check!r}."
            )
        self.manager = PagedStashManager.get_instance()
        self.overbudget_probes = tuple(overbudget_probes)
        self.max_attempts = max_attempts
        self.process_group = process_group
        self.overflow_check = overflow_check
        self._optimizers = optimizers
        self._reset_cuda_graphs = reset_cuda_graphs
        self._config_path = config_path
        self._disable_hint = disable_hint
        if overflow_check == "deferred":
            self._skip_overflowed_steps(optimizers, expert_biases)

        # Deferred-read state. Allocated up front because ``cudaHostAlloc``
        # synchronizes: doing it lazily on the first captured step would put one
        # surprise sync in the training path, which is exactly what this mode
        # exists to avoid. ``_pending_step`` is the step the in-flight copy
        # covers, which is what the overflow error names.
        self._flags_host = torch.zeros(
            3, dtype=torch.int32, device="cpu", pin_memory=True
        )
        self._flags_event = torch.cuda.Event()
        # Device-side scratch for the all-reduced flags, bound on first use and
        # then reused. A fresh allocation of a size the caching allocator has
        # not seen calls ``cudaMalloc``, which synchronizes the device -- so
        # building this tensor per step would reintroduce the very stall the
        # deferred read removes.
        self._flags_device: torch.Tensor | None = None
        self._pending_step: int | None = None
        if self.manager.device is not None:
            self._ensure_flag_scratch(self.manager.device)
            self._warmup_flag_path()

    def _warmup_flag_path(self) -> None:
        """Run the flag copies once so no training step pays their first launch.

        Lazy CUDA module loading resolves a kernel on its first launch, and that
        resolution synchronizes. Paying it here, before training starts, is what
        makes ``deferred`` mode sync-free for *every* step rather than every
        step after the first. The collective is deliberately skipped: it is not
        what stalls, and a constructor is the wrong place for one.
        """
        flags = self._fill_local_flags()
        self._flags_host.copy_(flags, non_blocking=True)
        self._flags_event.record()
        self._flags_event.synchronize()

    # ------------------------------------------------------------------
    # Flag plumbing
    # ------------------------------------------------------------------
    def _ensure_flag_scratch(self, device: torch.device) -> torch.Tensor:
        if self._flags_device is None:
            self._flags_device = torch.zeros(3, dtype=torch.int32, device=device)
        return self._flags_device

    def _fill_local_flags(self) -> torch.Tensor:
        """Write this rank's ``(overflow, overbudget, host_spill)`` into scratch.

        Allocates nothing: the kernels store exactly 0 or 1, so the raw flags
        can be copied straight in, whereas ``check_paged_stash_*`` would build a
        comparison temporary every step. Those helpers stay as the public API
        and as the fallback for a manager without allocated flags.
        """
        manager = self.manager
        if manager.overflow is not None and manager.host_spill is not None:
            stash_overflow = manager.overflow.view(-1)[0]
            host_spill = manager.host_spill.view(-1)[0]
        else:
            stash_overflow = check_paged_stash_overflow().view(-1)[0]
            host_spill = check_paged_stash_host_spill().view(-1)[0]
        flags = self._ensure_flag_scratch(stash_overflow.device)
        flags[0].copy_(stash_overflow)
        flags[1].zero_()
        flags[2].copy_(host_spill)
        for probe in self.overbudget_probes:
            flag = probe.read()
            if flag is not None:
                flags[1].logical_or_(flag.view(-1)[0])
        return flags

    def _device_flags(self) -> torch.Tensor:
        """``(stash_overflow, overbudget, host_spill)`` rank sums, one all_reduce.

        Returns a device tensor. Allocation-free, so nothing here synchronizes.
        """
        flags = self._fill_local_flags()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                flags,
                op=torch.distributed.ReduceOp.SUM,
                group=self.process_group,
            )
        return flags

    def _enqueue_probe(self) -> None:
        """Start an asynchronous read of this rank's flags. Never blocks.

        The flags are not reduced across ranks, so the raise they lead to is
        rank-local: only a rank that overflowed raises.

        The spill flag is cleared by the next ``begin_step``, which is enqueued
        after this copy on the same stream, so the mirror sees this step's
        spills. The overflow flag is sticky (see ``reset_buffers``), so
        overwriting a copy that was never read loses nothing: this copy still
        carries it.
        """
        self._flags_host.copy_(self._fill_local_flags(), non_blocking=True)
        self._flags_event.record()
        self._pending_step = self.manager.step

    def _take_verdict(
        self, *, blocking: bool
    ) -> tuple[int, tuple[int, int, int]] | None:
        """Consume the in-flight verdict if it has landed.

        Returns this rank's ``(step, (overflow, overbudget, host_spill))`` or
        ``None`` when nothing is pending, or when the copy is still in flight
        and ``blocking`` is False.
        """
        if self._pending_step is None:
            return None
        if blocking:
            self._flags_event.synchronize()
        elif not self._flags_event.query():
            return None
        step = self._pending_step
        self._pending_step = None
        counts = self._flags_host.tolist()
        return step, (int(counts[0]), int(counts[1]), int(counts[2]))

    def check_moe_overflow(self) -> tuple[int, int, int]:
        """``(stash_overflow, overbudget, host_spill)`` rank sums, read now.

        Megatron's synchronous form, kept for ``blocking`` mode.
        """
        flags = self._device_flags()
        return int(flags[0].item()), int(flags[1].item()), int(flags[2].item())

    # ------------------------------------------------------------------
    # Skipping overflowed optimizer steps (``deferred`` mode)
    # ------------------------------------------------------------------
    def _skip_overflowed_steps(
        self,
        optimizers: OptimizersContainer,
        expert_biases: Sequence[torch.Tensor],
    ) -> None:
        """Hook ``optimizers`` so that, while any rank's overflow flag is set,
        every rank skips its step on device and puts back ``expert_biases``."""
        unsupported = set()
        for optimizer in optimizers:
            if not skips_step_on_found_inf(optimizer):
                name = type(optimizer).__name__
                adam = isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW))
                unsupported.add(f"non-fused {name}" if adam else name)
        if unsupported:
            raise ValueError(
                "Paged stash's default overflow check "
                f"({self._config_path}.overflow_check = 'deferred') learns of an "
                "overflow a step or two after it happens and skips the "
                "optimizer steps in between on the GPU, which only fused Adam "
                "and AdamW can do (fused=True, their default). This job uses: "
                f"{', '.join(sorted(unsupported))}. For now, other optimizers "
                "are only supported with Megatron-style blocking overflow "
                "checks, which rerun an overflowing step instead: "
                f"{self._config_path}.overflow_check = 'blocking'"
            )
        self._expert_biases = list(expert_biases)
        self._expert_bias_snapshots = [bias.clone() for bias in self._expert_biases]
        # Exactly 0.0 or 1.0: the fused kernels skip only on ``found_inf == 1``
        # and subtract it from the step count to undo the step's increment.
        self._skip_step = torch.zeros(
            (), dtype=torch.float32, device=self.manager.device
        )
        optimizers.register_step_pre_hook(self._flag_overflowed_step)
        optimizers.register_step_post_hook(self._undo_overflowed_step)

    def _snapshot_expert_biases(self) -> None:
        """Record the expert biases the next optimizer step starts from. They
        only change in that step, so after any forward/backward will do."""
        if self._expert_biases:
            torch._foreach_copy_(self._expert_bias_snapshots, self._expert_biases)

    def _flag_overflowed_step(
        self, optimizers: OptimizersContainer, args: Any, kwargs: Any
    ) -> None:
        """Step pre-hook: hand every optimizer whether any rank has overflowed,
        as AMP's ``found_inf``. All-reduced on device, so nothing syncs."""
        overflow = self.manager.overflow
        if overflow is None:  # nothing has been paged yet
            self._skip_step.zero_()
        else:
            self._skip_step.copy_(overflow.view(()))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # MAX rather than the verdict's SUM, to keep the flag 0 or 1.
            torch.distributed.all_reduce(
                self._skip_step,
                op=torch.distributed.ReduceOp.MAX,
                group=self.process_group,
            )
        for optimizer in optimizers:
            # pyrefly: ignore [missing-attribute]
            optimizer.found_inf = self._skip_step

    def _undo_overflowed_step(
        self, optimizers: OptimizersContainer, args: Any, kwargs: Any
    ) -> None:
        """Step post-hook: drop ``found_inf``, as ``GradScaler`` does, and on a
        skipped step put back the expert biases the MoE load-balancing
        pre-hook moved."""
        for optimizer in optimizers:
            # pyrefly: ignore [missing-attribute]
            del optimizer.found_inf
        if self._expert_biases:
            skipped = self._skip_step.bool()
            for bias, snapshot in zip(
                self._expert_biases, self._expert_bias_snapshots, strict=True
            ):
                torch.where(skipped, snapshot, bias, out=bias)

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------
    def _prepare_for_rerun(self) -> None:
        """Megatron's ``prepare_for_rerun``: the next attempt runs with paging
        disabled, which cannot overflow, and measures its live pages; the step
        after it reallocates buffers that fit them and pages again."""
        logger.info(
            "Paged stash: rerunning forward-backward with paged stash disabled."
        )
        assert self.manager.overflow is not None
        assert self.manager.host_spill is not None
        self.manager.overflow.zero_()
        self.manager.host_spill.zero_()
        # Discard the overflowed attempt's gradients. The runner wraps all of
        # the step's forward-backward, and the engine sets every gradient to
        # None just before it, so None is exactly the state the attempt started
        # from. Holding the attempt's gradients back instead would not be
        # enough: under CUDA graphs a parameter's ``grad`` can be the graph's
        # own output buffer, which a replay overwrites in place.
        self._optimizers.zero_grad(set_to_none=True)
        # Drop the CUDA graphs before releasing the pages they captured
        # pointers to, as Megatron's ``reset_cuda_graph`` precedes
        # ``release_stash_buffers``. The rerun runs eagerly, and the next step
        # captures again over the reallocated buffers.
        if self._reset_cuda_graphs is not None:
            self._reset_cuda_graphs()
        self.manager.release_stash_buffers()
        self.manager.paging_disabled = True

    # ------------------------------------------------------------------
    # Verdict handling
    # ------------------------------------------------------------------
    def _log_diagnostics(
        self, overbudget: int, host_spill: int, *, local: bool
    ) -> None:
        """Report the two verdict flags that do not invalidate the step: this
        rank's own when ``local``, otherwise counts of ranks."""

        def where(count: int) -> str:
            return "this rank" if local else f"{count} rank(s)"

        if overbudget > 0:
            logger.info(
                "Paged stash: token drop during MoE token dispatch (over budget) on "
                "%s. Consider increasing the token dispatcher capacity factor.",
                where(overbudget),
            )
        if host_spill > 0:
            logger.info(
                "Paged stash: spilled activations to pinned host on %s (CUDA "
                "stash full). Consider increasing "
                "%s.buffer_size_factor_cuda for potentially better performance.",
                where(host_spill),
                self._config_path,
            )

    def _handle_deferred_verdict(
        self, verdict: tuple[int, tuple[int, int, int]] | None
    ) -> None:
        """Act on this rank's verdict for an already-completed step: report the
        diagnostics, and raise on an overflow.

        The raise is rank-local: only a rank whose stash overflowed raises. The
        others block in their next collective until the process group times out
        or torchrun tears the job down.
        """
        if verdict is None:
            return
        step, (overflow, overbudget, host_spill) = verdict
        self._log_diagnostics(overbudget, host_spill, local=True)
        if overflow:
            raise RuntimeError(
                f"Paged stash buffer overflow detected on this rank by step "
                f"{step}. The CUDA paged stash buffer is too small for the "
                "current routing pattern.\n\n"
                "Every rank skipped its optimizer steps from the overflow on, so "
                "the parameters, optimizer state and MoE expert biases, including "
                "those of any checkpoint saved since, are from before it. As "
                "with AMP's skipped steps, the step count, learning rate "
                "schedule and data position still advanced.\n\n"
                "To fix, try one of:\n"
                "  1. Enable host spillover: "
                f"{self._config_path}.buffer_size_factor_cpu = 1.0\n"
                "  2. Increase buffer size: "
                f"{self._config_path}.buffer_size_factor_cuda = 2.0\n"
                "  3. Rerun overflowing steps with paging disabled, as Megatron "
                f"does: {self._config_path}.overflow_check = 'blocking'\n"
                f"  4. Disable paged stash: {self._disable_hint}"
            )

    def final_overflow_check(self) -> None:
        """Drain the last in-flight verdict at shutdown, blocking.

        The deferred check needs a later step to observe an overflow, so an
        overflow on the final steps would otherwise escape and the job would
        exit cleanly with those steps silently skipped. One synchronizing read
        at shutdown closes that window at zero per-step cost.
        """
        if not self.manager.enabled or self.overflow_check == "blocking":
            return
        self._handle_deferred_verdict(self._take_verdict(blocking=True))

    # ------------------------------------------------------------------
    # Step entry points
    # ------------------------------------------------------------------
    def __call__(self, fwd_bwd_fn: Callable[[], _T]) -> _T:
        """Run ``fwd_bwd_fn`` under the configured overflow policy."""
        if not self.manager.enabled:
            return fwd_bwd_fn()
        if self.overflow_check == "blocking":
            return self._call_blocking(fwd_bwd_fn)
        return self._call_deferred(fwd_bwd_fn)

    def _run_attempt(self, fwd_bwd_fn: Callable[[], _T]) -> _T:
        """Reset the stash and the over-budget probes, then run ``fwd_bwd_fn``."""
        self.manager.begin_step()
        for probe in self.overbudget_probes:
            probe.reset()
        # Triton compiles on first launch, which a CUDA graph capture cannot
        # tolerate; prime the cache while we are still eager.
        warmup_paged_stash_kernels()
        return fwd_bwd_fn()

    def _call_deferred(self, fwd_bwd_fn: Callable[[], _T]) -> _T:
        # Read an earlier step's verdict if its copy has landed; never blocks.
        self._handle_deferred_verdict(self._take_verdict(blocking=False))
        result = self._run_attempt(fwd_bwd_fn)
        if self.manager.active:
            self._enqueue_probe()
        self._snapshot_expert_biases()
        return result

    def _call_blocking(self, fwd_bwd_fn: Callable[[], _T]) -> _T:
        """Megatron's ``PagedStashRunner.__call__``: read the verdict at the end
        of every attempt, and rerun an overflowing step once with paging
        disabled."""
        attempts = 0
        while True:
            attempts += 1
            assert attempts <= self.max_attempts, (
                f"PagedStashRunner: {attempts} attempts exceeded the maximum "
                f"of {self.max_attempts}."
            )
            result = self._run_attempt(fwd_bwd_fn)
            # Nothing was paged in the measure step or a rerun, so there is no
            # verdict to reach.
            overflow_ranks = 0
            if self.manager.active:
                (
                    overflow_ranks,
                    overbudget_ranks,
                    host_spill_ranks,
                ) = self.check_moe_overflow()
                # Megatron also reruns a step whose dispatch dropped tokens,
                # with the capacity factor cleared. Here the capacity is part
                # of what is captured, so a rerun would drop the same tokens
                # again; the drop is reported instead.
                self._log_diagnostics(overbudget_ranks, host_spill_ranks, local=False)
            if overflow_ranks == 0:
                # Page again from the next step, as Megatron restores
                # moe_paged_stash after a successful rerun.
                self.manager.paging_disabled = False
                return result

            logger.info(
                "Paged stash: stashing buffer overflow on %d rank(s). "
                "Consider increasing %s.buffer_size_factor_cuda "
                "or %s.buffer_size_factor_cpu.",
                overflow_ranks,
                self._config_path,
                self._config_path,
            )
            self._prepare_for_rerun()


def hybridep_overbudget_probe() -> OverbudgetProbe:
    """HybridEP's token-drop flag, accumulated over every dispatch of a step:
    the ``overbudget`` slot of Megatron's ``check_moe_overflow``."""
    from torchtitan.distributed.deepep.hybridep import (
        check_hybridep_over_budget,
        reset_hybridep_over_budget,
    )

    return OverbudgetProbe(check_hybridep_over_budget, reset_hybridep_over_budget)


__all__ = [
    "DEFAULT_MODULE_FQN",
    "MIN_STASH_BYTES",
    "OverbudgetProbe",
    "PageCounts",
    "PagedStashBuffer",
    "PagedStashManager",
    "PagedStashRunner",
    "SCALE_INV_BLOCK_SIZE",
    "ScheduleEntry",
    "StashRecord",
    "StashSlot",
    "buffer_dtype_for",
    "check_paged_stash_host_spill",
    "check_paged_stash_overflow",
    "hybridep_overbudget_probe",
    "infer_slot_dims",
    "max_live_stash_generations",
    "moe_expert_biases",
    "paged_layer_schedule",
    "replay_schedule_page_peak",
    "skips_step_on_found_inf",
    "wait_for_copy",
    "warmup_paged_stash_kernels",
    "worth_paging",
]
