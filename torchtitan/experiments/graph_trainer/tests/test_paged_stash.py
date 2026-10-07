# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for MoE paged stashing.

* the ops and kernels: exact round trips, eagerly and replayed from a CUDA
  graph; host spill and the sticky overflow flag; how long a paged activation
  lives;
* sizing: one pool of live pages per ``(dtype, hidden_size)``, and the GraphPP
  schedule replay;
* the schedule: the paged-layer schedule read off GraphPP's actions, which
  stashes stay resident, and reloads prefetched one schedule entry ahead,
  across pipeline actions;
* the graph pass: slot dims, the paged-stash region and what is paged in it,
  where Megatron's stash points land, the warning for a padded activation left
  unpaged, and the GraphPP partition of the stash ops;
* the overflow check: blocking with an unpaged rerun of a step captured in a
  CUDA graph, or deferred and fatal, skipping the optimizer steps before the
  raise on device; and the optimizers each mode accepts.

End-to-end training with CUDA graphs and with GraphPP is covered by
``integration_tests.py``.

Run with::

    pytest torchtitan/experiments/graph_trainer/tests/test_paged_stash.py
"""

import copy
import gc
import operator
import weakref
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.fx as fx
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.pipelining.schedules import (
    _Action,
    BACKWARD_INPUT,
    BACKWARD_WEIGHT,
    FORWARD,
    FULL_BACKWARD,
    RECV_F,
    SEND_F,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.components.optim.optimizer import AdamW, OptimizersContainer
from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node,
    accumulate_param_grads_,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.cuda_graph import cuda_graph_pass
from torchtitan.experiments.graph_trainer.graph_pp import partition_joint_graph
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    BACKWARD_WITH_REDUCE_GRAD,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import (
    annotate_paged_stash_region,
    apply_graph_pp_schedule,
    apply_paged_stash_pass,
    check_paged_stash_host_spill,
    check_paged_stash_overflow,
    infer_slot_dims,
    max_live_stash_generations,
    OverbudgetProbe,
    paged_layer_schedule,
    PAGED_STASH_REGION,
    PAGED_STASH_TAG,
    PagedStashBuffer,
    PagedStashManager,
    PagedStashRunner,
    replay_schedule_page_peak,
    ScheduleEntry,
    StashSlot,
    tag_paged_stash_activations,
    warmup_paged_stash_kernels,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="paged stash requires CUDA"
)

BF16 = torch.bfloat16
E4M3 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu
MAX_TOKENS = 4096  # capacity-padded rows per activation: the worst case
HIDDEN = 256
PAGE_SIZE = 64
KEY = (BF16, HIDDEN)


@pytest.fixture
def manager():
    """The process-wide manager, reset and enabled on cuda:0."""
    PagedStashManager.reset_instance()
    manager = PagedStashManager.get_instance()
    manager.enabled = True
    manager.device = torch.device("cuda:0")
    manager.page_size = PAGE_SIZE
    yield manager
    PagedStashManager.reset_instance()


def _make_slots(
    manager: PagedStashManager,
    count: int,
    *,
    dtype: torch.dtype = BF16,
    token_divisor: int = 1,
    max_tokens: int = MAX_TOKENS,
    stage_index: int = 0,
) -> list[StashSlot]:
    """One slot per layer, as the graph pass registers them."""
    slots = []
    for layer in range(count):
        slot = StashSlot(
            slot_id=manager.next_slot_id(),
            dtype=dtype,
            hidden_size=HIDDEN,
            max_num_tokens=max_tokens,
            token_divisor=token_divisor,
            layer_id=layer,
            node_name=f"s{stage_index}l{layer}",
            stage_index=stage_index,
        )
        manager.register_slot(slot)
        slots.append(slot)
    return slots


def _pages(rows: int) -> int:
    return -(-rows // PAGE_SIZE)


def _random_activation(slot: StashSlot) -> torch.Tensor:
    """A padded activation of random bit patterns; flat for block scales."""
    itemsize = torch.empty((), dtype=slot.dtype).element_size()
    shape = (slot.max_stash_tokens, HIDDEN * itemsize)
    rows = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    rows = rows.view(slot.dtype)
    return rows.reshape(-1) if slot.token_divisor > 1 else rows


def _live_bytes(tensor: torch.Tensor, slot: StashSlot, num_tokens: int):
    """The bytes of the rows ``num_tokens`` live tokens occupy."""
    rows = tensor.reshape(-1, slot.hidden_size).view(torch.uint8)
    return rows[: num_tokens // slot.token_divisor]


def _stash(slot, tensor, num_tokens, *, prefetch=0) -> torch.Tensor:
    """Page ``tensor`` out through the forward op, as the rewritten graph does,
    into a handle sized as the graph pass sizes it: page ids and a spill flag.
    The handle starts as -1s, so whatever the op does not write shows."""
    handle = torch.full(
        (_pages(slot.max_stash_tokens) + 1,), -1, dtype=torch.int64, device="cuda"
    )
    torch.ops.paged_stash.stash_(handle, tensor, num_tokens, slot.slot_id, prefetch)
    return handle


def _reload(slot, handle, num_tokens, like: torch.Tensor) -> torch.Tensor:
    """Page an activation shaped like ``like`` back in through the backward ops."""
    out = torch.ops.paged_stash.reload(
        handle, num_tokens, slot.slot_id, list(like.shape), like.dtype
    )
    return torch.ops.paged_stash.wait_reload(out)


def _round_trip(slots, tensors, counts) -> list[torch.Tensor]:
    """Stash every activation, then reload them all: one forward/backward."""
    handles = [_stash(*args) for args in zip(slots, tensors, counts, strict=True)]
    return [
        _reload(*args) for args in zip(slots, handles, counts, tensors, strict=True)
    ]


def _assert_live_rows_equal(slots, expected, actual, live) -> None:
    for slot, want, got, n in zip(slots, expected, actual, live, strict=True):
        assert torch.equal(_live_bytes(got, slot, n), _live_bytes(want, slot, n))


def _setup_stash(manager, slots, live: list[int]):
    """Take ``slots`` through the measure step with ``live`` tokens each.

    Returns the activations, their counts, and what the measure step's reloads
    handed back; the buffers are allocated and their kernels compiled.
    """
    tensors = [_random_activation(slot) for slot in slots]
    counts = [torch.tensor(n, dtype=torch.int64, device="cuda") for n in live]
    manager.begin_step()
    assert not manager.active, "the first step must run in measure mode"
    measured = _round_trip(slots, tensors, counts)
    manager.begin_step()
    assert manager.active, "buffers must exist after the measure step"
    warmup_paged_stash_kernels()
    return tensors, counts, measured


def _capture_round_trip(manager, slots, tensors, counts):
    """Capture one round trip in a CUDA graph.

    Returns what the graph reloads into, and a function that replays it on
    fresh activations with the given live token counts.
    """
    manager.begin_step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = _round_trip(slots, tensors, counts)

    def replay(live: list[int]) -> None:
        for slot, tensor, count, n in zip(slots, tensors, counts, live, strict=True):
            tensor.copy_(_random_activation(slot))
            count.fill_(n)
        manager.begin_step()
        graph.replay()
        torch.cuda.synchronize()

    return outputs, replay


def _actions(spec: str) -> list[_Action]:
    """``"F0.1 B1.0"``: forward of stage 0 microbatch 1, then full backward of
    stage 1 microbatch 0. ``I``/``W`` split the backward into its input and
    weight halves; ``S``/``R`` are forward sends and receives."""
    kinds = {
        "F": FORWARD,
        "B": FULL_BACKWARD,
        "I": BACKWARD_INPUT,
        "W": BACKWARD_WEIGHT,
        "S": SEND_F,
        "R": RECV_F,
    }
    actions = []
    for token in spec.split():
        stage, microbatch = token[1:].split(".")
        actions.append(_Action(int(stage), kinds[token[0]], int(microbatch)))
    return actions


# ---------------------------------------------------------------------------
# Ops and kernels
# ---------------------------------------------------------------------------


@requires_cuda
def test_round_trip_is_exact(manager):
    """Paging out and back is bit-exact for the live rows of every slot kind,
    eagerly and replayed from a CUDA graph.

    Data slots address one row per token; a block-scale slot is a flat e8m0
    buffer with one row per 32 tokens, paged in its own (uint8) pool. Live
    counts that are not page multiples leave partial pages. The measure step
    hands back the original activation, so it is numerically identical to a
    run without paged stashing. The kernels read the live counts and the
    freelists from device memory, so one capture serves steps with different
    counts, provided ``begin_step`` resets the freelists (outside the graph)
    before each replay.
    """
    slots = _make_slots(manager, 2) + _make_slots(
        manager, 1, dtype=E8M0, token_divisor=32
    )
    live = [1997, 700, 1024]  # block scales cover whole 32-token blocks
    tensors, counts, measured = _setup_stash(manager, slots, live)
    assert all(out is tensor for out, tensor in zip(measured, tensors))
    assert len(manager.stash_buffers) == 2, "one pool per (dtype, hidden_size)"

    manager.begin_step()
    _assert_live_rows_equal(slots, tensors, _round_trip(slots, tensors, counts), live)
    assert not check_paged_stash_overflow().item()

    outputs, replay = _capture_round_trip(manager, slots, tensors, counts)
    for live in ([900, 1500, 2048], [64, 1997, 32]):
        replay(live)
        _assert_live_rows_equal(slots, tensors, outputs, live)
        assert not check_paged_stash_overflow().item()


@requires_cuda
def test_host_spill_and_overflow(manager):
    """Out of CUDA pages a stash spills to pinned host pages; out of both, the
    kernels flag an overflow. Both are decided on device from the live counts
    and the freelists, so one captured step serves steps that fit, spill or
    overflow, and spilled rows round trip exactly. Every step starts from full
    freelists and a clear spill flag, but the overflow stays flagged."""
    manager.buffer_size_factor_cuda = 0.5  # 16 pages: one 1024-token stash
    manager.buffer_size_factor_cpu = 0.5  # and one more on the host
    slots = _make_slots(manager, 2)
    tensors, counts, _ = _setup_stash(manager, slots, [1024, 1024])
    outputs, replay = _capture_round_trip(manager, slots, tensors, counts)

    # 1024 + 1024 and 512 + 1000 tokens outgrow the 16 CUDA pages, so the second
    # stash spills; 64 + 900 tokens fit exactly.
    for live, spills in (
        ([1024, 1024], True),
        ([512, 1000], True),
        ([64, 900], False),
    ):
        replay(live)
        _assert_live_rows_equal(slots, tensors, outputs, live)
        assert check_paged_stash_host_spill().item() == spills
        assert not check_paged_stash_overflow().item(), "a spill is not an overflow"

    replay([2048, 2048])  # 32 pages each: neither fits in either pool
    assert check_paged_stash_overflow().item(), "expected the overflow flag"
    manager.begin_step()
    assert check_paged_stash_overflow().item(), "the overflow flag is sticky"
    assert not check_paged_stash_host_spill().item()
    buffer = manager.get_buffer(BF16, HIDDEN)
    assert buffer.free_list_head.tolist() == [0, 0]
    assert buffer.free_list_tail.tolist() == [
        buffer.num_cuda_pages,
        buffer.num_host_pages,
    ]


@requires_cuda
def test_paged_activation_lives_until_its_copy_is_joined(manager):
    """A paged stash holds its padded activation only until its copy is joined:
    at the next paged layer's group start, or -- for the last stash before
    backward -- at backward's first reload, as Megatron's backward start joins
    it. That release is what keeps padding from accumulating across layers;
    the rows come back from the pages."""
    slots = _make_slots(manager, 2)
    tensors, counts, _ = _setup_stash(manager, slots, [1000, 1000])

    def stash(slot, count):
        """Stash a fresh activation and drop it. Returns the handle, the live
        rows, and a weak reference to the activation."""
        activation = _random_activation(slot)
        rows = _live_bytes(activation, slot, 1000).clone()
        return _stash(slot, activation, count), rows, weakref.ref(activation)

    def released(ref: weakref.ref) -> bool:
        gc.collect()
        return ref() is None

    handle_0, rows_0, ref_0 = stash(slots[0], counts[0])
    assert not released(ref_0), "released before its copy was joined"
    torch.ops.paged_stash.group_start(tensors[1])  # the next layer's
    assert released(ref_0), "the next group start did not release it"

    handle_1, rows_1, ref_1 = stash(slots[1], counts[1])
    assert not released(ref_1), "released before its copy was joined"
    restored_1 = _reload(slots[1], handle_1, counts[1], tensors[1])
    assert released(ref_1), "backward's first reload did not release it"
    restored_0 = _reload(slots[0], handle_0, counts[0], tensors[0])
    torch.cuda.synchronize()
    assert torch.equal(_live_bytes(restored_1, slots[1], 1000), rows_1)
    assert torch.equal(_live_bytes(restored_0, slots[0], 1000), rows_0)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@requires_cuda
def test_sizing_keeps_one_pool_of_live_pages(manager):
    """Paging replaces ``layers x padded`` residency with one pool per
    ``(dtype, hidden_size)`` holding live pages, however many layers or stages.

    The live count is not a page multiple, so summing tokens -- Megatron's
    accounting -- would come out short of the pages the stashes occupy. Under
    GraphPP each stage contributes its live pages to the schedule replay from
    its forward until the backward that consumes them: a full or input
    backward releases them, a weight backward does not.
    """
    actual, padded = 1000, 4000  # capacity factor 4, partial last page
    manager.buffer_size_factor_cuda = 1.0
    slots = [
        slot
        for stage, layers in enumerate((4, 2))
        for slot in _make_slots(manager, layers, max_tokens=padded, stage_index=stage)
    ]
    manager.begin_step()
    for slot in slots:
        manager.record_stash(slot, actual)
    for slot in reversed(slots):
        manager.record_reload(slot, actual)
    footprint = manager.stage_page_footprint()
    manager.begin_step()  # allocates from the measured peak
    (pool,) = manager.stash_buffers.values()
    assert pool.num_cuda_pages == len(slots) * _pages(actual)

    stage_pages = {0: 4 * _pages(actual), 1: 2 * _pages(actual)}
    assert footprint == {stage: {KEY: pages} for stage, pages in stage_pages.items()}
    actions = _actions("F0.0 F1.0 I0.0 W0.0 F0.1 F0.2 F1.1")
    peak = 2 * stage_pages[0] + 2 * stage_pages[1]
    assert replay_schedule_page_peak(actions, footprint) == {KEY: peak}
    assert max_live_stash_generations(actions) == 4


# ---------------------------------------------------------------------------
# The schedule: skipping stashes and prefetching reloads
# ---------------------------------------------------------------------------


def _entries(schedule: list[ScheduleEntry]) -> list[str]:
    """``"F0.3.1"``: forward of stage 0's layer 3 for microbatch 1."""
    return [
        f"{'F' if e.forward else 'B'}{e.stage_index}.{e.layer_id}.{e.microbatch}"
        for e in schedule
    ]


def test_paged_layer_schedule():
    """Megatron's schedule, read off a rank's actions: every paged layer's
    forward and backward in execution order, layers ascending in a forward and
    descending in a backward. Stages without paged layers, communication, and
    weight-gradient backwards add nothing. A forward whose backward goes by
    another name -- GraphRuntime's executed schedule renames ``FULL_BACKWARD``
    -- is rejected rather than read as keeping its stashes live to the end of
    the step."""
    actions = _actions("F0.0 F1.0 S1.0 F0.1 R1.1 B1.0 I0.0 W0.0 B0.1 F2.0 B2.0")
    layers = {0: (3, 4), 2: (9,)}  # stage 1 is dense
    assert _entries(paged_layer_schedule(actions, layers)) == [
        "F0.3.0", "F0.4.0", "F0.3.1", "F0.4.1",
        "B0.4.0", "B0.3.0", "B0.4.1", "B0.3.1",
        "F2.9.0", "B2.9.0",
    ]  # fmt: skip

    renamed = _actions("F0.0 F0.1 B0.0") + [_Action(0, BACKWARD_WITH_REDUCE_GRAD, 1)]
    with pytest.raises(ValueError, match=r"unmatched .*: \[\(0, 1\)\]"):
        paged_layer_schedule(renamed, layers)


@requires_cuda
def test_schedule_decides_which_stashes_stay_resident(manager):
    """A forward keeps a layer resident exactly when the next schedule entry is
    that layer's own backward (Megatron's ``remove_paged_tensor_from_stash``),
    which only a stage's last paged layer can satisfy. A resident stash copies
    nothing and takes no pages, and its reload hands back the original
    storage. A per-microbatch decision cannot be frozen into a captured graph;
    without pipeline parallelism every microbatch decides the same, so it can.
    """
    slots = _make_slots(manager, 2)  # stage 0, layers 0 and 1
    tensors, counts, _ = _setup_stash(manager, slots, [1000, 1000])
    manager.skip_immediate_backward = True

    def skips(microbatch: int) -> list[bool]:
        return [
            manager.should_skip_stash(ScheduleEntry(True, 0, layer, microbatch))
            for layer in (0, 1)
        ]

    # Without pipeline parallelism every forward is followed by its backward.
    assert skips(0) == skips(5) == [False, True]
    buffer = manager.get_buffer(BF16, HIDDEN)
    free_pages = (buffer.free_list_tail - buffer.free_list_head).clone()
    handle = _stash(slots[1], tensors[1], counts[1])
    assert torch.equal(buffer.free_list_tail - buffer.free_list_head, free_pages)
    assert not handle.any(), "a resident stash must leave its handle zeroed"
    resident = _reload(slots[1], handle, counts[1], tensors[1])
    assert resident.data_ptr() == tensors[1].data_ptr()

    # The schedule is looked up by pipeline rank: with DP alongside PP it
    # differs from the global rank. A dense stage between a forward and its
    # backward adds no entry, as dense layers add none in Megatron.
    order = {
        0: _actions("F0.0 F0.1 B0.0 B0.1"),
        1: _actions("F0.0 F1.0 B1.0 B0.0 F0.1 F0.2 B0.1 B0.2"),
    }
    apply_graph_pp_schedule(SimpleNamespace(rank=1, pipeline_order_with_comms=order))
    assert [skips(m) for m in range(3)] == [
        [False, True],
        [False, False],
        [False, False],
    ]

    scratch = torch.zeros(8, device="cuda")
    torch.cuda.synchronize()

    def decide_under_capture() -> list[bool]:
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            scratch.add_(1)
            return skips(0)

    assert decide_under_capture() == [False, False]
    manager.set_pipeline_actions(None)
    assert decide_under_capture() == [False, True]

    manager.skip_immediate_backward = False
    assert skips(0) == [False, False]


class _PopLog:
    """Labels every pop the stash buffers launch with the current ``phase``."""

    def __init__(self, monkeypatch):
        self.phase = ""
        self.launched: list[str] = []
        launch_pop = PagedStashBuffer.launch_pop

        def logged_launch_pop(buffer, *args, **kwargs):
            self.launched.append(self.phase)
            return launch_pop(buffer, *args, **kwargs)

        monkeypatch.setattr(PagedStashBuffer, "launch_pop", logged_launch_pop)


@requires_cuda
def test_reloads_are_prefetched_one_schedule_entry_ahead(manager, monkeypatch):
    """Each backward's reloads are issued at the schedule entry before it, as
    Megatron issues them: a forward's group commit when a backward runs next,
    or the end of the previous layer's expert MLP backward -- in the same
    backward or in another pipeline action. The reload then hands backward the
    prefetched tensor without popping again, and the rows are exact. Without a
    prefetch, each reload pops its pages when backward reaches it.

    Layer 1 pages two tensors and ends its expert MLP backward before reading
    the second, as a weight gradient that runs after the input gradient does.
    """
    slots = _make_slots(manager, 2)  # stage 0, layers 0 and 1
    slots.append(replace(slots[1], slot_id=manager.next_slot_id(), node_name="w"))
    manager.register_slot(slots[2])
    live = [1000, 700, 500]
    manager.buffer_size_factor_cuda = 2.0  # room for two microbatches
    tensors, counts, _ = _setup_stash(manager, slots, live)
    order = {0: _actions("F0.0 F0.1 B0.0 B0.1")}
    apply_graph_pp_schedule(SimpleNamespace(rank=0, pipeline_order_with_comms=order))
    log = _PopLog(monkeypatch)
    reload_next = torch.ops.paged_stash.reload_next

    handles = []
    for microbatch in range(2):
        log.phase = f"F0.{microbatch}"
        # Layer 1's last stash is its group commit, which prefetches.
        handles.append(
            [
                _stash(slot, tensor, count, prefetch=int(slot is slots[2]))
                for slot, tensor, count in zip(slots, tensors, counts, strict=True)
            ]
        )

    restored = []

    def reload(microbatch: int, i: int) -> torch.Tensor:
        log.phase = f"B0.{microbatch} reloads {slots[i].node_name}"
        out = _reload(slots[i], handles[microbatch][i], counts[i], tensors[i])
        restored.append((i, out))
        return out

    for microbatch in range(2):
        out = reload(microbatch, 1)
        log.phase = f"B0.{microbatch} ends layer 1"
        reload_next(out, slots[2].slot_id, 1)  # before slot 2's reload
        reload(microbatch, 2)
        out = reload(microbatch, 0)
        log.phase = f"B0.{microbatch} ends layer 0"
        reload_next(out, slots[0].slot_id, 1)
    torch.cuda.synchronize()
    assert log.launched == [
        "F0.1",  # B0.0's layer 1: the next action
        "F0.1",
        "B0.0 ends layer 1",  # B0.0's layer 0
        "B0.0 ends layer 0",  # B0.1's layer 1: the next action
        "B0.0 ends layer 0",
        "B0.1 ends layer 1",  # B0.1's layer 0
    ], "every reload is issued one schedule entry ahead, none on demand"
    for i, out in restored:
        assert torch.equal(
            _live_bytes(out, slots[i], live[i]),
            _live_bytes(tensors[i], slots[i], live[i]),
        )
    assert not check_paged_stash_overflow().item()

    manager.begin_step()
    log.launched.clear()
    log.phase = "on demand"
    handles = [_stash(*args) for args in zip(slots, tensors, counts, strict=True)]
    for i in (1, 2, 0):
        _reload(slots[i], handles[i], counts[i], tensors[i])
    torch.cuda.synchronize()
    assert log.launched == ["on demand"] * 3


# ---------------------------------------------------------------------------
# The graph pass and the GraphPP partition
# ---------------------------------------------------------------------------


def test_slot_dims_are_inferred():
    """How the stash addresses a tensor as rows: ``(token_divisor, hidden)``."""
    cases = [
        # A data tensor's rows are what the copy kernel addresses...
        ((2048, 256), E4M3, 2048, (1, 256)),
        # ...so the right element count with the wrong leading dim is
        # rejected: its live rows would not be a prefix.
        ((256, 2048), E4M3, 2048, None),
        # e8m0 block scales carry no token axis, only 32 tokens per row.
        ((2048 // 32 * 256,), E8M0, 2048, (32, 256)),
        # A shaped scale buffer whose rows are not 32-token blocks may be
        # stored column by column, so its own shape rejects it; a flat
        # reshape of it can still vouch for it (test_annotated_region_is_paged).
        ((2048 // 32 * 256 // 512, 512), E8M0, 2048, None),
        # Indivisible element counts, and padding that is not whole blocks.
        ((1000,), E8M0, 2048, None),
        ((64,), E8M0, 100, None),
    ]
    for shape, dtype, padded, dims in cases:
        tensor = torch.empty(shape, dtype=dtype, device="meta")
        assert infer_slot_dims(tensor, padded) == dims, (shape, dtype, padded)


def test_region_annotation_covers_the_module_subtree():
    """``annotate_paged_stash_region`` tags everything traced inside the
    matching modules' forward, submodules included, and nothing else."""

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nn.Linear(8, 8)

        def forward(self, x):
            return self.inner(x).relu()

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.routed_experts = Experts()
            self.other = nn.Linear(8, 8)

        def forward(self, x):
            return self.other(self.routed_experts(x))

    model = nn.ModuleDict({"layer": Layer()})
    annotate_module_fqns(model)
    annotate_paged_stash_region(model, "layer.routed_experts")
    with preserve_node_meta():
        gm = make_fx(model["layer"])(torch.randn(4, 8))

    traced = [n for n in gm.graph.nodes if n.op == "call_function"]
    in_region = {
        n.name for n in traced if n.meta.get("custom", {}).get(PAGED_STASH_REGION)
    }
    under_experts = {
        n.name
        for n in traced
        if n.meta["custom"]["module_fqn"].startswith("layer.routed_experts")
    }
    assert in_region == under_experts
    assert in_region and len(in_region) < len(traced)


def _fake_experts_graph(*, fused_mxfp8: bool = False):
    """A traced-shaped routed-expert layer as the memory-policy pass leaves it
    for paged stashing: a static-capacity HybridEP dispatch, an expert MLP whose
    activations backward reads, and the combine, which scales the expert output
    by the per-row router scores first. Every node of it carries the paged-stash
    region annotation -- backward nodes too, as tracing copies a forward node's
    annotations onto its backward -- except one padded activation outside the
    region. SAC- and offload-style tags are already on it. Hand-built, so no
    process group, HybridEP buffer, or GPU is needed.

    The expert MLP is ``RoutedExperts``' BF16 SwiGLU, whose single gate/up GEMM
    backward reads only through the reshape and ``unbind`` that separate its
    halves, or, with ``fused_mxfp8``, a stand-in for the fused MXFP8 cuDNN
    composite, which saves its FC1 output and its columnwise-quantized FC1 and
    FC2 inputs with their e8m0 block scales. The FC1-input scales are a tile
    gather that backward reads flattened, as the composite saves them.

    Returns the graph module and its nodes by name.
    """
    import torchtitan.distributed.deepep.hybridep  # noqa: F401  (registers ops)

    padded = MAX_TOKENS
    fake_mode = FakeTensorMode()
    graph = torch.fx.Graph()
    call = graph.call_function
    aten = torch.ops.aten
    nodes: dict[str, fx.Node] = {}
    reads: list[fx.Node] = []  # forward activations backward reads

    def add(name, node, shape=None, dtype=BF16, *, read=False, region=True):
        custom: dict[str, Any] = {"module_fqn": "layers.0.moe.routed_experts"}
        if region:
            custom[PAGED_STASH_REGION] = True
        node.meta["custom"] = custom
        if shape is not None:
            with fake_mode:
                node.meta["val"] = torch.empty(shape, dtype=dtype, device="cuda")
        nodes[name] = node
        if read:
            reads.append(node)
        return node

    tokens = add("tokens", graph.placeholder("tokens"), (padded, HIDDEN))
    dispatch = add(
        "dispatch",
        call(
            torch.ops.hybridep.dispatch.default,
            (tokens, tokens, tokens, 8, 2, "ep", True, 1.0, None),
        ),
    )
    # Autograd numbers the dispatch; its backward carries the same number.
    dispatch.meta["seq_nr"] = 7
    x = add("x", call(operator.getitem, (dispatch, 0)), (padded, HIDDEN))
    scores = add(
        "scores", call(operator.getitem, (dispatch, 1)), (padded,), torch.float32
    )
    add("counts", call(operator.getitem, (dispatch, 2)), (4,), torch.int64)
    handle = add("handle", call(operator.getitem, (dispatch, 3)))

    if fused_mxfp8:
        # Stand-ins for the composite's cuDNN kernels.
        def produce(name, shape, dtype, *, read=True):
            return add(
                name,
                call(aten.empty.memory_format, (list(shape),), {"dtype": dtype}),
                shape,
                dtype,
                read=read,
            )

        num_scales = padded // 32 * HIDDEN
        produce("z", (padded, 2 * HIDDEN), BF16)
        produce("h_col_q", (padded, HIDDEN), E4M3)
        produce("h_col_sf", (num_scales,), E8M0)
        produce("x_col_q", (padded, HIDDEN), E4M3)
        # Gathered as 512-byte tiles, a shape that is not token rows, and
        # saved flattened: only the reshape backward reads shows its rows.
        x_col_sf = produce("x_col_sf", (num_scales // 512, 512), E8M0, read=False)
        add(
            "x_col_sf_flat",
            call(aten.view.default, (x_col_sf, [-1])),
            (num_scales,),
            E8M0,
            read=True,
        )
        out = produce("out", (padded, HIDDEN), BF16)
    else:
        w13 = add("w13", graph.placeholder("w13"), (4, HIDDEN, 2 * HIDDEN))
        w2 = add("w2", graph.placeholder("w2"), (4, HIDDEN, HIDDEN))
        reads.append(x)
        grouped_mm = aten._grouped_mm.default
        gate_up = add("gate_up", call(grouped_mm, (x, w13)), (padded, 2 * HIDDEN))
        gate_up_3d = add(
            "gate_up_3d",
            call(aten.reshape.default, (gate_up, [padded, 2, HIDDEN])),
            (padded, 2, HIDDEN),
        )
        halves = add("halves", call(aten.unbind.int, (gate_up_3d, -2)))
        gate = add(
            "gate", call(operator.getitem, (halves, 0)), (padded, HIDDEN), read=True
        )
        up = add("up", call(operator.getitem, (halves, 1)), (padded, HIDDEN), read=True)
        silu = add(
            "silu", call(aten.silu.default, (gate,)), (padded, HIDDEN), read=True
        )
        h = add("h", call(aten.mul.Tensor, (silu, up)), (padded, HIDDEN), read=True)
        out = add("out", call(grouped_mm, (h, w2)), (padded, HIDDEN), read=True)

    # HybridEP's combine_tokens: hidden_states * permuted_scores.reshape(-1, 1).
    # One score per row is too small to page.
    probs = add(
        "probs", call(aten._to_copy.default, (scores,), {"dtype": BF16}), (padded,)
    )
    probs_2d = add(
        "probs_2d", call(aten.reshape.default, (probs, [-1, 1])), (padded, 1), read=True
    )
    scaled = add("scaled", call(aten.mul.Tensor, (out, probs_2d)), (padded, HIDDEN))
    combine = add(
        "combine",
        call(torch.ops.hybridep.combine.default, (scaled, handle, padded // 8)),
        (padded // 8, HIDDEN),
    )
    # Same layer and shape as the region's activations, but not annotated.
    add(
        "outside",
        call(aten.neg.default, (tokens,)),
        (padded, HIDDEN),
        read=True,
        region=False,
    )
    bwd = [add(f"bwd_{n.name}", call(aten.mul.Tensor, (n, n))) for n in reads]
    # The gradient of the expert MLP's input, which the dispatch's backward
    # scatters back to the tokens.
    dx = add("dx", call(aten.add.Tensor, tuple(bwd[:2])), (padded, HIDDEN))
    dispatch_bwd = add(
        "dispatch_bwd",
        call(
            torch.ops.hybridep.combine_bwd.default,
            (dx, probs, handle, padded // 8, 8),
        ),
    )
    dispatch_bwd.meta["seq_nr"] = dispatch.meta["seq_nr"]
    for node in (*bwd, dx, dispatch_bwd):
        node.meta["autograd_backward"] = True
    graph.output((combine, *bwd, dispatch_bwd))

    for node in graph.nodes:
        if node.op == "call_function" and not node.meta.get("autograd_backward"):
            node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
    for node in dispatch.users:  # SAC saves the dispatch and combine collectives
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    for node in (dispatch, combine):
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
    if not fused_mxfp8:
        nodes["gate_up"].meta["recompute"] = CheckpointPolicy.MUST_CPU_OFFLOAD
    return torch.fx.GraphModule(torch.nn.Module(), graph), nodes


def _stashed(nodes: dict[str, fx.Node]) -> set[str]:
    return {name for name, node in nodes.items() if node.meta.get(PAGED_STASH_TAG)}


_VIEW_TARGETS = (
    operator.getitem,
    torch.ops.aten.unbind.int,
    torch.ops.aten.reshape.default,
    torch.ops.aten.view.default,
)


def _with_views(node: fx.Node) -> list[fx.Node]:
    """``node`` and the forward views backward may read it through."""
    views = [node]
    for user in node.users:
        if user.target in _VIEW_TARGETS and not _is_backward_node(user):
            views += _with_views(user)
    return views


# What the stash pages in each expert MLP: node -> (dtype, hidden_size,
# token_divisor) of its slot.
_PAGED = {
    # The dispatch output, the gate/up projection, the SiLU, the gate product
    # and the down projection's output.
    "bf16": {
        "x": (BF16, HIDDEN, 1),
        "gate_up": (BF16, 2 * HIDDEN, 1),
        "silu": (BF16, HIDDEN, 1),
        "h": (BF16, HIDDEN, 1),
        "out": (BF16, HIDDEN, 1),
    },
    # What the composite saves and the expert output the combine scales, but
    # not the dispatch output: backward reads its quantized copy instead.
    "mxfp8": {
        "z": (BF16, 2 * HIDDEN, 1),
        "h_col_q": (E4M3, HIDDEN, 1),
        "h_col_sf": (E8M0, HIDDEN, 32),
        "x_col_q": (E4M3, HIDDEN, 1),
        "x_col_sf": (E8M0, HIDDEN, 32),
        "out": (BF16, HIDDEN, 1),
    },
}


@requires_cuda
@pytest.mark.parametrize("expert_mlp", ["bf16", "mxfp8"])
def test_annotated_region_is_paged(manager, caplog, expert_mlp):
    """Paged stashing decides its annotated region, with nothing declared and
    whatever SAC or CPU offload tagged it: every activation there that backward
    reads is stashed and saved, none is recomputed, and after the pass backward
    reads them -- and the views it reads them through -- only from reloads. A
    buffer whose own shape is not token rows is addressed through the flat
    reshape backward reads it by. Activations the stash cannot hold keep SAC's
    decision, as do the collectives and everything outside the region."""
    gm, n = _fake_experts_graph(fused_mxfp8=expert_mlp == "mxfp8")
    paged = _PAGED[expert_mlp]
    tag_paged_stash_activations(gm)

    assert _stashed(n) == set(paged)
    for name in paged:
        assert n[name].meta["recompute"] is CheckpointPolicy.MUST_SAVE, name
    for name in ("probs", "outside"):
        assert n[name].meta["recompute"] is CheckpointPolicy.PREFER_RECOMPUTE, name
    for name in ("dispatch", "combine"):
        assert n[name].meta["recompute"] is CheckpointPolicy.MUST_SAVE, name
    recomputed = CheckpointPolicy.MUST_RECOMPUTE
    assert all(node.meta.get("recompute") is not recomputed for node in gm.graph.nodes)

    originals = [view for name in paged for view in _with_views(n[name])]
    with caplog.at_level("WARNING"):
        apply_paged_stash_pass(gm, None, page_size=PAGE_SIZE, prefetch_lookahead=0)
    assert "unpaged" not in caplog.text
    assert not [
        user for node in originals for user in node.users if _is_backward_node(user)
    ], "backward still reads a padded original"
    targets = [node.target for node in gm.graph.nodes]
    ops = torch.ops.paged_stash
    assert targets.count(ops.stash_.default) == len(paged)
    assert targets.count(ops.wait_reload.default) == len(paged)
    slots = {
        slot.node_name: (slot.dtype, slot.hidden_size, slot.token_divisor)
        for slot in manager.slots.values()
    }
    assert slots == {n[name].name: dims for name, dims in paged.items()}


@requires_cuda
def test_pass_places_megatron_stash_points(manager):
    """The rewritten graph stashes, joins and prefetches where Megatron does.

    * The expert MLP reads the dispatch output through a ``group_start`` just
      before its first read, where Megatron's ``paged_stash_group_start`` joins
      the previous layer's copies; the stash still reads the dispatch output
      itself.
    * A layer's stashes launch together and in order once the expert MLP has
      produced everything it saves, as ``paged_stash_group_commit`` launches
      them after the grouped MLP, so no copy runs alongside the expert GEMMs.
      The last one is where the next schedule entries' reloads start.
    * ``reload_next`` passes the gradient of the expert MLP's input, on its way
      to the dispatch's backward, as soon as it exists: Megatron issues the
      next reloads once the grouped MLP's backward has produced it.

    With prefetching off there is no ``reload_next``, and with nothing tagged
    the pass changes nothing.
    """
    ops = torch.ops.paged_stash
    gm, n = _fake_experts_graph()
    # A second, later forward read of the dispatch output, which the same
    # group_start must serve.
    with gm.graph.inserting_after(n["silu"]):
        late_read = gm.graph.call_function(torch.ops.aten.neg.default, (n["x"],))
    late_read.meta.update(n["x"].meta)
    tag_paged_stash_activations(gm)
    readers = [user for user in n["x"].users if not _is_backward_node(user)]
    assert len(readers) == 2
    apply_paged_stash_pass(gm, None, page_size=PAGE_SIZE, prefetch_lookahead=1)
    nodes = list(gm.graph.nodes)

    (start,) = [node for node in nodes if node.target is ops.group_start.default]
    assert start.args == (n["x"],)
    assert nodes[nodes.index(start) + 1] is min(readers, key=nodes.index)
    assert all(start in reader.all_input_nodes for reader in readers)
    assert {user.target for user in n["x"].users} == {
        ops.group_start.default,
        ops.stash_.default,
    }

    stashes = [node for node in nodes if node.target is ops.stash_.default]
    order = ("x", "gate_up", "silu", "h", "out")
    assert [stash.args[1] for stash in stashes] == [n[name] for name in order]
    assert [stash.args[4] for stash in stashes] == [0, 0, 0, 0, 1]
    first, last = nodes.index(stashes[0]), nodes.index(stashes[-1])
    assert first > nodes.index(n["out"]), "launched before the MLP finished"
    launch_ops = (ops.stash_.default, torch.ops.aten.empty.memory_format)
    assert all(node.target in launch_ops for node in nodes[first : last + 1])

    reload_next = nodes[nodes.index(n["dx"]) + 1]
    assert reload_next.target is ops.reload_next.default
    assert _is_backward_node(reload_next)
    assert reload_next.args[0] is n["dx"] and reload_next.args[2] == 1
    assert manager.slots[reload_next.args[1]].layer_id == 0
    assert n["dispatch_bwd"].args[0] is reload_next

    gm, n = _fake_experts_graph()
    tag_paged_stash_activations(gm)
    apply_paged_stash_pass(gm, None, page_size=PAGE_SIZE, prefetch_lookahead=0)
    assert ops.reload_next.default not in {node.target for node in gm.graph.nodes}
    assert n["dispatch_bwd"].args[0] is n["dx"]

    gm, _ = _fake_experts_graph()  # nothing tagged
    apply_paged_stash_pass(gm, None)
    assert ops.stash_.default not in {node.target for node in gm.graph.nodes}


@requires_cuda
def test_warns_when_a_padded_activation_crosses_unpaged(manager, caplog):
    """Tagging sees the graph before remat. If a backward reader appears after
    it -- remat rebuilding a tensor the stash could not hold from its padded
    input -- that input crosses into backward at full size, and the pass names
    it."""
    gm, n = _fake_experts_graph()
    tag_paged_stash_activations(gm)
    scaled = n["scaled"]  # padded and in the region, but not read by backward
    with gm.graph.inserting_before(next(iter(reversed(gm.graph.nodes)))):
        reader = gm.graph.call_function(torch.ops.aten.neg.default, (scaled,))
    reader.meta.update(scaled.meta, autograd_backward=True)
    with caplog.at_level("WARNING"):
        apply_paged_stash_pass(gm, None, page_size=PAGE_SIZE, prefetch_lookahead=0)
    assert "cross into backward unpaged" in caplog.text
    assert scaled.name in caplog.text


def test_graphpp_partition_keeps_stashes_in_forward_and_reloads_in_backward(manager):
    """GraphPP splits the rewritten graph with the stash in forward and the
    reload in backward.

    The forward stash mutates an input-free ``empty`` handle, and the reload
    reads only values forward already has. Extraction copies every node whose
    inputs are available; the partitioner's ``must_be_in_forward`` /
    ``must_be_in_backward`` tags on impure ops keep the reload ops out of
    forward, and backward reads the handle forward filled rather than an
    ``empty`` of its own. ``group_start`` and ``reload_next`` pass a forward and
    a backward value through, so each stays where its consumers are -- and
    backward does not replay ``group_start``, or the forward ops it reads, even
    though it has what they need. Measure mode parks the activation in forward
    and hands it back in backward, so the split graphs must compute the joint
    graph's results.
    """
    ops = torch.ops.paged_stash
    (slot,) = _make_slots(manager, 1, dtype=torch.float32, max_tokens=8)
    num_tokens = torch.tensor(8)

    class _StashedSquare(torch.autograd.Function):
        # The pass's rewrite, written by hand.
        @staticmethod
        def forward(ctx, h):
            squared = ops.group_start(h) * h
            handle = torch.empty(_pages(slot.max_stash_tokens) + 1, dtype=torch.int64)
            ops.stash_(handle, h, num_tokens, slot.slot_id, 1)
            ctx.save_for_backward(handle)
            return squared

        @staticmethod
        def backward(ctx, grad):
            (handle,) = ctx.saved_tensors
            h = ops.reload(handle, num_tokens, slot.slot_id, [8, HIDDEN], grad.dtype)
            grad_h = grad * 2.0 * ops.wait_reload(h)
            return ops.reload_next(grad_h, slot.slot_id, 1)

    def stage_step(x):
        # Backward reads ``x``, so it could recompute what group_start reads.
        loss = (_StashedSquare.apply(x * 3.0) * x).sum()
        (grad_x,) = torch.autograd.grad(loss, x)
        return [loss, grad_x]

    x = torch.randn(8, HIDDEN, requires_grad=True)
    traced = minimal_fx_tracer(stage_step)(x)
    fw, bw, meta = partition_joint_graph(traced, num_fwd_outputs=1)
    fw_targets = {node.target for node in fw.graph.nodes}
    bw_targets = {node.target for node in bw.graph.nodes}
    forward_ops = {ops.group_start.default, ops.stash_.default}
    backward_ops = {
        ops.reload.default,
        ops.wait_reload.default,
        ops.reload_next.default,
    }
    assert forward_ops <= fw_targets and not backward_ops & fw_targets
    assert backward_ops <= bw_targets and not forward_ops & bw_targets
    assert torch.ops.aten.empty.memory_format not in bw_targets
    recomputed = [
        node
        for node in bw.graph.nodes
        if node.target is torch.ops.aten.mul.Tensor
        and node.args[1] == 3.0
        and node.args[0].op == "placeholder"
    ]
    assert not recomputed, "backward recomputes x * 3 from the saved x"

    fw_out = fx.Interpreter(fw).boxed_run([x])
    saved = fw_out[meta.num_fwd_user_outputs :][: meta.num_saved_for_backward]
    by_name = dict(zip(meta.saved_for_backward_names, saved, strict=True))
    bw_out = fx.Interpreter(bw).boxed_run([by_name[n] for n in meta.bwd_input_names])
    joint = traced.gm(x)
    assert torch.equal(fw_out[0], joint[0])
    assert torch.equal(bw_out[0], joint[1])
    assert torch.allclose(bw_out[0], 27.0 * x * x)


# ---------------------------------------------------------------------------
# Overflow verdict
# ---------------------------------------------------------------------------


# AdamW implementations, as their AdamW config fields.
_ADAMW_IMPLEMENTATIONS: dict[str, dict[str, Any]] = {
    "fused": {"fused": True},
    "fused_bf16_moments": {"fused": True, "moment_dtype": "bfloat16"},
    "foreach": {"fused": False, "foreach": True},
    "for_loop": {"fused": False, "foreach": False},
}


def _optimizers(model: nn.Module, implementation: str = "fused") -> OptimizersContainer:
    """AdamW over every parameter of ``model``."""
    config = OptimizersContainer.Config(
        optimizers=[
            AdamW.Config(
                pattern=r".*", lr=0.1, **_ADAMW_IMPLEMENTATIONS[implementation]
            )
        ]
    )
    return config.build(model_parts=[model])


def _runner(
    manager: PagedStashManager,
    mode: str,
    *,
    optimizers: OptimizersContainer | None = None,
    **kwargs: Any,
) -> PagedStashRunner:
    """A runner whose optimizers, unless given, own a parameter no step uses."""
    if manager.overflow is None:
        manager.overflow = torch.zeros(1, dtype=torch.int64, device="cuda")
        manager.host_spill = torch.zeros(1, dtype=torch.int64, device="cuda")
    if optimizers is None:
        optimizers = _optimizers(nn.Linear(1, 1, device="cuda"))
    return PagedStashRunner(optimizers, overflow_check=mode, **kwargs)


def _activate(manager: PagedStashManager) -> None:
    """Give the manager one small buffer, as the measure step would."""
    manager.status = "captured"
    manager.stash_buffers = {
        KEY: PagedStashBuffer(
            PAGE_SIZE,
            HIDDEN,
            PAGE_SIZE,
            manager.device,
            manager.overflow,
            manager.host_spill,
            BF16,
        )
    }


def _drains_the_stream(fn: Callable[[], Any]) -> tuple[Any, bool]:
    """Call ``fn`` behind ~1.7s of queued GPU work.

    Returns its result and whether that work had drained by the time ``fn``
    returned, which is what a host synchronization inside ``fn`` would cause.
    """
    torch.cuda.synchronize()
    end = torch.cuda.Event()
    torch.cuda._sleep(2_000_000_000)
    result = fn()
    end.record()
    drained = end.query()
    torch.cuda.synchronize()
    return result, drained


@requires_cuda
def test_blocking_mode(manager, caplog):
    """The verdict is read before the step returns, as in Megatron.

    A HybridEP token drop comes from the capacity factor, which is compiled
    into the graph, so it is reported and the step kept; the flag accumulates
    over every dispatch and is reset in place for the next step. A stash
    overflow inside a captured step drops the graph along with the pages it
    points into and reruns the step eagerly with paging disabled. The rerun's
    gradient replaces the attempt's, even though the parameter's ``grad`` is
    the graph's own output buffer, which the attempt's replay wrote in place.
    The next step pages again and captures a new graph over reallocated buffers
    of the same size.
    """
    from torchtitan.distributed.deepep import hybridep

    slots = _make_slots(manager, 1)
    tensors, counts, _ = _setup_stash(manager, slots, [1000])
    tensors[0].copy_(torch.randn_like(tensors[0]))  # finite, so sums compare
    inputs = (tensors[0], counts[0])

    def graph_grad(tensor, count):  # a gradient computed from reloaded rows
        return _round_trip(slots, [tensor], [count])[0][:PAGE_SIZE].float().sum(0)

    gm = cuda_graph_pass(make_fx(graph_grad)(*inputs), inputs, static_input_indices=[])
    holder = nn.Module()
    holder.weight = nn.Parameter(torch.empty(HIDDEN, device="cuda"))
    weight = holder.weight
    probe = OverbudgetProbe(
        hybridep.check_hybridep_over_budget, hybridep.reset_hybridep_over_budget
    )
    runner = _runner(
        manager, "blocking", optimizers=_optimizers(holder), overbudget_probes=[probe]
    )
    expected = tensors[0][:PAGE_SIZE].float().sum(0)
    paged: list[bool] = []  # whether each attempt of the last step paged

    def step(dropped: tuple[int, ...] = ()) -> torch.Tensor:
        """One step from the engine's starting state: every gradient None.
        ``dropped`` is the over-budget flag of each of its dispatches."""
        paged.clear()
        weight.grad = None

        def forward_backward():
            paged.append(manager.active)
            for flag in dropped:
                hybridep._record_over_budget(
                    torch.tensor([flag], dtype=torch.int32, device="cuda")
                )
            accumulate_param_grads_([weight], [gm(*inputs)])

        runner(forward_backward)
        return weight.grad

    for _ in range(2):  # warmup, then capture
        assert torch.equal(step(), expected)
    captured = gm.forward._graph
    assert (
        weight.grad.data_ptr() == gm.forward._output.data_ptr()
    ), "the captured step's gradient must be the graph's output buffer"

    with caplog.at_level("INFO"):
        assert torch.equal(step(dropped=(0, 1, 0)), expected)
    assert paged == [True], "a token drop must not rerun the step"
    assert "over budget" in caplog.text
    over_budget = hybridep.check_hybridep_over_budget()

    pages = manager.get_buffer(BF16, HIDDEN).num_cuda_pages
    counts[0].fill_(1200)  # 19 pages; the buffer holds int(16 * 1.10) = 17
    assert torch.equal(step(), expected), "the attempt's gradient was kept"
    assert paged == [True, False], "the rerun must run with paging disabled"
    assert gm.forward._graph is None, "the rerun must run eagerly"
    assert not over_budget.item(), "the drop flag must be reset in place"

    counts[0].fill_(1000)
    assert torch.equal(step(), expected)
    assert paged == [True], "the next step must page again"
    assert gm.forward._graph not in (None, captured), "the next step recaptures"
    assert manager.get_buffer(BF16, HIDDEN).num_cuda_pages == pages
    assert torch.equal(step(), expected), "a replay added into its own output"


@requires_cuda
def test_deferred_mode(manager):
    """The verdict for a step is read on a later step once its copy has
    landed, never blocking, and an overflow is fatal when it is read. The flag
    is sticky, so when the CPU runs ahead and a step's copy is overwritten
    before it is read, the next copy still carries the overflow; and the check
    at shutdown reads the final step's verdict, which no later step would."""
    runner = _runner(manager, "deferred")
    _activate(manager)

    def step(overflow: bool = False) -> torch.Tensor:
        if overflow:
            manager.overflow.fill_(1)  # as the copy kernel does when out of pages
        return torch.zeros((), device="cuda")

    step(overflow=True)  # load every kernel a step launches: a kernel's first
    manager.overflow.zero_()  # launch would wait out the queued work below
    for _ in range(3):
        runner(step)
    runner.final_overflow_check()  # clean steps report nothing

    torch.cuda._sleep(2_000_000_000)  # keep the GPU busy: the CPU runs ahead
    runner(lambda: step(overflow=True))
    assert not runner._flags_event.query(), "the deferred check waited for the GPU"
    runner(step)  # overwrites the overflowing step's copy before it is read
    torch.cuda.synchronize()
    with pytest.raises(RuntimeError, match="Paged stash buffer overflow"):
        runner(step)  # reads a copy that has landed

    manager.overflow.zero_()  # a fresh run
    runner(step)
    runner(lambda: step(overflow=True))
    with pytest.raises(RuntimeError, match="Paged stash buffer overflow"):
        runner.final_overflow_check()


@requires_cuda
@pytest.mark.parametrize("implementation", ["fused", "fused_bf16_moments"])
def test_deferred_mode_skips_overflowed_optimizer_steps(manager, implementation):
    """A clean step updates exactly as without the runner. While an overflow is
    flagged, the optimizer step leaves the parameters and their state as they
    were and puts back the expert bias that the load-balancing pre-hook moved,
    all on device; the next step raises the overflow."""
    model = nn.Linear(HIDDEN, HIDDEN, device="cuda")
    reference = copy.deepcopy(model)
    optimizers = _optimizers(model, implementation)
    reference_optimizers = _optimizers(reference, implementation)
    expert_bias = torch.zeros(8, device="cuda")

    def update_expert_bias(*_: Any) -> None:  # the MoE load-balancing hook
        expert_bias.add_(1.0)

    # Registered before the runner's hooks, as the trainer registers it.
    optimizers.register_step_pre_hook(update_expert_bias)
    runner = _runner(
        manager, "deferred", optimizers=optimizers, expert_biases=[expert_bias]
    )
    _activate(manager)
    inputs = torch.randn(4, HIDDEN, device="cuda")

    def train_step(overflow: bool = False) -> None:
        def forward_backward() -> torch.Tensor:
            if overflow:
                manager.overflow.fill_(1)  # as the copy kernel does
            loss = model(inputs).sum()
            loss.backward()
            return loss

        runner(forward_backward)
        optimizers.step()
        optimizers.zero_grad()

    manager.overflow.fill_(1)  # load every kernel the timed step launches: a
    manager.overflow.zero_()  # kernel's first launch would wait out the sleep
    train_step()
    reference(inputs).sum().backward()
    reference_optimizers.step()
    params = zip(model.parameters(), reference.parameters(), strict=True)
    for param, expected in params:
        assert torch.equal(param, expected), "a clean step must be unchanged"
    assert expert_bias.eq(1.0).all()

    state = optimizers.optimizers[0].state[model.weight]
    before = [t.clone() for t in (model.weight, *state.values(), expert_bias)]
    _, drained = _drains_the_stream(lambda: train_step(overflow=True))
    assert not drained, "skipping the step drained the stream"
    after = [model.weight, *state.values(), expert_bias]
    for got, want in zip(after, before, strict=True):
        assert torch.equal(got, want), "the overflowed step was applied"
    with pytest.raises(RuntimeError, match="skipped its optimizer steps"):
        train_step()


@requires_cuda
def test_only_blocking_mode_accepts_optimizers_that_cannot_skip_a_step(manager):
    """Deferred mode skips overflowed steps on device, which only fused Adam and
    AdamW can do, so it rejects any other optimizer up front and points at
    blocking mode, which reruns an overflowing step instead and leaves the
    optimizer alone."""
    model = nn.Linear(HIDDEN, HIDDEN, device="cuda")
    _runner(manager, "deferred", optimizers=_optimizers(model, "fused_bf16_moments"))
    for implementation in ("foreach", "for_loop"):
        optimizers = _optimizers(model, implementation)
        with pytest.raises(
            ValueError, match="non-fused AdamW.*overflow_check = 'blocking'"
        ):
            _runner(manager, "deferred", optimizers=optimizers)
        _runner(manager, "blocking", optimizers=optimizers)
        model(torch.randn(4, HIDDEN, device="cuda")).sum().backward()
        optimizers.step()  # a found_inf it cannot honor would assert here
