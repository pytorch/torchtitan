# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for MoE paged stashing.

* ops and kernels: exact round trips for data and block-scale slots, host
  spill, overflow and reset, activation lifetime, and replay inside a captured
  CUDA graph;
* sizing: one pool of live pages per ``(dtype, hidden_size)``, and the GraphPP
  schedule replay;
* skipping the stash when a forward's own backward runs next;
* the graph pass, slot dims, and the GraphPP partition of the stash ops;
* the overflow check: deferred and fatal, skipping the optimizer steps before
  the raise on device, or blocking with an unpaged rerun, including a rerun of
  a step captured in a CUDA graph; and the optimizers each mode accepts.

End-to-end training with CUDA graphs and with GraphPP is covered by
``integration_tests.py``.

Run with::

    pytest torchtitan/experiments/graph_trainer/tests/test_paged_stash_overflow.py
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

from torchtitan.components.optimizer import default_adamw, OptimizersContainer
from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node,
    accumulate_param_grads_,
)
from torchtitan.experiments.graph_trainer.cudagraph import cudagraph_pass
from torchtitan.experiments.graph_trainer.graph_pp import partition_joint_graph
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import (
    apply_graph_pp_skip_plan,
    apply_paged_stash_pass,
    check_paged_stash_host_spill,
    check_paged_stash_overflow,
    declared_slot_dims,
    forward_skip_plan,
    infer_slot_dims,
    mark_pageable,
    max_live_stash_generations,
    OverbudgetProbe,
    PAGED_STASH_TAG,
    PagedStashBuffer,
    PagedStashManager,
    PagedStashRunner,
    replay_schedule_page_peak,
    set_marking_enabled,
    StashSlot,
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


def _fresh_manager(**attrs: Any) -> PagedStashManager:
    """Reset the process-wide manager and enable it on cuda:0."""
    PagedStashManager.reset_instance()
    manager = PagedStashManager.get_instance()
    manager.enabled = True
    manager.device = torch.device("cuda:0")
    manager.page_size = PAGE_SIZE
    for name, value in attrs.items():
        setattr(manager, name, value)
    return manager


@pytest.fixture
def manager():
    yield _fresh_manager()
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


def _count(num_tokens: int) -> torch.Tensor:
    return torch.tensor(num_tokens, dtype=torch.int64, device="cuda")


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


def _handle(slot: StashSlot) -> torch.Tensor:
    """A stash handle sized as the graph pass sizes it: page ids, spill flag."""
    return torch.empty(
        _pages(slot.max_stash_tokens) + 1, dtype=torch.int64, device="cuda"
    )


def _stash(slot, tensor, num_tokens, *, group_index=-1, skippable=False):
    """Page ``tensor`` out through the forward ops, as the rewritten graph does."""
    handle = _handle(slot)
    torch.ops.paged_stash.stash_(
        handle, tensor, num_tokens, slot.slot_id, group_index, skippable
    )
    torch.ops.paged_stash.wait_stash_(handle, tensor, None)
    return handle


def _reload(slot, handle, num_tokens, like: torch.Tensor) -> torch.Tensor:
    """Page an activation shaped like ``like`` back in through the backward ops."""
    out = torch.ops.paged_stash.reload(
        handle, num_tokens, slot.slot_id, list(like.shape), like.dtype
    )
    return torch.ops.paged_stash.wait(out)


def _round_trip(slots, tensors, counts) -> list[torch.Tensor]:
    """Stash every activation, then reload them all: one forward/backward."""
    handles = [_stash(*args) for args in zip(slots, tensors, counts, strict=True)]
    return [
        _reload(*args) for args in zip(slots, handles, counts, tensors, strict=True)
    ]


def _assert_live_rows_equal(slots, expected, actual, live) -> None:
    for args in zip(slots, expected, actual, live, strict=True):
        slot, want, got, n = args
        assert torch.equal(_live_bytes(got, slot, n), _live_bytes(want, slot, n))


def _setup_stash(manager, slots, live: list[int]):
    """Take ``slots`` through the measure step with ``live`` tokens each.

    Returns the activations, their counts, and what the measure step's reloads
    handed back; the buffers are allocated and their kernels compiled.
    """
    tensors = [_random_activation(slot) for slot in slots]
    counts = [_count(n) for n in live]
    manager.begin_step()
    assert not manager.active, "the first step must run in measure mode"
    measured = _round_trip(slots, tensors, counts)
    manager.begin_step()
    assert manager.active, "buffers must exist after the measure step"
    warmup_paged_stash_kernels()
    return tensors, counts, measured


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
    """Paging out and back is bit-exact for the live rows of every slot kind.

    Data slots address one row per token; a block-scale slot is a flat e8m0
    buffer with one row per 32 tokens, paged in its own (uint8) pool. Live
    counts that are not page multiples leave partial pages. The measure step
    hands back the original activation, so it is numerically identical to a
    run without paged stashing.
    """
    slots = _make_slots(manager, 2) + _make_slots(
        manager, 1, dtype=E8M0, token_divisor=32
    )
    live = [1997, 700, 1024]  # block scales cover whole 32-token blocks
    tensors, counts, measured = _setup_stash(manager, slots, live)
    assert all(out is tensor for out, tensor in zip(measured, tensors))
    assert len(manager.stash_buffers) == 2, "one pool per (dtype, hidden_size)"

    for _ in range(2):
        manager.begin_step()
        outputs = _round_trip(slots, tensors, counts)
        _assert_live_rows_equal(slots, tensors, outputs, live)
        assert not check_paged_stash_overflow().item()


@requires_cuda
def test_host_spill_overflow_and_reset(manager):
    """Out of CUDA pages a stash spills to pinned host pages and still round
    trips exactly; out of both, the kernels raise overflow. The next step starts
    from full freelists and a clear spill flag, but the overflow stays
    flagged."""
    manager.buffer_size_factor_cuda = 0.25  # one of the four stashes fits
    manager.buffer_size_factor_cpu = 0.25  # and one more fits on the host
    slots = _make_slots(manager, 4)
    tensors, counts, _ = _setup_stash(manager, slots, [1024] * 4)

    manager.begin_step()
    outputs = _round_trip(slots[:2], tensors[:2], counts[:2])
    assert check_paged_stash_host_spill().item(), "expected a host spill"
    assert not check_paged_stash_overflow().item(), "spill must not be overflow"
    _assert_live_rows_equal(slots[:2], tensors[:2], outputs, [1024, 1024])

    manager.begin_step()
    _round_trip(slots, tensors, counts)
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
def test_paged_activation_dies_at_its_wait_and_skipped_one_stays(manager):
    """What paging buys, and what skipping costs.

    A paged stash holds the padded activation only until its wait, which is
    what keeps padding from accumulating across layers; its rows come back
    from the pages. A skipped stash -- the last layer, whose backward runs
    next -- copies nothing, consumes no pages, and hands back the original
    storage, which is what Megatron gets by never calling offload_to_stash.
    """
    slots = _make_slots(manager, 2)
    tensors, counts, _ = _setup_stash(manager, slots, [1000, 1000])
    manager.skip_immediate_backward = True
    manager.set_static_skip(True)
    buffer = manager.get_buffer(BF16, HIDDEN)

    activation = _random_activation(slots[0])
    expected = _live_bytes(activation, slots[0], 1000).clone()
    ref = weakref.ref(activation)
    paged = _handle(slots[0])
    torch.ops.paged_stash.stash_(paged, activation, counts[0], slots[0].slot_id, 0)
    del activation
    gc.collect()
    assert ref() is not None, "released before the copy was joined"
    torch.ops.paged_stash.wait_stash_(paged, None, None)
    gc.collect()
    assert ref() is None, "paged stash still holds the padded activation"

    free_pages = (buffer.free_list_tail - buffer.free_list_head).clone()
    skipped = _stash(slots[1], tensors[1], counts[1], skippable=True)
    assert torch.equal(buffer.free_list_tail - buffer.free_list_head, free_pages)
    assert int(skipped.sum()) == 0, "a skipped stash must leave the handle zeroed"

    restored = _reload(slots[0], paged, counts[0], tensors[0])
    resident = _reload(slots[1], skipped, counts[1], tensors[1])
    torch.cuda.synchronize()
    assert torch.equal(_live_bytes(restored, slots[0], 1000), expected)
    assert resident.data_ptr() == tensors[1].data_ptr()


@requires_cuda
def test_paging_replays_inside_a_cuda_graph(manager):
    """The ops capture into a CUDA graph and page correctly on every replay.

    The kernels read the live counts and the freelists from device memory, so
    one capture serves steps with different token counts, provided
    ``begin_step`` resets the freelists (outside the graph) before each replay.
    """
    slots = _make_slots(manager, 2)
    tensors, counts, _ = _setup_stash(manager, slots, [1000, 700])
    manager.begin_step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = _round_trip(slots, tensors, counts)

    for live in ([900, 800], [64, 1500]):
        for slot, tensor, count, n in zip(slots, tensors, counts, live):
            tensor.copy_(_random_activation(slot))
            count.fill_(n)
        manager.begin_step()
        graph.replay()
        torch.cuda.synchronize()
        _assert_live_rows_equal(slots, tensors, outputs, live)
        assert not check_paged_stash_overflow().item()


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@requires_cuda
def test_sizing_keeps_one_pool_of_live_pages():
    """Paging replaces ``layers x padded`` residency with one pool per
    ``(dtype, hidden_size)`` holding live pages, however many layers or stages.

    The live count is not a page multiple, so summing tokens -- Megatron's
    accounting -- would come out short of the pages the stashes occupy. Under
    GraphPP each stage contributes its live pages to the schedule replay from
    its forward until the backward that consumes them: a full or input
    backward releases them, a weight backward does not.
    """
    actual, padded = 1000, 4000  # capacity factor 4, partial last page
    manager = _fresh_manager(buffer_size_factor_cuda=1.0)
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
    PagedStashManager.reset_instance()

    stage_pages = {0: 4 * _pages(actual), 1: 2 * _pages(actual)}
    assert footprint == {stage: {KEY: pages} for stage, pages in stage_pages.items()}
    actions = _actions("F0.0 F1.0 I0.0 W0.0 F0.1 F0.2 F1.1")
    peak = 2 * stage_pages[0] + 2 * stage_pages[1]
    assert replay_schedule_page_peak(actions, footprint) == {KEY: peak}
    assert max_live_stash_generations(actions) == 4


# ---------------------------------------------------------------------------
# Skipping the stash when a forward's own backward runs next
# ---------------------------------------------------------------------------


@requires_cuda
def test_skip_on_immediate_backward(manager):
    """A forward keeps its last paged layer resident exactly when its own
    backward is the next compute action (Megatron's
    ``remove_paged_tensor_from_stash``)."""
    # Sends and receives do not count; another microbatch's or stage's
    # backward does not match; the last forward has no successor.
    actions = _actions("F0.0 S0.0 R1.0 B0.0 F0.1 F0.2 B0.1 F1.3 B0.3 F0.4")
    assert forward_skip_plan(actions) == [True, False, False, False, False]

    # The plan is looked up by pipeline rank: with DP alongside PP it differs
    # from the global rank, which is 0 in this process.
    manager.skip_immediate_backward = True
    assert not manager.skip_enabled and not manager.should_skip_stash()
    order = {0: _actions("F0.0 F0.1 B0.0 B0.1"), 1: _actions("F1.0 B1.0 F1.1 B1.1")}
    schedule = SimpleNamespace(rank=1, pipeline_order_with_comms=order)
    assert apply_graph_pp_skip_plan(schedule) == 2
    assert manager.skip_plan == [True, True]

    # Per-forward decisions follow the plan; past its end, stash.
    manager.set_skip_plan([False, True])
    decisions = []
    for _ in range(3):
        manager.begin_forward()
        decisions.append(manager.should_skip_stash())
    assert decisions == [False, True, False]

    # A per-microbatch decision cannot be frozen into a captured graph; a
    # static one does not vary between replays, so it stays on.
    manager.begin_step()
    manager.begin_forward()
    manager.begin_forward()
    assert manager.should_skip_stash() is True
    scratch = torch.zeros(8, device="cuda")
    torch.cuda.synchronize()

    def decide_under_capture() -> bool:
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            scratch.add_(1)
            return manager.should_skip_stash()

    assert decide_under_capture() is False
    manager.set_static_skip(True)
    assert decide_under_capture() is True

    # Resident activations do not leak into the next step.
    manager.note_stash(0, resident=scratch)
    manager.begin_step()
    assert manager.take_resident(0) is None


# ---------------------------------------------------------------------------
# Slot dims, the graph pass, and the GraphPP partition
# ---------------------------------------------------------------------------


def test_slot_dims_are_inferred_or_declared():
    """How the stash addresses a tensor as rows: ``(token_divisor, hidden)``."""
    cases = [
        # A data tensor's rows are what the copy kernel addresses...
        ((2048, 256), E4M3, 2048, (1, 256)),
        # ...so the right element count with the wrong leading dim is
        # rejected: its live rows would not be a prefix.
        ((256, 2048), E4M3, 2048, None),
        # e8m0 block scales carry no token axis, only 32 tokens per row.
        ((2048 // 32 * 256,), E8M0, 2048, (32, 256)),
        # Indivisible element counts, and padding that is not whole blocks.
        ((1000,), E8M0, 2048, None),
        ((64,), E8M0, 100, None),
    ]
    for shape, dtype, padded, dims in cases:
        tensor = torch.empty(shape, dtype=dtype, device="meta")
        assert infer_slot_dims(tensor, padded) == dims, (shape, dtype, padded)

    # A producer can instead declare them, which survives tracing. Marking is
    # off unless paged stashing is enabled, so other runs trace no marks.
    x = torch.randn(64, 8, device="meta")
    assert mark_pageable(x, token_divisor=32, hidden_size=8) is x
    set_marking_enabled(True)
    try:
        gm = make_fx(
            lambda t: mark_pageable(t * 2, token_divisor=32, hidden_size=8),
            tracing_mode="fake",
        )(x)
    finally:
        set_marking_enabled(False)
    marks = [n for n in gm.graph.nodes if declared_slot_dims(n) is not None]
    assert [declared_slot_dims(n) for n in marks] == [(32, 8)]


def _fake_moe_graph(padded: int = 512):
    """A traced-shaped MoE layer, hand-built so no process group, HybridEP
    buffer, or GPU is needed: a static-capacity dispatch whose padded output
    has a forward and a backward consumer, and an e8m0 block-scale buffer
    declared with ``mark_pageable`` and read by backward.

    The 4096 scales are 16 rows of 256 by inference (``padded // 32`` rows)
    but declared as 8 rows of 512 -- the element count alone cannot tell.
    """
    import torchtitan.distributed.deepep.hybridep  # noqa: F401  (registers ops)

    fake_mode = FakeTensorMode()
    graph = torch.fx.Graph()

    def add(node, shape=None, dtype=BF16, *, backward=False):
        node.meta["custom"] = {"module_fqn": "layers.0.moe.routed_experts"}
        node.meta["val"] = shape and fake_mode.from_tensor(
            torch.empty(shape, dtype=dtype, device="meta")
        )
        if backward:
            node.meta["autograd_backward"] = True
        return node

    call = graph.call_function
    tokens = add(graph.placeholder("tokens"), (padded, 256))
    weight = add(graph.placeholder("weight"), (256, 256))
    grad = add(graph.placeholder("grad"), (padded, 256))
    dispatch = add(
        call(
            torch.ops.hybridep.dispatch.default,
            (tokens, tokens, tokens, 8, 2, "ep", True, 1.0, None),
        )
    )
    hidden = add(call(operator.getitem, (dispatch, 0)), (padded, 256))
    add(call(operator.getitem, (dispatch, 2)), (4,), torch.int64)
    fwd = add(call(torch.ops.aten.mm.default, (hidden, weight)), (padded, 256))
    bwd = add(
        call(torch.ops.aten.mm.default, (grad, hidden)), (padded, 256), backward=True
    )
    scales = add(
        call(torch.ops.aten.empty.memory_format, ([4096],), {"dtype": E8M0}),
        (4096,),
        E8M0,
    )
    marked = add(
        call(torch.ops.paged_stash.mark_pageable.default, (scales, 32, 512)),
        (4096,),
        E8M0,
    )
    scales_bwd = add(
        call(torch.ops.aten.clone.default, (marked,)), (4096,), E8M0, backward=True
    )
    graph.output((fwd, bwd, scales_bwd))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    return gm, hidden, marked, scales, bwd


def test_graph_pass_pages_tagged_activations(manager):
    """The padded activation stops crossing forward -> backward.

    That crossing is the point of the pass: under GraphPP it decides whether
    the tensor lands in ``saved_values_for_backward``. Declared slot dims win
    over inference, and every declaration is erased from the executed graph,
    whether or not the policy tagged the tensor it marks.
    """
    ops = torch.ops.paged_stash
    gm, hidden, marked, scales, bwd = _fake_moe_graph()
    hidden.meta[PAGED_STASH_TAG] = True
    marked.meta[PAGED_STASH_TAG] = True
    apply_paged_stash_pass(gm, None, page_size=PAGE_SIZE, prefetch_lookahead=0)

    targets = [node.target for node in gm.graph.nodes]
    for op in (ops.stash_, ops.wait_stash_, ops.reload, ops.wait):
        assert targets.count(op.default) == 2, op
    assert ops.mark_pageable.default not in targets
    assert not [u for u in hidden.users if _is_backward_node(u)]
    assert any(n.target is ops.wait.default for n in bwd.all_input_nodes)
    stashed = {n.args[1] for n in gm.graph.nodes if n.target is ops.stash_.default}
    assert stashed == {hidden, scales}, "the stash must read the marked producer"
    slots = {
        (s.dtype, s.hidden_size, s.token_divisor, s.max_num_tokens)
        for s in manager.slots.values()
    }
    assert slots == {(BF16, 256, 1, 512), (E8M0, 512, 32, 512)}

    gm, *_ = _fake_moe_graph()  # nothing tagged
    apply_paged_stash_pass(gm, None)
    targets = [node.target for node in gm.graph.nodes]
    assert ops.mark_pageable.default not in targets
    assert ops.stash_.default not in targets


def test_graphpp_partition_keeps_each_stash_op_on_its_side(manager):
    """GraphPP splits the rewritten graph with the stash in forward and the
    reload in backward.

    The forward stash mutates an input-free ``empty`` handle, and the reload
    reads only values forward already has. Extraction copies every node whose
    inputs are available, so without the partition's side filtering backward
    would re-create the handle, uninitialized, and forward would run the
    reload. Measure mode parks the activation in forward and hands it back in
    backward, so the split graphs must compute the joint graph's results.
    """
    ops = torch.ops.paged_stash
    (slot,) = _make_slots(manager, 1, dtype=torch.float32, max_tokens=8)
    num_tokens = torch.tensor(8)

    class _StashedSquare(torch.autograd.Function):
        # The pass's rewrite, written by hand.
        @staticmethod
        def forward(ctx, h):
            handle = torch.empty(_pages(slot.max_stash_tokens) + 1, dtype=torch.int64)
            ops.stash_(handle, h, num_tokens, slot.slot_id)
            ops.wait_stash_(handle, h, None)
            ctx.save_for_backward(handle)
            return h * h

        @staticmethod
        def backward(ctx, grad):
            (handle,) = ctx.saved_tensors
            h = ops.reload(handle, num_tokens, slot.slot_id, [8, HIDDEN], grad.dtype)
            return grad * 2.0 * ops.wait(h)

    def stage_step(x):
        loss = _StashedSquare.apply(x * 3.0).sum()
        (grad_x,) = torch.autograd.grad(loss, x)
        return [loss, grad_x]

    x = torch.randn(8, HIDDEN, requires_grad=True)
    traced = minimal_fx_tracer(stage_step)(x)
    fw, bw, meta = partition_joint_graph(traced, num_fwd_outputs=1)
    fw_targets = {node.target for node in fw.graph.nodes}
    bw_targets = {node.target for node in bw.graph.nodes}
    forward_ops = {ops.stash_.default, ops.wait_stash_.default}
    backward_ops = {ops.reload.default, ops.wait.default}
    assert forward_ops <= fw_targets and not backward_ops & fw_targets
    assert backward_ops <= bw_targets and not forward_ops & bw_targets
    assert torch.ops.aten.empty.memory_format not in bw_targets

    fw_out = fx.Interpreter(fw).boxed_run([x])
    saved = fw_out[meta.num_fwd_user_outputs :][: meta.num_saved_for_backward]
    by_name = dict(zip(meta.saved_for_backward_names, saved, strict=True))
    bw_out = fx.Interpreter(bw).boxed_run([by_name[n] for n in meta.bwd_input_names])
    joint = traced.gm(x)
    assert torch.equal(fw_out[0], joint[0])
    assert torch.equal(bw_out[0], joint[1])
    assert torch.allclose(bw_out[0], 18.0 * x)


# ---------------------------------------------------------------------------
# Overflow verdict
# ---------------------------------------------------------------------------


def _optimizers(model: nn.Module, implementation: str = "fused") -> OptimizersContainer:
    """torchtitan's default AdamW over ``model``."""
    config = replace(default_adamw(lr=0.1), implementation=implementation)
    return OptimizersContainer(config, model_parts=[model])


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

    A HybridEP token drop comes from the capacity factor, which is compiled into
    the graph, so it is reported and the step kept; the flag accumulates over
    every dispatch and is reset in place for the next step. A stash overflow
    reruns the step with paging disabled, and the next step pages again from
    buffers of the same size.
    """
    from torchtitan.distributed.deepep import hybridep

    slots = _make_slots(manager, 1)
    tensors, counts, _ = _setup_stash(manager, slots, [1000])
    probe = OverbudgetProbe(
        hybridep.check_hybridep_over_budget, hybridep.reset_hybridep_over_budget
    )
    runner = _runner(manager, "blocking", overbudget_probes=[probe])

    attempts = []

    def dropping_step():
        attempts.append("drop")
        for dropped in (0, 1, 0):  # only the middle dispatch dropped tokens
            flag = torch.tensor([dropped], dtype=torch.int32, device="cuda")
            hybridep._record_over_budget(flag)
        return _round_trip(slots, tensors, counts)

    with caplog.at_level("INFO"):
        runner(dropping_step)
    assert attempts == ["drop"] and "over budget" in caplog.text
    over_budget = hybridep.check_hybridep_over_budget()

    def overflowing_step():
        attempts.append((bool(over_budget), manager.active))
        outputs = _round_trip(slots, tensors, counts)
        if len(attempts) == 2:
            manager.overflow.fill_(1)  # as the copy kernel does when out of pages
        return outputs

    pages = manager.get_buffer(BF16, HIDDEN).num_cuda_pages
    outputs = runner(overflowing_step)
    (_, (flag_1, paged_1), (flag_2, paged_2)) = attempts
    assert not flag_1 and not flag_2, "the drop flag must be reset in place"
    assert paged_1 and not paged_2, "the rerun must run with paging disabled"
    _assert_live_rows_equal(slots, tensors, outputs, [1000])
    runner(lambda: _round_trip(slots, tensors, counts))
    assert manager.active, "the next step must page again"
    assert manager.get_buffer(BF16, HIDDEN).num_cuda_pages == pages


@requires_cuda
def test_blocking_rerun_under_a_cuda_graph(manager):
    """An overflow inside a captured step drops the graph along with the pages it
    points into and reruns the step with paging disabled; the next step captures
    a new graph over reallocated buffers of the same size. Only the rerun's
    gradient joins what earlier accumulation groups left, and no graph output
    becomes the gradient itself."""
    slots = _make_slots(manager, 1)
    tensors, counts, _ = _setup_stash(manager, slots, [1000])
    tensors[0].copy_(torch.randn_like(tensors[0]))  # finite, so sums compare
    inputs = (tensors[0], counts[0])

    def graph_grad(tensor, count):  # a gradient computed from reloaded rows
        return _round_trip(slots, [tensor], [count])[0][:PAGE_SIZE].float().sum(0)

    gm = cudagraph_pass(make_fx(graph_grad)(*inputs), inputs, static_input_indices=[])
    runner = _runner(manager, "blocking")
    weight = torch.nn.Parameter(torch.empty(HIDDEN, device="cuda"))
    weight.grad = torch.empty_like(weight)
    expected = tensors[0][:PAGE_SIZE].float().sum(0)

    def step(incoming: float) -> torch.Tensor:
        """One accumulation group, after earlier ones summed to ``incoming``."""
        weight.grad.fill_(incoming)
        runner(lambda: accumulate_param_grads_([weight], [gm(*inputs)]))
        return weight.grad

    for _ in range(2):  # warmup, then capture
        assert torch.equal(step(0), expected)
    captured = gm.forward._graph
    pages = manager.get_buffer(BF16, HIDDEN).num_cuda_pages
    counts[0].fill_(1200)  # 19 pages; the buffer holds int(16 * 1.10) = 17
    assert torch.equal(step(1), expected + 1)
    assert gm.forward._graph is None, "the rerun must run eagerly, unpaged"
    counts[0].fill_(1000)
    assert torch.equal(step(0), expected)
    assert gm.forward._graph not in (None, captured), "the next step recaptures"
    assert manager.get_buffer(BF16, HIDDEN).num_cuda_pages == pages
    assert torch.equal(step(0), expected), "a replay added into its own output"


@requires_cuda
def test_deferred_mode(manager):
    """The verdict for a step is read on a later step once its copy landed,
    never blocking, and an overflow is fatal when it is read. Clean steps raise
    nothing; an overflow on the final step, which no later step reads, is
    raised by the check at shutdown."""
    runner = _runner(manager, "deferred")
    _activate(manager)

    def step(overflow: bool = False) -> torch.Tensor:
        if overflow:
            manager.overflow.fill_(1)  # as the copy kernel does when out of pages
        return torch.zeros((), device="cuda")

    step(overflow=True)  # load every kernel a step launches: a kernel's first
    manager.overflow.zero_()  # launch would wait out the queued work below
    runner(step)
    _, drained = _drains_the_stream(lambda: runner(lambda: step(overflow=True)))
    assert not drained, "the deferred check drained the stream"
    with pytest.raises(RuntimeError, match="Paged stash buffer overflow"):
        runner(step)  # the overflowing step's copy has landed by now

    manager.overflow.zero_()  # a fresh run
    runner(step)
    runner.final_overflow_check()
    runner(lambda: step(overflow=True))
    with pytest.raises(RuntimeError, match="Paged stash buffer overflow"):
        runner.final_overflow_check()


@requires_cuda
def test_deferred_check_survives_cpu_run_ahead(manager):
    """Clean steps report nothing. With the CPU steps ahead of the GPU, a step's
    copy of the flags is overwritten before it is read; the next copy still
    carries the overflow, so the shutdown check reports it."""
    runner = _runner(manager, "deferred")
    _activate(manager)

    def overflowing_step():
        manager.overflow.fill_(1)  # as the copy kernel does when out of pages
        return torch.zeros((), device="cuda")

    # Clean steps first. With the fill below they also load every kernel the
    # later steps launch, whose first launch would wait out the queued sleep.
    manager.overflow.fill_(1)
    manager.overflow.zero_()
    for _ in range(3):
        runner(lambda: torch.zeros((), device="cuda"))
    _, flags = runner._take_verdict(blocking=True)
    assert flags == (0, 0, 0), "a clean step reported an overflow"

    torch.cuda._sleep(2_000_000_000)
    runner(overflowing_step)
    assert not runner._flags_event.query(), "the CPU must be ahead of the copy"
    runner(lambda: torch.zeros((), device="cuda"))  # overwrites the unread copy
    with pytest.raises(RuntimeError, match="Paged stash buffer overflow"):
        runner.final_overflow_check()


@requires_cuda
@pytest.mark.parametrize("implementation", ["fused", "fused_opt_states_bf16"])
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
    _runner(manager, "deferred", optimizers=_optimizers(model, "fused_opt_states_bf16"))
    for implementation in ("foreach", "for-loop"):
        optimizers = _optimizers(model, implementation)
        with pytest.raises(ValueError, match="non-fused AdamW.*overflow_check block"):
            _runner(manager, "deferred", optimizers=optimizers)
        _runner(manager, "blocking", optimizers=optimizers)
        model(torch.randn(4, HIDDEN, device="cuda")).sum().backward()
        optimizers.step()  # a found_inf it cannot honor would assert here
