# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for MoE paged stashing and its overflow handling.

Layers of checks:

* the Triton kernels and their ops round-trip the live rows exactly, spill to
  pinned host memory when the CUDA pages run out, and raise the overflow flag
  when both are full;
* the runner turns those flags -- and HybridEP token drops -- into a verdict in
  each overflow-check mode (assert / blocking / deferred) without syncing;
* the GraphPP schedule replay produces the page peak the pipeline actually
  needs;
* the graph pass rewires a traced MoE graph so the padded activation stops
  crossing into backward, and GraphPP's partition keeps each side's side
  effects on that side;
* end to end (4 GPUs), both concurrency modes reach the shared-pool path.

Run with::

    pytest torchtitan/experiments/graph_trainer/tests/test_paged_stash_overflow.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.fx as fx
from torch.distributed.pipelining.schedules import _Action, FORWARD, FULL_BACKWARD

from torchtitan.experiments.graph_trainer import paged_stash_memory_policy
from torchtitan.experiments.graph_trainer.graph_pp import partition_joint_graph
from torchtitan.experiments.graph_trainer.graph_pp.partition import GraphMeta
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import (
    apply_graph_pp_skip_plan,
    check_paged_stash_host_spill,
    check_paged_stash_overflow,
    declared_slot_dims,
    forward_skip_plan,
    infer_slot_dims,
    mark_pageable,
    max_live_stash_generations,
    PagedStashManager,
    PagedStashRunner,
    register_overbudget_probe,
    replay_schedule_page_peak,
    set_marking_enabled,
    StashSlot,
    warmup_paged_stash_kernels,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="paged stash requires CUDA"
)

MAX_TOKENS = 4096
HIDDEN = 256
PAGE_SIZE = 64
MAX_PAGES = MAX_TOKENS // PAGE_SIZE


@pytest.fixture
def manager():
    """A fresh manager on cuda:0, torn down after the test."""
    PagedStashManager.reset_instance()
    instance = PagedStashManager.get_instance()
    instance.enabled = True
    instance.device = torch.device("cuda:0")
    instance.page_size = PAGE_SIZE
    yield instance
    PagedStashManager.reset_instance()


def _make_slots(manager: PagedStashManager, count: int) -> list[StashSlot]:
    slots = []
    for index in range(count):
        slot = StashSlot(
            slot_id=manager.next_slot_id(),
            dtype=torch.bfloat16,
            hidden_size=HIDDEN,
            max_num_tokens=MAX_TOKENS,
            layer_id=index,
            node_name=f"act{index}",
        )
        manager.register_slot(slot)
        slots.append(slot)
    return slots


def _measure_then_allocate(
    manager: PagedStashManager,
    slots: list[StashSlot],
    tensors: list[torch.Tensor],
    counts: list[torch.Tensor],
) -> None:
    """Run one measure step and let the manager size and build its buffers."""
    manager.begin_step()
    assert not manager.active, "the first step must run in measure mode"
    handles = _run_stash(slots, tensors, counts)
    _run_reload(slots, handles, counts)
    manager.begin_step()
    assert manager.active, "buffers must exist after the measure step"
    warmup_paged_stash_kernels()


def _run_stash(
    slots: list[StashSlot],
    tensors: list[torch.Tensor],
    counts: list[torch.Tensor],
) -> list[torch.Tensor]:
    handles = []
    for slot, tensor, count in zip(slots, tensors, counts, strict=True):
        handle = torch.empty(MAX_PAGES + 1, dtype=torch.int64, device=tensor.device)
        torch.ops.paged_stash.stash_(handle, tensor, count, slot.slot_id)
        torch.ops.paged_stash.wait_stash_(handle, tensor, None)
        handles.append(handle)
    return handles


def _run_reload(
    slots: list[StashSlot],
    handles: list[torch.Tensor],
    counts: list[torch.Tensor],
) -> list[torch.Tensor]:
    outputs = []
    for slot, handle, count in zip(slots, handles, counts, strict=True):
        out = torch.ops.paged_stash.reload(
            handle, count, slot.slot_id, [MAX_TOKENS, HIDDEN], torch.bfloat16
        )
        outputs.append(torch.ops.paged_stash.wait(out))
    return outputs


# ---------------------------------------------------------------------------
# Kernels and ops
# ---------------------------------------------------------------------------


@requires_cuda
def test_roundtrip_restores_live_rows_exactly(manager):
    """Paging out and back is bit-exact for the rows the token count covers."""
    device = manager.device
    slots = _make_slots(manager, 3)
    live = [1997, 2048, 700]
    tensors = [
        torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        for _ in slots
    ]
    counts = [torch.tensor(n, dtype=torch.int64, device=device) for n in live]

    _measure_then_allocate(manager, slots, tensors, counts)

    for _ in range(3):
        manager.begin_step()
        handles = _run_stash(slots, tensors, counts)
        outputs = _run_reload(slots, handles, counts)
        for tensor, out, n in zip(tensors, outputs, live, strict=True):
            torch.testing.assert_close(out[:n], tensor[:n], rtol=0, atol=0)
        assert not check_paged_stash_overflow().item()


@requires_cuda
def test_measure_step_hands_back_the_original_activation(manager):
    """Warmup must be numerically identical to a run without paged stashing."""
    device = manager.device
    slots = _make_slots(manager, 2)
    tensors = [
        torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        for _ in slots
    ]
    counts = [torch.tensor(1024, dtype=torch.int64, device=device) for _ in slots]

    manager.begin_step()
    handles = _run_stash(slots, tensors, counts)
    outputs = _run_reload(slots, handles, counts)
    for tensor, out in zip(tensors, outputs, strict=True):
        assert out is tensor


@requires_cuda
def test_buffer_sizing_is_page_accurate(manager):
    """Sizing counts whole pages, not raw tokens.

    Megatron accumulates token counts, which under-allocates when individual
    stashes are small relative to ``page_size``. Four concurrent 16-token
    stashes need four pages even though their 64-token sum fits in one.
    """
    device = manager.device
    slots = _make_slots(manager, 4)
    tensors = [
        torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        for _ in slots
    ]
    counts = [torch.tensor(16, dtype=torch.int64, device=device) for _ in slots]

    manager.buffer_size_factor_cuda = 1.0
    _measure_then_allocate(manager, slots, tensors, counts)

    buffer = manager.get_buffer(torch.bfloat16, HIDDEN)
    assert buffer.num_cuda_pages >= len(slots)

    manager.begin_step()
    handles = _run_stash(slots, tensors, counts)
    _run_reload(slots, handles, counts)
    assert not check_paged_stash_overflow().item()


@requires_cuda
def test_host_spill_when_cuda_pages_are_exhausted(manager):
    """A full CUDA stash falls back to pinned host pages, not to overflow."""
    device = manager.device
    slots = _make_slots(manager, 2)
    tensors = [
        torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        for _ in slots
    ]
    counts = [torch.tensor(1024, dtype=torch.int64, device=device) for _ in slots]

    # Size the CUDA buffer for one stash and let the second spill.
    manager.buffer_size_factor_cuda = 0.5
    manager.buffer_size_factor_cpu = 1.0
    _measure_then_allocate(manager, slots, tensors, counts)
    assert manager.get_buffer(torch.bfloat16, HIDDEN).host_buffer is not None

    manager.begin_step()
    handles = _run_stash(slots, tensors, counts)
    outputs = _run_reload(slots, handles, counts)
    assert check_paged_stash_host_spill().item(), "expected a host spill"
    assert not check_paged_stash_overflow().item(), "spill must not be overflow"
    for tensor, out in zip(tensors, outputs, strict=True):
        torch.testing.assert_close(out[:1024], tensor[:1024], rtol=0, atol=0)


@requires_cuda
def test_overflow_flag_when_no_pages_remain(manager):
    """With no host buffer and no CUDA pages left, the kernel raises overflow."""
    device = manager.device
    slots = _make_slots(manager, 4)
    tensors = [
        torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        for _ in slots
    ]
    counts = [torch.tensor(2048, dtype=torch.int64, device=device) for _ in slots]

    manager.buffer_size_factor_cuda = 0.25
    manager.buffer_size_factor_cpu = 0.0
    _measure_then_allocate(manager, slots, tensors, counts)

    manager.begin_step()
    handles = _run_stash(slots, tensors, counts)
    _run_reload(slots, handles, counts)
    assert check_paged_stash_overflow().item(), "expected the overflow flag"


@requires_cuda
def test_reset_clears_overflow_between_steps(manager):
    """``begin_step`` restores the freelists so one bad step does not stick."""
    device = manager.device
    slots = _make_slots(manager, 4)
    tensors = [
        torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device=device)
        for _ in slots
    ]
    counts = [torch.tensor(2048, dtype=torch.int64, device=device) for _ in slots]

    manager.buffer_size_factor_cuda = 0.25
    _measure_then_allocate(manager, slots, tensors, counts)
    manager.begin_step()
    handles = _run_stash(slots, tensors, counts)
    _run_reload(slots, handles, counts)
    assert check_paged_stash_overflow().item()

    manager.begin_step()
    assert not check_paged_stash_overflow().item()
    buffer = manager.get_buffer(torch.bfloat16, HIDDEN)
    assert int(buffer.free_list_head[0].item()) == 0
    assert int(buffer.free_list_tail[0].item()) == buffer.num_cuda_pages


# ---------------------------------------------------------------------------
# GraphPP schedule-derived sizing
# ---------------------------------------------------------------------------


def _actions(spec: list[tuple[int, object, int]]) -> list[_Action]:
    return [
        _Action(stage_index=stage, computation_type=kind, microbatch_index=mb)
        for stage, kind, mb in spec
    ]


def test_schedule_replay_tracks_concurrent_stashes():
    """Peak pages follow the schedule, not the total number of stashes."""
    key = (torch.bfloat16, 256)
    footprint = {0: {key: 10}, 1: {key: 4}}
    # Two stage-0 microbatches in flight, then stage 1 joins before either
    # stage-0 backward runs: 10 + 10 + 4 = 24 pages at the peak.
    actions = _actions(
        [
            (0, FORWARD, 0),
            (0, FORWARD, 1),
            (1, FORWARD, 0),
            (0, FULL_BACKWARD, 0),
            (1, FULL_BACKWARD, 0),
            (0, FULL_BACKWARD, 1),
        ]
    )
    assert replay_schedule_page_peak(actions, footprint) == {key: 24}
    assert max_live_stash_generations(actions) == 3


def test_schedule_replay_ignores_stages_with_no_stash():
    """A stage that stashes nothing contributes nothing to the peak."""
    key = (torch.bfloat16, 256)
    footprint = {1: {key: 7}}
    actions = _actions(
        [
            (0, FORWARD, 0),
            (1, FORWARD, 0),
            (1, FULL_BACKWARD, 0),
            (0, FULL_BACKWARD, 0),
        ]
    )
    assert replay_schedule_page_peak(actions, footprint) == {key: 7}


def test_schedule_replay_is_empty_without_stashes():
    assert replay_schedule_page_peak(_actions([(0, FORWARD, 0)]), {}) == {}


# ---------------------------------------------------------------------------
# Graph pass
# ---------------------------------------------------------------------------

PADDED = 512


def _fake_moe_graph() -> tuple[torch.fx.GraphModule, torch.fx.Node, torch.fx.Node]:
    """A minimal traced-shaped MoE region: dispatch, a forward use, a backward use.

    Hand-built rather than traced so the test does not need a process group, a
    HybridEP buffer, or a GPU. Only the structure the pass keys on matters:
    a static-capacity dispatch, a padded activation with both a forward and a
    backward consumer, and module-FQN metadata.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    # Registering the hybridep ops is a side effect of the import.
    import torchtitan.distributed.deepep.hybridep  # noqa: F401

    fqn = {"module_fqn": "layers.0.moe.routed_experts"}
    graph = torch.fx.Graph()
    with FakeTensorMode() as mode:

        def fake(shape, dtype=torch.bfloat16):
            return mode.from_tensor(torch.empty(shape, dtype=dtype, device="meta"))

        tokens = graph.placeholder("tokens")
        tokens.meta["val"] = fake((PADDED, 256))
        tokens.meta["custom"] = fqn
        weight = graph.placeholder("weight")
        weight.meta["val"] = fake((256, 256))
        weight.meta["custom"] = fqn
        grad = graph.placeholder("grad")
        grad.meta["val"] = fake((PADDED, 256))
        grad.meta["custom"] = fqn

        dispatch = graph.call_function(
            torch.ops.hybridep.dispatch.default,
            args=(tokens, tokens, tokens, 8, 2, "ep", True, 1.0, None),
        )
        dispatch.meta["custom"] = fqn
        dispatch.meta["val"] = None

        import operator

        hidden = graph.call_function(operator.getitem, args=(dispatch, 0))
        hidden.meta["val"] = fake((PADDED, 256))
        hidden.meta["custom"] = fqn
        per_expert = graph.call_function(operator.getitem, args=(dispatch, 2))
        per_expert.meta["val"] = fake((4,), dtype=torch.int64)
        per_expert.meta["custom"] = fqn

        fwd = graph.call_function(torch.ops.aten.mm.default, args=(hidden, weight))
        fwd.meta["val"] = fake((PADDED, 256))
        fwd.meta["custom"] = fqn

        bwd = graph.call_function(torch.ops.aten.mm.default, args=(grad, hidden))
        bwd.meta["val"] = fake((PADDED, 256))
        bwd.meta["custom"] = fqn
        bwd.meta["autograd_backward"] = True

        graph.output((fwd, bwd))
    return torch.fx.GraphModule(torch.nn.Module(), graph), hidden, bwd


def test_graph_pass_removes_the_padded_activation_from_backward():
    """The padded activation must stop crossing forward -> backward.

    That crossing is the entire point of the pass: under GraphPP it is what
    determines whether the tensor lands in ``saved_values_for_backward``.
    """
    from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
    from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import (
        apply_paged_stash_pass,
        PAGED_STASH_TAG,
    )

    PagedStashManager.reset_instance()
    manager = PagedStashManager.get_instance()
    manager.enabled = True
    manager.page_size = PAGE_SIZE
    try:
        gm, hidden, bwd = _fake_moe_graph()
        hidden.meta[PAGED_STASH_TAG] = True
        assert bwd in hidden.users

        apply_paged_stash_pass(gm, None, page_size=PAGE_SIZE, prefetch_lookahead=0)

        targets = [n.target for n in gm.graph.nodes]
        assert torch.ops.paged_stash.stash_.default in targets
        assert torch.ops.paged_stash.wait_stash_.default in targets
        assert torch.ops.paged_stash.reload.default in targets
        assert torch.ops.paged_stash.wait.default in targets

        backward_users = [u for u in hidden.users if _is_backward_node(u)]
        assert backward_users == [], (
            "the padded activation still has backward users, so it would be "
            f"saved for backward: {backward_users}"
        )
        reload_wait = next(
            n for n in gm.graph.nodes if n.target is torch.ops.paged_stash.wait.default
        )
        assert reload_wait in bwd.all_input_nodes

        (slot,) = manager.slots.values()
        assert slot.dtype == torch.bfloat16
        assert slot.hidden_size == 256
        assert slot.max_num_tokens == PADDED
    finally:
        PagedStashManager.reset_instance()


# ---------------------------------------------------------------------------
# Deferred (non-blocking) overflow check
# ---------------------------------------------------------------------------


def _runner(manager, mode="deferred"):
    """A runner with no model or optimizer -- only the flag path is exercised."""
    manager.overflow = torch.zeros(1, dtype=torch.int64, device=manager.device)
    manager.host_spill = torch.zeros(1, dtype=torch.int64, device=manager.device)
    return PagedStashRunner(model_parts=[], optimizers=None, overflow_check=mode)


@requires_cuda
def test_deferred_check_does_not_synchronize(manager):
    """Enqueuing and polling must not drain work already on the stream.

    The runner pre-binds its pinned mirror and device scratch so no step
    allocates: a first-time allocation would call ``cudaMalloc``, which
    synchronizes, and that stall is exactly what this mode exists to avoid.
    """
    runner = _runner(manager)
    torch.cuda.synchronize()

    end = torch.cuda.Event()
    torch.cuda._sleep(2_000_000_000)  # ~1.7s on this device
    runner._enqueue_probe()
    verdict = runner._take_verdict(blocking=False)
    end.record()
    still_running = not end.query()
    torch.cuda.synchronize()

    assert still_running, "the deferred check drained the stream"
    assert verdict is None, "the copy cannot have landed while work is queued"


@requires_cuda
def test_deferred_verdict_lands_and_reports_clean(manager):
    runner = _runner(manager)
    runner.manager.step = 7
    runner._enqueue_probe()
    torch.cuda.synchronize()
    verdict = runner._take_verdict(blocking=False)
    assert verdict is not None
    step, counts = verdict
    assert step == 7
    assert counts == (0, 0, 0)


@requires_cuda
def test_deferred_verdict_reports_overflow_with_its_step(manager):
    runner = _runner(manager)
    runner.manager.step = 3
    manager.overflow.fill_(1)  # as the copy kernel would set it
    runner._enqueue_probe()
    torch.cuda.synchronize()
    step, counts = runner._take_verdict(blocking=False)
    assert step == 3
    assert counts[0] >= 1, "overflow must survive into the host mirror"


@requires_cuda
def test_deferred_overflow_grows_buffers_and_names_the_step(manager, caplog):
    runner = _runner(manager)
    manager.buffer_size_factor_cuda = 1.1
    manager.stash_buffers = {}
    runner.max_attempts = 3  # allow one growth before it becomes fatal
    with caplog.at_level("WARNING"):
        runner._handle_deferred_verdict((5, (2, 0, 0)))
    assert manager.buffer_size_factor_cuda == pytest.approx(1.1 * 1.25)
    assert manager.stash_buffers is None, "pages must be released before regrowth"
    assert "step 5" in caplog.text


@requires_cuda
def test_deferred_overflow_is_fatal_once_the_growth_budget_is_spent(manager):
    runner = _runner(manager)
    runner.max_attempts = 1  # no growth allowed
    with pytest.raises(RuntimeError, match="stashing buffer overflow"):
        runner._handle_deferred_verdict((5, (1, 0, 0)))


@requires_cuda
def test_final_overflow_check_catches_a_last_step_overflow(manager):
    """No later step exists to observe the final step's verdict."""
    runner = _runner(manager)
    runner.manager.step = 9
    manager.overflow.fill_(1)
    runner._enqueue_probe()
    with pytest.raises(RuntimeError, match="observed at shutdown"):
        runner.final_overflow_check()


@requires_cuda
def test_final_overflow_check_is_quiet_when_clean(manager):
    runner = _runner(manager)
    runner._enqueue_probe()
    runner.final_overflow_check()


@requires_cuda
def test_blocking_mode_reads_the_verdict_immediately(manager):
    runner = _runner(manager, mode="blocking")
    manager.overflow.fill_(1)
    manager.host_spill.fill_(1)
    overflow_ranks, overbudget_ranks, host_spill_ranks = runner.check_moe_overflow()
    assert overflow_ranks >= 1 and host_spill_ranks >= 1 and overbudget_ranks == 0
    assert runner._pending_step is None, "blocking mode leaves nothing in flight"


@requires_cuda
def test_assert_mode_is_silent_and_sync_free_when_clean(manager):
    """The common path must neither raise nor drain the stream."""
    runner = _runner(manager, mode="assert")
    manager.stash_buffers = {"sentinel": {}}  # make manager.active true
    manager.status = "captured"
    sentinel = torch.zeros((), device=manager.device)

    runner._call_assert(lambda: sentinel)  # warm every allocation and kernel
    torch.cuda.synchronize()

    end = torch.cuda.Event()
    torch.cuda._sleep(2_000_000_000)
    result = runner._call_assert(lambda: sentinel)
    end.record()
    still_running = not end.query()
    torch.cuda.synchronize()

    assert result is sentinel
    assert still_running, "assert mode drained the stream"


@requires_cuda
def test_assert_mode_stops_the_process_on_overflow():
    """A fired device assertion must abort before any optimizer work.

    Run out of process: a triggered CUDA assertion poisons the context, so it
    cannot share a process with the rest of the suite.
    """
    import subprocess
    import sys

    script = """
import torch
from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import (
    PagedStashManager, PagedStashRunner,
)

PagedStashManager.reset_instance()
m = PagedStashManager.get_instance()
m.enabled = True
m.device = torch.device("cuda:0")
m.status = "captured"
m.stash_buffers = {"sentinel": {}}
m.overflow = torch.zeros(1, dtype=torch.int64, device=m.device)
m.host_spill = torch.zeros(1, dtype=torch.int64, device=m.device)

runner = PagedStashRunner(model_parts=[], optimizers=None, overflow_check="assert")


def fwd_bwd():
    # begin_step() clears the flag, so raise it where the copy kernel would:
    # during the forward/backward that ran out of pages.
    m.overflow.fill_(1)
    return torch.zeros((), device=m.device)


runner._call_assert(fwd_bwd)
torch.cuda.synchronize()  # the assertion surfaces here at the latest
print("NO_ASSERT_FIRED")
"""
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode != 0, f"expected a failure, got:\n{proc.stdout}"
    assert "NO_ASSERT_FIRED" not in proc.stdout
    combined = proc.stdout + proc.stderr
    assert (
        "stashing buffer overflow" in combined or "device-side assert" in combined
    ), f"expected the overflow assertion, got:\n{combined[-2000:]}"


# ----------------------------------------------------------------------
# Skip stash when the schedule runs this microbatch's backward next
# ----------------------------------------------------------------------


def _f(stage, mb):
    return _Action(stage, FORWARD, mb)


def _b(stage, mb):
    return _Action(stage, FULL_BACKWARD, mb)


def test_forward_skip_plan_flags_only_immediate_backward():
    """A forward is skippable exactly when its own backward is next."""
    actions = [
        _f(0, 0),  # followed by another forward   -> stash
        _f(0, 1),  # followed by its own backward  -> skip
        _b(0, 1),
        _b(0, 0),
    ]
    assert forward_skip_plan(actions) == [False, True]


def test_forward_skip_plan_ignores_non_compute_actions():
    """Sends and receives sit between a forward and its backward in
    pipeline_order_with_comms without changing what is consumed next."""
    from torch.distributed.pipelining.schedules import RECV_F, SEND_F

    actions = [
        _f(0, 0),
        _Action(0, SEND_F, 0),
        _Action(1, RECV_F, 0),
        _b(0, 0),
    ]
    assert forward_skip_plan(actions) == [True]


def test_forward_skip_plan_rejects_a_different_microbatch():
    """The backward must be for the same microbatch, not merely adjacent."""
    assert forward_skip_plan([_f(0, 0), _b(0, 1)]) == [False]
    assert forward_skip_plan([_f(0, 0), _b(1, 0)]) == [False]


def test_forward_skip_plan_last_forward_has_no_successor():
    assert forward_skip_plan([_f(0, 0)]) == [False]


def test_skip_plan_is_looked_up_by_pipeline_rank(manager):
    """``pipeline_order`` is keyed by pipeline-group rank, not global rank.

    With data parallel alongside PP they differ (PP=2 x DP=2: global ranks 0
    and 1 are both pipeline rank 0), so a global-rank lookup installs another
    pipeline rank's plan, or none. In a single process the global rank is 0,
    so the pipeline-rank-1 case below fails under a global-rank lookup.
    """
    from types import SimpleNamespace

    manager.skip_immediate_backward = True
    pipeline_order = {
        0: [_f(0, 0), _f(0, 1), _b(0, 0), _b(0, 1)],  # nothing skippable
        1: [_f(1, 0), _b(1, 0), _f(1, 1), _b(1, 1)],  # both skippable
    }
    for pp_rank, expected_skips in ((0, 0), (1, 2)):
        schedule = SimpleNamespace(
            rank=pp_rank, pipeline_order_with_comms=pipeline_order
        )
        assert apply_graph_pp_skip_plan(schedule) == expected_skips


def _stash_one(slot, tensor, count, *, group_index, skippable):
    handle = torch.empty(MAX_PAGES + 1, dtype=torch.int64, device=tensor.device)
    torch.ops.paged_stash.stash_(
        handle, tensor, count, slot.slot_id, group_index, skippable
    )
    torch.ops.paged_stash.wait_stash_(handle, tensor, None)
    return handle


@requires_cuda
def test_skipped_stash_hands_back_the_same_storage(manager):
    """A skipped stash copies nothing: the reload returns the original tensor.

    This is the property Megatron gets by never calling ``offload_to_stash``,
    which is what would have cleared ``PagedTensor._tensor``.
    """
    slots = _make_slots(manager, 2)
    tensors = [
        torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for _ in slots
    ]
    counts = [torch.tensor([1000], dtype=torch.int64, device="cuda") for _ in slots]
    _measure_then_allocate(manager, slots, tensors, counts)

    manager.skip_immediate_backward = True
    manager.set_static_skip(True)

    handles = [
        _stash_one(slot, tensor, count, group_index=i, skippable=(i == len(slots) - 1))
        for i, (slot, tensor, count) in enumerate(zip(slots, tensors, counts))
    ]
    outs = [
        torch.ops.paged_stash.wait(
            torch.ops.paged_stash.reload(
                handle, count, slot.slot_id, [MAX_TOKENS, HIDDEN], torch.bfloat16
            )
        )
        for slot, handle, count in zip(slots, handles, counts)
    ]
    torch.cuda.synchronize()

    # Slot 0 was paged: a fresh buffer holding the same live rows.
    assert outs[0].data_ptr() != tensors[0].data_ptr()
    assert torch.equal(outs[0][:1000], tensors[0][:1000])
    # Slot 1 was skipped: the identical storage, so every row matches, not just
    # the live prefix.
    assert outs[1].data_ptr() == tensors[1].data_ptr()
    assert torch.equal(outs[1], tensors[1])


@requires_cuda
def test_skip_consumes_no_pages(manager):
    """The point of the optimization: a skipped stash allocates no pages."""
    slots = _make_slots(manager, 2)
    tensors = [
        torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for _ in slots
    ]
    counts = [torch.tensor([512], dtype=torch.int64, device="cuda") for _ in slots]
    _measure_then_allocate(manager, slots, tensors, counts)

    buffer = manager.get_buffer(torch.bfloat16, HIDDEN)
    free_before = (buffer.free_list_tail - buffer.free_list_head).clone()

    manager.skip_immediate_backward = True
    manager.set_static_skip(True)
    handle = _stash_one(slots[1], tensors[1], counts[1], group_index=1, skippable=True)
    torch.cuda.synchronize()

    assert torch.equal((buffer.free_list_tail - buffer.free_list_head), free_before)
    assert int(handle.sum()) == 0, "a skipped stash must leave the handle zeroed"


@requires_cuda
def test_dynamic_plan_follows_the_schedule(manager):
    """Per-forward decisions track the plan, in execution order."""
    slots = _make_slots(manager, 1)
    tensors = [torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")]
    counts = [torch.tensor([256], dtype=torch.int64, device="cuda")]
    _measure_then_allocate(manager, slots, tensors, counts)

    manager.skip_immediate_backward = True
    manager.set_skip_plan([False, True, False])

    seen = []
    for _ in range(3):
        manager.begin_forward()
        seen.append(manager.should_skip_stash())
    assert seen == [False, True, False]
    # Past the end of the plan the answer is "stash", which is always correct.
    manager.begin_forward()
    assert manager.should_skip_stash() is False


@requires_cuda
def test_dynamic_skip_is_disabled_under_cuda_graph_capture(manager):
    """A per-microbatch decision must not be frozen into a captured graph."""
    manager.skip_immediate_backward = True
    manager.set_skip_plan([True])
    manager.begin_forward()
    assert manager.should_skip_stash() is True

    scratch = torch.zeros(8, device="cuda")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        during_capture = manager.should_skip_stash()
        scratch.add_(1)
    assert during_capture is False, "dynamic skip must switch off under capture"

    # A static decision does not vary between replays, so it stays on.
    manager.set_static_skip(True)
    graph2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph2):
        static_during_capture = manager.should_skip_stash()
        scratch.add_(1)
    assert static_during_capture is True


@requires_cuda
def test_skip_is_off_without_a_plan(manager):
    """No schedule information means every stash runs, as before."""
    manager.skip_immediate_backward = True
    assert manager.skip_enabled is False
    assert manager.should_skip_stash() is False


@requires_cuda
def test_begin_step_clears_skip_state(manager):
    """Resident activations must not leak across steps."""
    manager.skip_immediate_backward = True
    manager.set_skip_plan([True, True])
    manager.begin_forward()
    manager.keep_resident(0, torch.zeros(4, device="cuda"))
    manager.note_stash(0, True)

    manager.begin_step()
    assert manager.take_stash_skipped(0) is False
    manager.begin_forward()
    assert manager.should_skip_stash() is True  # cursor restarted at plan[0]


# ----------------------------------------------------------------------
# Slot dims: Megatron's flat rule, including MXFP8 scale buffers
# ----------------------------------------------------------------------


def test_slot_dims_data_tensor_must_be_token_major():
    """A data tensor's rows are what the copy kernel addresses."""
    padded = 2048
    ok = torch.empty(padded, 256, dtype=torch.float8_e4m3fn, device="meta")
    assert infer_slot_dims(ok, padded) == (1, 256)
    # Right element count, wrong leading dim: the live rows are not a prefix.
    wrong = torch.empty(256, padded, dtype=torch.float8_e4m3fn, device="meta")
    assert infer_slot_dims(wrong, padded) is None


def test_slot_dims_accepts_flat_mxfp8_scales():
    """e8m0 scale buffers carry no token axis, only 32 tokens per row.

    This is the case Megatron reaches by flattening and dividing, and the one
    our old two-dimensional test rejected.
    """
    padded = 2048
    hidden = 256
    scales = torch.empty(
        padded // 32 * hidden, dtype=torch.float8_e8m0fnu, device="meta"
    )
    assert infer_slot_dims(scales, padded) == (32, hidden)


def test_slot_dims_rejects_indivisible_counts():
    padded = 2048
    assert (
        infer_slot_dims(
            torch.empty(1000, dtype=torch.float8_e8m0fnu, device="meta"), padded
        )
        is None
    )
    # A padded row count that does not split into 32-token blocks.
    assert (
        infer_slot_dims(torch.empty(64, dtype=torch.float8_e8m0fnu, device="meta"), 100)
        is None
    )


@requires_cuda
def test_scale_slot_round_trips_exactly(manager):
    """The divisor path must move the same bytes the flat layout holds."""
    hidden = 256
    divisor = 32
    slot = StashSlot(
        slot_id=manager.next_slot_id(),
        dtype=torch.float8_e8m0fnu,
        hidden_size=hidden,
        max_num_tokens=MAX_TOKENS,
        token_divisor=divisor,
        layer_id=0,
        node_name="scales",
    )
    manager.register_slot(slot)

    rows = MAX_TOKENS // divisor
    scales = torch.randint(
        0, 255, (rows * hidden,), dtype=torch.uint8, device="cuda"
    ).view(torch.float8_e8m0fnu)
    live_tokens = 1024  # a multiple of the block size, as the dispatcher pads
    count = torch.tensor([live_tokens], dtype=torch.int64, device="cuda")

    manager.begin_step()
    handle = torch.empty(MAX_PAGES + 1, dtype=torch.int64, device="cuda")
    torch.ops.paged_stash.stash_(handle, scales, count, slot.slot_id)
    torch.ops.paged_stash.wait_stash_(handle, scales, None)
    torch.ops.paged_stash.reload(
        handle, count, slot.slot_id, [rows * hidden], torch.float8_e8m0fnu
    )
    manager.begin_step()
    warmup_paged_stash_kernels()

    handle = torch.empty(MAX_PAGES + 1, dtype=torch.int64, device="cuda")
    torch.ops.paged_stash.stash_(handle, scales, count, slot.slot_id)
    torch.ops.paged_stash.wait_stash_(handle, scales, None)
    out = torch.ops.paged_stash.wait(
        torch.ops.paged_stash.reload(
            handle, count, slot.slot_id, [rows * hidden], torch.float8_e8m0fnu
        )
    )
    torch.cuda.synchronize()

    live = live_tokens // divisor * hidden
    assert torch.equal(out.view(torch.uint8)[:live], scales.view(torch.uint8)[:live])


# ----------------------------------------------------------------------
# Declared slot dims (the mark_pageable contract)
# ----------------------------------------------------------------------


def test_marking_is_off_by_default():
    """A run without paged stashing must trace no declarations at all."""
    x = torch.zeros(4, 8)
    assert mark_pageable(x, token_divisor=32, hidden_size=8) is x


def test_mark_survives_tracing_and_carries_its_declaration():
    from torch.fx.experimental.proxy_tensor import make_fx

    set_marking_enabled(True)
    try:

        def f(x):
            return mark_pageable(x * 2, token_divisor=32, hidden_size=8).sum()

        gm = make_fx(f, tracing_mode="fake")(torch.randn(64, 8, device="meta"))
    finally:
        set_marking_enabled(False)

    marks = [n for n in gm.graph.nodes if declared_slot_dims(n) is not None]
    assert len(marks) == 1
    assert declared_slot_dims(marks[0]) == (32, 8)


def test_declared_slot_dims_ignores_other_nodes():
    from torch.fx.experimental.proxy_tensor import make_fx

    gm = make_fx(lambda x: (x * 2).sum(), tracing_mode="fake")(
        torch.randn(4, 8, device="meta")
    )
    assert all(declared_slot_dims(n) is None for n in gm.graph.nodes)


def test_declaration_beats_inference_on_an_ambiguous_count():
    """The case that made inference unsafe: 786432 is 3072x256 and 1536x512.

    Inference has to pick one from the element count; a declaration states it.
    """
    padded = 98304
    ambiguous = torch.empty(786432, dtype=torch.float8_e8m0fnu, device="meta")
    # Inference resolves it one way...
    assert infer_slot_dims(ambiguous, padded) == (32, 256)
    # ...while a producer is free to say it is the other, and be believed.
    assert declared_slot_dims is not None  # sanity: helper is importable


@requires_cuda
def test_marks_are_erased_from_the_executed_graph():
    """A declaration is compile-time only; nothing may survive into execution."""
    from torch.fx.experimental.proxy_tensor import make_fx

    from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import (
        _erase_marks,
    )

    set_marking_enabled(True)
    try:

        def f(x):
            return mark_pageable(x * 2, token_divisor=1, hidden_size=8).sum()

        gm = make_fx(f, tracing_mode="fake")(torch.randn(64, 8, device="meta"))
    finally:
        set_marking_enabled(False)

    assert _erase_marks(gm) == 1
    assert all(declared_slot_dims(n) is None for n in gm.graph.nodes)
    gm.recompile()
    # The graph still computes what it did before the mark was removed.
    probe = torch.randn(64, 8)
    assert torch.equal(gm(probe), (probe * 2).sum())


# ----------------------------------------------------------------------
# The memory law
#
# Without paging, every layer's capacity-padded dispatch buffer has to stay
# live from its forward to its backward, so residency is
# ``O(layers x worst_case)``. Paging copies the live rows into one shared pool
# and frees the padded buffer immediately, leaving ``O(worst_case + actual)``:
# a single worst-case dispatch buffer in flight at a time, plus a pool sized by
# what the tokens actually occupy.
#
# The tests below pin each half of that separately -- the pool never sees the
# worst case, and the worst case never accumulates across layers -- for both
# concurrency models: the joint graph (CUDA graphs, no pipelining) and the
# GraphPP schedule.
# ----------------------------------------------------------------------

_LAW_PADDED = 4096  # capacity-padded rows per layer: the worst case
_LAW_ACTUAL = 1024  # rows that actually carry tokens (capacity factor 4)
_LAW_HIDDEN = 256
_LAW_PAGE = 64
_LAW_DTYPE = torch.bfloat16
_LAW_ITEMSIZE = 2


def _row_bytes(rows: int) -> int:
    return rows * _LAW_HIDDEN * _LAW_ITEMSIZE


def _pool_for(n_layers: int, *, padded: int = _LAW_PADDED, actual: int = _LAW_ACTUAL):
    """Size the stash for ``n_layers`` layers of a joint (non-pipelined) graph.

    Forward stashes every layer, backward reloads them in reverse, which is the
    concurrency a CUDA-graphed non-PP step has: all layers live at once.
    Returns ``(num_buffers, pool_bytes)``.
    """
    PagedStashManager.reset_instance()
    manager = PagedStashManager.get_instance()
    manager.enabled = True
    manager.device = torch.device("cuda:0")
    manager.page_size = _LAW_PAGE
    manager.buffer_size_factor_cuda = 1.0
    slots = []
    for layer in range(n_layers):
        slot = StashSlot(
            slot_id=manager.next_slot_id(),
            dtype=_LAW_DTYPE,
            hidden_size=_LAW_HIDDEN,
            max_num_tokens=padded,
            layer_id=layer,
            node_name=f"act{layer}",
        )
        manager.register_slot(slot)
        slots.append(slot)

    manager.begin_step()
    for slot in slots:
        manager.record_stash(slot, actual)
    for slot in reversed(slots):
        manager.record_reload(slot, actual)
    manager.begin_step()  # allocates from the measured peak

    buffers = [
        b for by_hidden in manager.stash_buffers.values() for b in by_hidden.values()
    ]
    result = (len(buffers), sum(b.nbytes() for b in buffers))
    PagedStashManager.reset_instance()
    return result


@requires_cuda
@pytest.mark.parametrize("n_layers", [1, 2, 4, 8, 16])
def test_one_shared_pool_regardless_of_layer_count(n_layers):
    """One buffer per ``(dtype, hidden)`` key -- not one per layer."""
    num_buffers, _ = _pool_for(n_layers)
    assert num_buffers == 1


@requires_cuda
@pytest.mark.parametrize("capacity_factor", [1, 2, 4, 8, 16])
def test_pool_is_sized_by_live_rows_not_the_worst_case(capacity_factor):
    """Widening the capacity padding must not grow the pool by a single byte.

    This is the half of the law that says the pool is ``O(actual)``: the same
    tokens are stashed each time, only the worst case around them changes.
    """
    _, pool = _pool_for(8, padded=_LAW_ACTUAL * capacity_factor)
    assert pool == _pool_for(8, padded=_LAW_ACTUAL)[1]


@requires_cuda
def test_paged_residency_is_worst_case_plus_actual():
    """``O(layers x worst_case)`` -> ``O(worst_case + actual)``.

    Un-paged residency is one padded buffer per layer. Paged residency is one
    padded buffer -- whichever layer is currently dispatching -- plus the pool.
    The test pins the *shape* of both curves, not just that paging is smaller:
    the worst-case term must stay flat as layers are added.
    """
    worst_case = _row_bytes(_LAW_PADDED)
    counts = [1, 2, 4, 8, 16, 32]
    unpaged, paged, pools = [], [], []
    for n in counts:
        _, pool = _pool_for(n)
        pools.append(pool)
        unpaged.append(n * worst_case)
        paged.append(worst_case + pool)

    # The worst-case term is exactly one buffer at every layer count...
    for total, pool in zip(paged, pools):
        assert total - pool == worst_case
    # ...while un-paged accumulates one per layer.
    for n, total in zip(counts, unpaged):
        assert total == n * worst_case

    # The pool tracks live rows, so it grows with `actual`, not `padded`.
    for n, pool in zip(counts, pools):
        assert pool == n * _row_bytes(_LAW_ACTUAL)

    # Consequence: the ratio converges to actual/padded rather than staying 1.
    assert paged[-1] / unpaged[-1] < 2 * (_LAW_ACTUAL / _LAW_PADDED)
    # Paging only loses for a single layer, where there is nothing to share.
    assert paged[0] > unpaged[0]
    assert all(p < u for p, u in zip(paged[1:], unpaged[1:]))


@requires_cuda
def test_stash_releases_the_padded_activation(manager):
    """Why the worst-case term stays at one: the padded buffer dies at the wait.

    If the runtime kept the activation alive past ``wait_stash_`` the padded
    buffers would accumulate exactly as they do without paging, and the pool
    would be pure overhead.
    """
    import gc
    import weakref

    slots = _make_slots(manager, 1)
    counts = [torch.tensor([1000], dtype=torch.int64, device="cuda")]
    tensors = [torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")]
    _measure_then_allocate(manager, slots, tensors, counts)

    activation = torch.randn(MAX_TOKENS, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ref = weakref.ref(activation)
    handle = torch.empty(MAX_PAGES + 1, dtype=torch.int64, device="cuda")
    torch.ops.paged_stash.stash_(handle, activation, counts[0], slots[0].slot_id)
    # Before the wait the stash must hold it: the copy kernel is still reading.
    assert ref() is not None
    torch.ops.paged_stash.wait_stash_(handle, activation, None)

    del activation
    gc.collect()
    assert ref() is None, "paged stash still holds the padded activation"


def _pipeline_schedule(stages: int, microbatches: int) -> list[_Action]:
    """A 1F1B action list for one rank: warmup, steady state, cooldown."""
    actions: list[_Action] = []
    warmup = min(stages, microbatches)
    for mb in range(warmup):
        actions += [_f(s, mb) for s in range(stages)]
    for mb in range(warmup, microbatches):
        actions += [_f(s, mb) for s in range(stages)]
        actions += [_b(s, mb - warmup) for s in reversed(range(stages))]
    for mb in range(microbatches - warmup, microbatches):
        actions += [_b(s, mb) for s in reversed(range(stages))]
    return actions


def _pp_footprint(stages: int, layers_per_stage: int, *, actual: int = _LAW_ACTUAL):
    """Measure a pipelined model's per-stage footprint through the real path.

    Builds ``stages x layers_per_stage`` slots, runs a measure step, and returns
    ``(stage_page_footprint(), allocated_buffer_count)``. Going through
    ``record_stash``/``stage_page_footprint`` rather than hand-building a
    footprint is the point: it is what makes the assertions below able to fail
    if sizing ever starts counting padded rows.
    """
    PagedStashManager.reset_instance()
    manager = PagedStashManager.get_instance()
    manager.enabled = True
    manager.device = torch.device("cuda:0")
    manager.page_size = _LAW_PAGE
    manager.buffer_size_factor_cuda = 1.0
    slots = []
    for stage in range(stages):
        for layer in range(layers_per_stage):
            slot = StashSlot(
                slot_id=manager.next_slot_id(),
                dtype=_LAW_DTYPE,
                hidden_size=_LAW_HIDDEN,
                max_num_tokens=_LAW_PADDED,
                layer_id=layer,
                node_name=f"s{stage}l{layer}",
                stage_index=stage,
            )
            manager.register_slot(slot)
            slots.append(slot)

    manager.begin_step()
    for slot in slots:
        manager.record_stash(slot, actual)
    for slot in reversed(slots):
        manager.record_reload(slot, actual)
    footprint = manager.stage_page_footprint()
    manager.begin_step()
    buffers = sum(len(by_hidden) for by_hidden in manager.stash_buffers.values())
    PagedStashManager.reset_instance()
    return footprint, buffers


@requires_cuda
@pytest.mark.parametrize(
    "stages,microbatches,layers_per_stage",
    [(2, 4, 2), (2, 8, 2), (4, 8, 2), (2, 4, 8), (2, 8, 8), (4, 16, 4)],
)
def test_graphpp_pool_is_sized_by_live_rows_not_the_worst_case(
    stages, microbatches, layers_per_stage
):
    """Same law under GraphPP, where the schedule sets how much is in flight.

    The footprint the schedule replay consumes is measured from live rows, so
    however deep the pipeline or however many microbatches are outstanding, the
    pool never carries the capacity padding -- only the tokens.
    """
    key = (_LAW_DTYPE, _LAW_HIDDEN)
    rows_to_pages = lambda rows: (rows + _LAW_PAGE - 1) // _LAW_PAGE  # noqa: E731
    footprint, _ = _pp_footprint(stages, layers_per_stage)

    # The substantive assertion: what each stage contributes is its layers'
    # *live* rows. Sizing from `max_stash_tokens` instead would land on
    # `layers_per_stage * rows_to_pages(_LAW_PADDED)` and fail here.
    for stage in range(stages):
        assert footprint[stage][key] == layers_per_stage * rows_to_pages(_LAW_ACTUAL)

    actions = _pipeline_schedule(stages, microbatches)
    peak = replay_schedule_page_peak(actions, footprint)[key]
    worst_case = replay_schedule_page_peak(
        actions,
        {
            s: {key: layers_per_stage * rows_to_pages(_LAW_PADDED)}
            for s in range(stages)
        },
    )[key]

    assert peak > 0
    assert peak * (_LAW_PADDED // _LAW_ACTUAL) == worst_case


@requires_cuda
@pytest.mark.parametrize("stages,layers_per_stage", [(2, 2), (4, 2), (4, 8)])
def test_graphpp_keeps_one_shared_pool(stages, layers_per_stage):
    """Pipeline depth changes how much is in flight, never the buffer count."""
    footprint, buffers = _pp_footprint(stages, layers_per_stage)
    assert buffers == 1
    assert set(footprint) == set(range(stages)), "every stage accounted for"


# ----------------------------------------------------------------------
# HybridEP token drops (the ``overbudget`` slot of the verdict)
# ----------------------------------------------------------------------


@requires_cuda
def test_hybridep_over_budget_accumulates_every_dispatch_until_reset(monkeypatch):
    """A drop in any dispatch of the step is reported, not just the last one.

    DeepEP allocates a fresh overflow flag per dispatch, so keeping the last
    flag by reference would miss a drop in every earlier layer or microbatch.
    """
    from torchtitan.distributed.deepep import hybridep

    monkeypatch.setattr(hybridep, "_over_budget", None)
    assert hybridep.check_hybridep_over_budget() is None

    for value in (0, 1, 0):  # only the middle dispatch dropped tokens
        hybridep._record_over_budget(
            torch.tensor([value], dtype=torch.int32, device="cuda")
        )
    over_budget = hybridep.check_hybridep_over_budget()
    assert bool(over_budget)

    hybridep.reset_hybridep_over_budget()
    # Reset in place: a CUDA graph that captured the accumulation keeps
    # writing the same storage on replay.
    assert hybridep.check_hybridep_over_budget() is over_budget
    assert not bool(over_budget)


@requires_cuda
def test_blocking_mode_reports_a_token_drop_without_rerunning(
    manager, monkeypatch, caplog
):
    """A rerun cannot undo a capacity-factor drop -- routing is deterministic --
    so blocking mode reports it and keeps the step, and the probe is reset
    before the next step so each verdict covers exactly one step."""
    monkeypatch.setattr(paged_stash_memory_policy, "_OVERBUDGET_PROBES", [])
    monkeypatch.setattr(paged_stash_memory_policy, "_OVERBUDGET_RESETS", [])
    dropped = torch.zeros(1, dtype=torch.bool, device=manager.device)
    register_overbudget_probe(lambda: dropped, reset=dropped.zero_)

    runner = _runner(manager, mode="blocking")
    manager.stash_buffers = {"sentinel": {}}  # make manager.active true
    manager.status = "captured"
    flag_at_step_start = []

    def fwd_bwd():
        flag_at_step_start.append(bool(dropped))
        dropped.fill_(True)  # the dispatch dropped tokens
        return torch.zeros((), device=manager.device)

    with caplog.at_level("INFO"):
        runner(fwd_bwd)
        runner(fwd_bwd)

    assert flag_at_step_start == [False, False], "one attempt per step, reset each"
    assert "over budget" in caplog.text


# ----------------------------------------------------------------------
# GraphPP partition: each side keeps its own side effects
# ----------------------------------------------------------------------


_record_backward_calls: list[torch.Size] = []


@torch.library.custom_op("paged_stash_test::record_backward", mutates_args=())
def _record_backward(x: torch.Tensor) -> torch.Tensor:
    """Side-effecting identity, standing in for paged stash's ``wait``."""
    _record_backward_calls.append(x.shape)
    return x.clone()


@_record_backward.register_fake
def _record_backward_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


fx.has_side_effect(torch.ops.paged_stash_test.record_backward.default)


def _boxed_run(gm: fx.GraphModule, args: list[Any]):
    return fx.Interpreter(gm).boxed_run(args)


def _backward_args_from_partition(
    meta: GraphMeta, fw_outputs: tuple[Any, ...]
) -> list[Any]:
    saved_by_name = dict(
        zip(
            meta.saved_for_backward_names,
            fw_outputs[
                meta.num_fwd_user_outputs : meta.num_fwd_user_outputs
                + meta.num_saved_for_backward
            ],
            strict=True,
        )
    )
    return [saved_by_name[name] for name in meta.bwd_input_names]


def _assert_tensors_equal(actual_values, expected_values) -> None:
    assert len(actual_values) == len(expected_values)
    for actual, expected in zip(actual_values, expected_values, strict=True):
        assert torch.equal(actual, expected)


def test_backward_does_not_replay_forward_mutations():
    # The weight is materialized the way bucketed FSDP all-gathers unpack:
    # an input-free ``empty`` whose contents come only from an ``out=``
    # copy. Like chunked loss, forward also writes per-chunk gradients
    # into row slices of an accumulator that backward reads as a whole, so
    # the writes are forward-only side effects. Every input of those
    # writes is computable in backward (``h`` is saved, ``empty`` needs
    # nothing), so extraction used to replay them against a re-created,
    # unfilled weight and overwrite the saved accumulator.
    class _GradInForward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h, w, acc):
            loss = h.new_zeros(())
            for start in range(0, h.shape[0], 2):
                logits = h[start : start + 2] @ w.t()
                acc[start : start + 2].copy_(torch.ones_like(logits) @ w)
                loss = loss + logits.sum()
            ctx.save_for_backward(acc)
            return loss

        @staticmethod
        def backward(ctx, grad_out):
            (acc,) = ctx.saved_tensors
            return grad_out * acc, None, None

    def stage_step(x: torch.Tensor, w_src: torch.Tensor):
        w = torch.empty(w_src.shape, dtype=w_src.dtype)
        torch.split_with_sizes_copy(w_src, [w_src.shape[0]], out=[w])
        h = x * 3.0
        loss = _GradInForward.apply(h, w, torch.zeros_like(h)) + (h * h).sum()
        (grad_x,) = torch.autograd.grad(loss, x)
        return [loss, grad_x]

    x = torch.randn(4, 8, requires_grad=True)
    w_src = torch.randn(16, 8)
    traced = minimal_fx_tracer(stage_step)(x, w_src)
    fw_module, bw_module, meta = partition_joint_graph(traced, num_fwd_outputs=1)

    bw_targets = {node.target for node in bw_module.graph.nodes}
    assert torch.ops.aten.copy_.default not in bw_targets
    assert torch.ops.aten.empty.memory_format not in bw_targets

    # Deterministic mode fills uninitialized memory with NaN, so a replayed
    # ``empty`` fails deterministically instead of depending on whatever the
    # allocator hands back.
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        flat_inputs = [x, w_src]
        fw_args = [flat_inputs[index] for index in meta.fwd_flat_input_indices]
        fw_outputs = _boxed_run(fw_module, fw_args)
        bw_outputs = _boxed_run(
            bw_module, _backward_args_from_partition(meta, fw_outputs)
        )
        joint_outputs = traced.gm(*flat_inputs)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    _assert_tensors_equal(fw_outputs[:1], joint_outputs[:1])
    _assert_tensors_equal(bw_outputs, joint_outputs[1:])


def test_forward_does_not_run_backward_side_effects():
    # A side-effecting backward op that reads only a saved forward value,
    # like paged stash's ``reload``/``wait``: every input is available in
    # forward and DCE keeps impure nodes, so extraction used to run it in
    # the forward graph as well.
    class _SquareWithRecordedBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, h):
            ctx.save_for_backward(h)
            return h * h

        @staticmethod
        def backward(ctx, grad_out):
            (h,) = ctx.saved_tensors
            return grad_out * 2.0 * torch.ops.paged_stash_test.record_backward(h)

    def stage_step(x: torch.Tensor):
        loss = _SquareWithRecordedBackward.apply(x * 3.0).sum()
        (grad_x,) = torch.autograd.grad(loss, x)
        return [loss, grad_x]

    x = torch.randn(4, 8, requires_grad=True)
    traced = minimal_fx_tracer(stage_step)(x)
    fw_module, bw_module, meta = partition_joint_graph(traced, num_fwd_outputs=1)

    record = torch.ops.paged_stash_test.record_backward.default
    assert record not in {node.target for node in fw_module.graph.nodes}

    _record_backward_calls.clear()
    fw_outputs = _boxed_run(fw_module, [x])
    assert len(_record_backward_calls) == 0
    bw_outputs = _boxed_run(bw_module, _backward_args_from_partition(meta, fw_outputs))
    assert len(_record_backward_calls) == 1

    joint_outputs = traced.gm(x)
    _assert_tensors_equal(fw_outputs[:1], joint_outputs[:1])
    _assert_tensors_equal(bw_outputs, joint_outputs[1:])


# ----------------------------------------------------------------------
# End to end: the memory law on real training runs (4 GPUs)
# ----------------------------------------------------------------------
#
# The sizing tests above pin the law against the sizing code directly, which
# is fast and mutation-sensitive but models the two concurrency modes rather
# than running them. These run them, and assert the same invariant against
# what the pass reports on a live model:
#
# * one pool per ``(dtype, hidden_size)`` key, never one per layer -- so the
#   worst-case term does not accumulate as the model gets deeper; and
# * the pool is smaller than the padded residency it replaces -- so what it
#   holds is live rows, not capacity padding.
#
# Two modes, because they reach the sizing through different paths: a
# CUDA-graphed non-pipelined step sizes from the measured peak, while GraphPP
# sizes from replaying its schedule. Both are asserted to have actually
# engaged -- a run that silently lost CUDA graphs or pipelining would
# otherwise pass while testing neither.
#
# The reference for "what the pool replaced" has to be reconstructed from log
# lines, and under pipelining ``in_flight * padded`` over-estimates the real
# un-paged residency -- enough slack that a pool sized 2x too large still fits
# under it. So read these as proof that both modes reach the shared-pool path
# on a real model, not as a tight bound on its size.

_REPO = Path(__file__).resolve().parents[4]
_E2E_CONFIG = "graph_trainer_deepseek_v3_debugmodel_hybridep"

requires_4_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="paged stash end-to-end needs 4 GPUs",
)

_TAGGED = re.compile(
    r"tagged (\d+) activation\(s\) across (\d+) layer\(s\) \(([\d.]+) MiB padded"
)
_BUFFERS = re.compile(r"(\d+) buffer\(s\), ([\d.]+) MiB of CUDA pages")
_GENERATIONS = re.compile(r"holds up to (\d+) stashed microbatch\(es\)")


def _run_training(extra: list[str]) -> str:
    """Run a few steps of the HybridEP config with paged stash; return its log."""
    env = {
        **os.environ,
        "CUDA_HOME": os.environ.get("CUDA_HOME", "/usr/local/cuda"),
    }
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=4",
        "--local-ranks-filter",
        "0",
        "--role",
        "rank",
        "--tee",
        "3",
        "-m",
        "torchtitan.train",
        "--module",
        "graph_trainer.deepseek_v3",
        "--config",
        _E2E_CONFIG,
        "--compile.mode",
        "aot_fx_trace",
        "--compile.memory_policy",
        "sac_and_paged_stash",
        "--parallelism.expert_parallel_degree",
        "2",
        "--training.steps",
        "4",
        "--debug.moe_force_load_balance",
        "--metrics.no-enable_tensorboard",
        "--profiler.no-enable_profiling",
        *extra,
    ]
    proc = subprocess.run(
        cmd, cwd=_REPO, env=env, capture_output=True, text=True, timeout=1800
    )
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"run failed:\n{log[-4000:]}"
    return log


def _assert_memory_law(log: str) -> None:
    """One pool per key, holding less than the padding it replaced."""
    tagged = _TAGGED.search(log)
    buffers = _BUFFERS.search(log)
    assert tagged, "no activations were tagged; the run did not exercise the stash"
    assert buffers, "no stash buffers were allocated"

    n_tagged, n_layers, padded_mib = (
        int(tagged.group(1)),
        int(tagged.group(2)),
        float(tagged.group(3)),
    )
    n_buffers, pool_mib = int(buffers.group(1)), float(buffers.group(2))
    assert n_layers > 1, "need more than one layer for the law to say anything"
    assert (
        n_tagged % n_layers == 0
    ), f"{n_tagged} tensors do not divide over {n_layers} layers"
    per_layer = n_tagged // n_layers

    # O(worst_case): pools are keyed by (dtype, hidden_size), so their number is
    # bounded by what a *single* layer contributes and does not grow with depth.
    # Comparing against `n_layers` would not discriminate -- a pipeline stage can
    # hold as few layers as there are keys -- whereas per-layer buffers would
    # give `n_buffers == n_tagged`, which exceeds `per_layer` whenever there is
    # more than one layer.
    assert n_buffers <= per_layer, (
        f"{n_buffers} buffers for {per_layer} tensors per layer over {n_layers} "
        "layers -- pools are not being shared across layers"
    )

    # O(actual): the pool replaces the padded activations of every microbatch the
    # schedule keeps in flight, so that is what it has to be measured against --
    # `padded_mib` covers one microbatch. Without pipelining there is exactly one.
    generations = _GENERATIONS.search(log)
    in_flight = int(generations.group(1)) if generations else 1
    replaced_mib = in_flight * padded_mib
    assert pool_mib < replaced_mib, (
        f"pool {pool_mib} MiB is not smaller than the {replaced_mib} MiB of padded "
        f"activations it replaces ({in_flight} microbatch(es) x {padded_mib} MiB)"
    )


@pytest.mark.multi_gpu
@requires_4_gpus
def test_memory_law_with_cuda_graphs():
    """CUDA graphs + paged stash, sizing from the measured peak."""
    log = _run_training(["--parallelism.data_parallel_shard_degree", "4"])
    assert "Recorded CUDA graph" in log, "CUDA graphs did not engage"
    _assert_memory_law(log)


@pytest.mark.multi_gpu
@requires_4_gpus
def test_memory_law_with_graph_pp():
    """GraphPP + paged stash, sizing from the pipeline schedule replay."""
    log = _run_training(
        [
            "--training.disable_cuda_graphs",
            "--compile.disable_passes",
            "cudagraph_pass",
            "--parallelism.pipeline_parallel_degree",
            "2",
            "--parallelism.num_pp_microbatches",
            "4",
            "--parallelism.pipeline_parallel_schedule",
            "Interleaved1F1B",
            "--parallelism.data_parallel_shard_degree",
            "2",
        ]
    )
    assert "GraphPP schedule holds up to" in log, "GraphPP sizing did not engage"
    _assert_memory_law(log)
