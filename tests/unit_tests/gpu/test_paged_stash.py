# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the MoE paged stash runtime, which the eager trainer and
GraphTrainer share (``torchtitan/distributed/deepep/paged_stash/runtime.py``):

* the kernels and buffers: exact round trips, eagerly and replayed from a CUDA
  graph; host spill and the sticky overflow flag; how long a paged activation
  lives;
* sizing: one pool of live pages per ``(dtype, hidden_size)``, and the
  pipeline schedule replay;
* the schedule: the paged-layer schedule read off a pipeline's actions, which
  stashes stay resident, and reloads prefetched one schedule entry ahead,
  across pipeline actions;
* the row layout a tensor's shape implies;
* the deferred overflow check, which skips the optimizer steps before the
  raise on device; and the optimizers each mode accepts.

The frontends are covered by ``test_paged_stash_eager.py`` and GraphTrainer's
``test_paged_stash.py``, which also test the ``blocking`` overflow check under
their CUDA graphs.

Run with::

    pytest tests/unit_tests/gpu/test_paged_stash.py
"""

import copy
import gc
import weakref
from dataclasses import replace
from typing import Any

import pytest
import torch
import torch.nn as nn
from torchtitan.components.optim.optimizer import OptimizersContainer
from torchtitan.distributed.cuda_graph import (
    cuda_graph_teardown,
    NUM_CUDA_GRAPH_WARMUP_STEPS,
    wrap_fwd_bwd_with_cuda_graph,
)
from torchtitan.distributed.deepep.paged_stash.runtime import (
    check_paged_stash_host_spill,
    check_paged_stash_overflow,
    infer_slot_dims,
    max_live_stash_generations,
    paged_layer_schedule,
    PagedStashBuffer,
    PagedStashManager,
    PagedStashRunner,
    replay_schedule_page_peak,
    ScheduleEntry,
)

from tests.unit_tests.gpu.paged_stash_utils import (
    actions,
    assert_live_rows_equal,
    BF16,
    capture_round_trip,
    drains_the_stream,
    E4M3,
    E8M0,
    fresh_manager,
    HIDDEN,
    KEY,
    live_bytes,
    make_slots,
    optimizers,
    PAGE_SIZE,
    pages,
    random_activation,
    reload,
    round_trip,
    setup_stash,
    stash,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="paged stash requires CUDA"
)


@pytest.fixture
def manager():
    with fresh_manager() as manager:
        yield manager


# ---------------------------------------------------------------------------
# Kernels and buffers
# ---------------------------------------------------------------------------


@requires_cuda
def test_round_trip_is_exact(manager):
    """Paging out and back is bit-exact for the live rows of every slot kind,
    eagerly and replayed from a CUDA graph.

    Data slots address one row per token; a block-scale slot is a flat e8m0
    buffer with one row per 32 tokens, paged in its own (uint8) pool. Live
    counts that are not page multiples leave partial pages. The measure step
    keeps only the live rows, as Megatron's capture iteration does, and hands
    them back in a tensor of the activation's shape. The kernels read the live
    counts and the
    freelists from device memory, so one capture serves steps with different
    counts, provided ``begin_step`` resets the freelists (outside the graph)
    before each replay.
    """
    slots = make_slots(manager, 2) + make_slots(
        manager, 1, dtype=E8M0, token_divisor=32
    )
    live = [1997, 700, 1024]  # block scales cover whole 32-token blocks
    tensors, counts, measured = setup_stash(manager, slots, live)
    assert_live_rows_equal(slots, tensors, measured, live)
    assert [out.shape for out in measured] == [t.shape for t in tensors]
    assert len(manager.stash_buffers) == 2, "one pool per (dtype, hidden_size)"

    manager.begin_step()
    assert_live_rows_equal(slots, tensors, round_trip(slots, tensors, counts), live)
    assert not check_paged_stash_overflow().item()

    outputs, replay = capture_round_trip(manager, slots, tensors, counts)
    for live in ([900, 1500, 2048], [64, 1997, 32]):
        replay(live)
        assert_live_rows_equal(slots, tensors, outputs, live)
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
    slots = make_slots(manager, 2)
    tensors, counts, _ = setup_stash(manager, slots, [1024, 1024])
    outputs, replay = capture_round_trip(manager, slots, tensors, counts)

    # 1024 + 1024 and 512 + 1000 tokens outgrow the 16 CUDA pages, so the second
    # stash spills; 64 + 900 tokens fit exactly.
    for live, spills in (
        ([1024, 1024], True),
        ([512, 1000], True),
        ([64, 900], False),
    ):
        replay(live)
        assert_live_rows_equal(slots, tensors, outputs, live)
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
    slots = make_slots(manager, 2)
    tensors, counts, _ = setup_stash(manager, slots, [1000, 1000])

    def stash_and_drop(slot, count):
        """Stash a fresh activation and drop it. Returns the handle, the live
        rows, and a weak reference to the activation."""
        activation = random_activation(slot)
        rows = live_bytes(activation, slot, 1000).clone()
        return stash(slot, activation, count), rows, weakref.ref(activation)

    def released(ref: weakref.ref) -> bool:
        gc.collect()
        return ref() is None

    handle_0, rows_0, ref_0 = stash_and_drop(slots[0], counts[0])
    assert not released(ref_0), "released before its copy was joined"
    manager.join_stashes()  # the next layer's group start
    assert released(ref_0), "the next group start did not release it"

    handle_1, rows_1, ref_1 = stash_and_drop(slots[1], counts[1])
    assert not released(ref_1), "released before its copy was joined"
    restored_1 = reload(slots[1], handle_1, counts[1], tensors[1])
    assert released(ref_1), "backward's first reload did not release it"
    restored_0 = reload(slots[0], handle_0, counts[0], tensors[0])
    torch.cuda.synchronize()
    assert torch.equal(live_bytes(restored_1, slots[1], 1000), rows_1)
    assert torch.equal(live_bytes(restored_0, slots[0], 1000), rows_0)

    # Under capture the release is what the feature turns on: an activation
    # held past its join would keep its padding in the graph's private pool for
    # the life of the graph, which is the per-layer residency paging removes.
    manager.begin_step()
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        _, _, ref = stash_and_drop(slots[0], counts[0])
        assert not released(ref), "released before its copy was joined"
        manager.join_stashes()
        assert released(ref), "capture kept the padded activation past its join"


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@requires_cuda
def test_sizing_keeps_one_pool_of_live_pages(manager):
    """Paging replaces ``layers x padded`` residency with one pool per
    ``(dtype, hidden_size)`` holding live pages, however many layers or stages.

    The live count is not a page multiple, so summing tokens -- Megatron's
    accounting -- would come out short of the pages the stashes occupy. With
    pipeline parallelism each stage contributes its live pages to the schedule
    replay from its forward until the backward that consumes them: a full or
    input backward releases them, a weight backward does not.
    """
    actual, padded = 1000, 4000  # capacity factor 4, partial last page
    manager.buffer_size_factor_cuda = 1.0
    slots = [
        slot
        for stage, layers in enumerate((4, 2))
        for slot in make_slots(manager, layers, max_tokens=padded, stage_index=stage)
    ]
    manager.begin_step()
    for slot in slots:
        manager.record_stash(slot, actual)
    for slot in reversed(slots):
        manager.record_reload(slot, actual)
    footprint = manager.stage_page_footprint()
    manager.begin_step()  # allocates from the measured peak
    (pool,) = manager.stash_buffers.values()
    assert pool.num_cuda_pages == len(slots) * pages(actual)

    stage_pages = {0: 4 * pages(actual), 1: 2 * pages(actual)}
    assert footprint == {stage: {KEY: n} for stage, n in stage_pages.items()}
    schedule = actions("F0.0 F1.0 I0.0 W0.0 F0.1 F0.2 F1.1")
    peak = 2 * stage_pages[0] + 2 * stage_pages[1]
    assert replay_schedule_page_peak(schedule, footprint) == {KEY: peak}
    assert max_live_stash_generations(schedule) == 4

    # ``O(layers x worst_case)`` -> ``O(worst_case + actual)``. Un-paged, every
    # layer holds its own padded buffer; a captured step pins all of them in
    # the graph pool at once. Paged, the padded buffer dies at the join, so
    # only the dispatching layer's is live and the pool holds the rest as live
    # pages. The decomposition is what matters: the worst-case term is one
    # buffer whatever the depth, so only the pool term follows the layer count.
    worst_case = padded * HIDDEN * torch.empty((), dtype=BF16).element_size()
    unpaged = len(slots) * worst_case
    paged = worst_case + pool.nbytes()
    assert paged - pool.nbytes() == worst_case, "the worst-case term must be one buffer"
    assert pool.nbytes() == len(slots) * pages(actual) * PAGE_SIZE * HIDDEN * 2
    assert paged < unpaged


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
    weight-gradient backwards add nothing."""
    schedule = actions("F0.0 F1.0 S1.0 F0.1 R1.1 B1.0 I0.0 W0.0 B0.1 F2.0 B2.0")
    layers = {0: (3, 4), 2: (9,)}  # stage 1 is dense
    assert _entries(paged_layer_schedule(schedule, layers)) == [
        "F0.3.0", "F0.4.0", "F0.3.1", "F0.4.1",
        "B0.4.0", "B0.3.0", "B0.4.1", "B0.3.1",
        "F2.9.0", "B2.9.0",
    ]  # fmt: skip


@requires_cuda
def test_schedule_decides_which_stashes_stay_resident(manager):
    """A forward keeps a layer resident exactly when the next schedule entry is
    that layer's own backward (Megatron's ``remove_paged_tensor_from_stash``),
    which only a stage's last paged layer can satisfy. A resident stash copies
    nothing and takes no pages, and its reload hands back the original
    storage. A decision read off pipeline actions differs per microbatch, so it
    cannot be frozen into a captured graph; without pipeline parallelism, or
    with a schedule recorded as it ran, every microbatch's can.
    """
    slots = make_slots(manager, 2)  # stage 0, layers 0 and 1
    tensors, counts, _ = setup_stash(manager, slots, [1000, 1000])
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
    handle = stash(slots[1], tensors[1], counts[1])
    assert torch.equal(buffer.free_list_tail - buffer.free_list_head, free_pages)
    assert not handle.any(), "a resident stash must leave its handle zeroed"
    resident = reload(slots[1], handle, counts[1], tensors[1])
    assert resident.data_ptr() == tensors[1].data_ptr()

    # A dense stage between a forward and its backward adds no entry, as dense
    # layers add none in Megatron.
    manager.set_pipeline_actions(actions("F0.0 F1.0 B1.0 B0.0 F0.1 F0.2 B0.1 B0.2"))
    assert [skips(m) for m in range(3)] == [
        [False, True],
        [False, False],
        [False, False],
    ]

    scratch = torch.zeros(8, device="cuda")
    torch.cuda.synchronize()

    def decide_under_capture(microbatch: int) -> list[bool]:
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            scratch.add_(1)
            return skips(microbatch)

    # Capture decides as an eager step does. Both frontends capture the whole
    # step, so every microbatch replays at its own place in the schedule.
    assert [decide_under_capture(m) for m in range(3)] == [
        [False, True],
        [False, False],
        [False, False],
    ]
    # A recorded schedule -- the eager frontend's -- takes precedence over the
    # pipeline actions.
    manager.set_recorded_schedule(
        [
            ScheduleEntry(forward, 0, layer, 0)
            for forward, layer in ((True, 0), (True, 1), (False, 1), (False, 0))
        ]
    )
    assert decide_under_capture(0) == [False, True]
    manager.set_recorded_schedule(None)
    manager.set_pipeline_actions(None)
    assert decide_under_capture(0) == [False, True]

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
    slots = make_slots(manager, 2)  # stage 0, layers 0 and 1
    slots.append(replace(slots[1], slot_id=manager.next_slot_id(), node_name="w"))
    manager.register_slot(slots[2])
    live = [1000, 700, 500]
    manager.buffer_size_factor_cuda = 2.0  # room for two microbatches
    tensors, counts, _ = setup_stash(manager, slots, live)
    manager.set_pipeline_actions(actions("F0.0 F0.1 B0.0 B0.1"))
    log = _PopLog(monkeypatch)

    def end_layer_backward(slot) -> None:
        """Megatron's ``PipelinePreScheduleFunction.backward``."""
        manager.prefetch_after(manager.backward_entry(slot), 1)

    handles = []
    for microbatch in range(2):
        log.phase = f"F0.{microbatch}"
        # Layer 1's last stash is its group commit, which prefetches.
        handles.append(
            [
                stash(slot, tensor, count, prefetch=int(slot is slots[2]))
                for slot, tensor, count in zip(slots, tensors, counts, strict=True)
            ]
        )

    restored = []

    def reload_for(microbatch: int, i: int) -> torch.Tensor:
        log.phase = f"B0.{microbatch} reloads {slots[i].node_name}"
        out = reload(slots[i], handles[microbatch][i], counts[i], tensors[i])
        restored.append((i, out))
        return out

    for microbatch in range(2):
        reload_for(microbatch, 1)
        log.phase = f"B0.{microbatch} ends layer 1"
        end_layer_backward(slots[2])  # before slot 2's reload
        reload_for(microbatch, 2)
        reload_for(microbatch, 0)
        log.phase = f"B0.{microbatch} ends layer 0"
        end_layer_backward(slots[0])
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
            live_bytes(out, slots[i], live[i]),
            live_bytes(tensors[i], slots[i], live[i]),
        )
    assert not check_paged_stash_overflow().item()

    manager.begin_step()
    log.launched.clear()
    log.phase = "on demand"
    handles = [stash(*args) for args in zip(slots, tensors, counts, strict=True)]
    for i in (1, 2, 0):
        reload(slots[i], handles[i], counts[i], tensors[i])
    torch.cuda.synchronize()
    assert log.launched == ["on demand"] * 3

    # Prefetching is how the stash runs under a whole-step graph, so the pops
    # the schedule placed have to survive capture: they are recorded where
    # prefetching put them, and a replay reloads the same rows without the
    # schedule being consulted again.
    manager.begin_step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = [
            stash(slot, tensor, count, prefetch=int(slot is slots[2]))
            for slot, tensor, count in zip(slots, tensors, counts, strict=True)
        ]
        reload(slots[1], captured[1], counts[1], tensors[1])
        end_layer_backward(slots[2])
        replayed = [
            reload(slots[i], captured[i], counts[i], tensors[i]) for i in (2, 0)
        ]
    for slot, tensor in zip(slots, tensors, strict=True):
        tensor.copy_(random_activation(slot))
    manager.begin_step()
    graph.replay()
    torch.cuda.synchronize()
    for out, i in zip(replayed, (2, 0), strict=True):
        assert torch.equal(
            live_bytes(out, slots[i], live[i]),
            live_bytes(tensors[i], slots[i], live[i]),
        ), "a replayed prefetched reload returned the wrong rows"
    assert not check_paged_stash_overflow().item()


# ---------------------------------------------------------------------------
# Row layout
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
        # reshape of it can still vouch for it.
        ((2048 // 32 * 256 // 512, 512), E8M0, 2048, None),
        # Indivisible element counts, and padding that is not whole blocks.
        ((1000,), E8M0, 2048, None),
        ((64,), E8M0, 100, None),
    ]
    for shape, dtype, padded, dims in cases:
        tensor = torch.empty(shape, dtype=dtype, device="meta")
        assert infer_slot_dims(tensor, padded) == dims, (shape, dtype, padded)


# ---------------------------------------------------------------------------
# Overflow verdict
# ---------------------------------------------------------------------------


def _runner(
    manager: PagedStashManager,
    mode: str,
    *,
    optimizers_container: OptimizersContainer | None = None,
    **kwargs: Any,
) -> PagedStashRunner:
    """A runner whose optimizers, unless given, own a parameter no step uses."""
    if manager.overflow is None:
        manager.overflow = torch.zeros(1, dtype=torch.int64, device="cuda")
        manager.host_spill = torch.zeros(1, dtype=torch.int64, device="cuda")
    if optimizers_container is None:
        optimizers_container = optimizers(nn.Linear(1, 1, device="cuda"))
    return PagedStashRunner(optimizers_container, overflow_check=mode, **kwargs)


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

    # The same around a step the trainer captured, which is how the default
    # policy runs. The verdict is read on the host, outside the graph, so
    # capture does not change when it lands -- but a replayed step launches no
    # Python, so the flag a replay raises has to be read on a later step.
    manager.overflow.zero_()
    runner = _runner(manager, "deferred")
    scratch = torch.zeros((), device="cuda")
    graphed = wrap_fwd_bwd_with_cuda_graph(
        lambda: scratch.add_(1),
        parameters=(),
        num_warmup_iterations=NUM_CUDA_GRAPH_WARMUP_STEPS,
    )
    try:
        for _ in range(NUM_CUDA_GRAPH_WARMUP_STEPS + 2):
            runner(graphed)  # two eager warmups, the capture, then a replay
        runner.final_overflow_check()  # clean replays report nothing
        manager.overflow.fill_(1)  # as a replayed step's copy kernel does
        runner(graphed)
        torch.cuda.synchronize()
        with pytest.raises(RuntimeError, match="Paged stash buffer overflow"):
            runner(graphed)
    finally:
        cuda_graph_teardown()


@requires_cuda
@pytest.mark.parametrize("implementation", ["fused", "fused_bf16_moments"])
def test_deferred_mode_skips_overflowed_optimizer_steps(manager, implementation):
    """A clean step updates exactly as without the runner. While an overflow is
    flagged, the optimizer step leaves the parameters and their state as they
    were and puts back the expert bias that the load-balancing pre-hook moved,
    all on device; the next step raises the overflow."""
    model = nn.Linear(HIDDEN, HIDDEN, device="cuda")
    reference = copy.deepcopy(model)
    model_optimizers = optimizers(model, implementation)
    reference_optimizers = optimizers(reference, implementation)
    expert_bias = torch.zeros(8, device="cuda")

    def update_expert_bias(*_: Any) -> None:  # the MoE load-balancing hook
        expert_bias.add_(1.0)

    # Registered before the runner's hooks, as the trainer registers it.
    model_optimizers.register_step_pre_hook(update_expert_bias)
    runner = _runner(
        manager,
        "deferred",
        optimizers_container=model_optimizers,
        expert_biases=[expert_bias],
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
        model_optimizers.step()
        model_optimizers.zero_grad()

    manager.overflow.fill_(1)  # load every kernel the timed step launches: a
    manager.overflow.zero_()  # kernel's first launch would wait out the sleep
    train_step()
    reference(inputs).sum().backward()
    reference_optimizers.step()
    params = zip(model.parameters(), reference.parameters(), strict=True)
    for param, expected in params:
        assert torch.equal(param, expected), "a clean step must be unchanged"
    assert expert_bias.eq(1.0).all()

    state = model_optimizers.optimizers[0].state[model.weight]
    before = [t.clone() for t in (model.weight, *state.values(), expert_bias)]
    _, drained = drains_the_stream(lambda: train_step(overflow=True))
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
    _runner(
        manager,
        "deferred",
        optimizers_container=optimizers(model, "fused_bf16_moments"),
    )
    for implementation in ("foreach", "for_loop"):
        container = optimizers(model, implementation)
        with pytest.raises(
            ValueError, match="non-fused AdamW.*overflow_check = 'blocking'"
        ):
            _runner(manager, "deferred", optimizers_container=container)
        _runner(manager, "blocking", optimizers_container=container)
        model(torch.randn(4, HIDDEN, device="cuda")).sum().backward()
        container.step()  # a found_inf it cannot honor would assert here
