# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for GraphTrainer's MoE paged stashing frontend.

* its custom ops: exact round trips through them, eagerly and replayed from a
  CUDA graph;
* the GraphPP schedule: the actions it reads, and a backward it cannot match;
* the graph pass: the paged-stash region and what is paged in it, where
  Megatron's stash points land, the warning for a padded activation left
  unpaged, and the GraphPP partition of the stash ops;
* the blocking overflow check under GraphTrainer's CUDA graph pass, which
  reruns an overflowing step unpaged.

The runtime they share with the eager trainer is covered by
``tests/unit_tests/gpu/test_paged_stash.py``. End-to-end training with CUDA
graphs and with GraphPP is covered by ``integration_tests.py``.

Run with::

    pytest torchtitan/experiments/graph_trainer/tests/test_paged_stash.py
"""

import operator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from tests.unit_tests.gpu.paged_stash_utils import (
    actions,
    assert_live_rows_equal,
    BF16,
    capture_round_trip,
    E4M3,
    E8M0,
    fresh_manager,
    HIDDEN,
    make_slots,
    MAX_TOKENS,
    optimizers,
    PAGE_SIZE,
    pages,
    round_trip,
    setup_stash,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.pipelining.schedules import _Action
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.cuda_graph import (
    cuda_graph_teardown,
    NUM_CUDA_GRAPH_WARMUP_STEPS,
    reset_cuda_graphs,
    wrap_fwd_bwd_with_cuda_graph,
)
from torchtitan.distributed.deepep.paged_stash.runtime import (
    check_paged_stash_overflow,
    OverbudgetProbe,
    paged_layer_schedule,
    PagedStashManager,
    PagedStashRunner,
    ScheduleEntry,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node,
    accumulate_param_grads_,
    annotate_module_fqns,
)
from torchtitan.experiments.graph_trainer.graph_pp import partition_joint_graph
from torchtitan.experiments.graph_trainer.graph_pp.runner import (
    BACKWARD_WITH_REDUCE_GRAD,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.paged_stash_memory_policy import (
    annotate_paged_stash_region,
    apply_graph_pp_schedule,
    apply_paged_stash_pass,
    PAGED_STASH_REGION,
    PAGED_STASH_TAG,
    tag_paged_stash_activations,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="paged stash requires CUDA"
)


@pytest.fixture
def manager():
    with fresh_manager() as manager:
        yield manager


def _stash(slot, tensor, num_tokens, *, prefetch=0) -> torch.Tensor:
    """Page ``tensor`` out through the forward op, as the rewritten graph does,
    into a handle sized as the graph pass sizes it: page ids and a spill flag.
    The handle starts as -1s, so whatever the op does not write shows."""
    handle = torch.full(
        (pages(slot.max_stash_tokens) + 1,), -1, dtype=torch.int64, device="cuda"
    )
    torch.ops.paged_stash.stash_(handle, tensor, num_tokens, slot.slot_id, prefetch)
    return handle


def _reload(slot, handle, num_tokens, like: torch.Tensor) -> torch.Tensor:
    """Page an activation shaped like ``like`` back in through the backward ops."""
    out = torch.ops.paged_stash.reload(
        handle, num_tokens, slot.slot_id, list(like.shape), like.dtype
    )
    return torch.ops.paged_stash.wait_reload(out)


# ---------------------------------------------------------------------------
# The custom ops
# ---------------------------------------------------------------------------


@requires_cuda
def test_ops_round_trip_is_exact(manager):
    """The stash and reload ops page out and back bit-exactly, eagerly and
    replayed from a CUDA graph, for data and block-scale slots."""
    slots = make_slots(manager, 2) + make_slots(
        manager, 1, dtype=E8M0, token_divisor=32
    )
    live = [1997, 700, 1024]
    ops = {"stash": _stash, "reload": _reload}
    tensors, counts, measured = setup_stash(manager, slots, live, **ops)
    assert_live_rows_equal(slots, tensors, measured, live)

    manager.begin_step()
    assert_live_rows_equal(
        slots, tensors, round_trip(slots, tensors, counts, **ops), live
    )
    outputs, replay = capture_round_trip(manager, slots, tensors, counts, **ops)
    for live in ([900, 1500, 2048], [64, 1997, 32]):
        replay(live)
        assert_live_rows_equal(slots, tensors, outputs, live)
        assert not check_paged_stash_overflow().item()


# ---------------------------------------------------------------------------
# The GraphPP schedule
# ---------------------------------------------------------------------------


def test_paged_layer_schedule_rejects_a_renamed_backward():
    """A forward whose backward goes by another name -- GraphRuntime's executed
    schedule renames ``FULL_BACKWARD`` -- is rejected rather than read as
    keeping its stashes live to the end of the step."""
    layers = {0: (3, 4), 2: (9,)}
    renamed = actions("F0.0 F0.1 B0.0") + [_Action(0, BACKWARD_WITH_REDUCE_GRAD, 1)]
    with pytest.raises(ValueError, match=r"unmatched .*: \[\(0, 1\)\]"):
        paged_layer_schedule(renamed, layers)


@requires_cuda
def test_graph_pp_schedule_is_this_pipeline_ranks(manager):
    """The schedule is looked up by pipeline rank: with DP alongside PP it
    differs from the global rank."""
    make_slots(manager, 2)  # stage 0, layers 0 and 1
    manager.skip_immediate_backward = True
    order = {
        0: actions("F0.0 F0.1 B0.0 B0.1"),
        1: actions("F0.0 F1.0 B1.0 B0.0 F0.1 F0.2 B0.1 B0.2"),
    }
    apply_graph_pp_schedule(SimpleNamespace(rank=1, pipeline_order_with_comms=order))
    assert [
        [
            manager.should_skip_stash(ScheduleEntry(True, 0, layer, microbatch))
            for layer in (0, 1)
        ]
        for microbatch in range(3)
    ] == [[False, True], [False, False], [False, False]]


# ---------------------------------------------------------------------------
# The graph pass and the GraphPP partition
# ---------------------------------------------------------------------------


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
    (slot,) = make_slots(manager, 1, dtype=torch.float32, max_tokens=8)
    num_tokens = torch.tensor(8)

    class _StashedSquare(torch.autograd.Function):
        # The pass's rewrite, written by hand.
        @staticmethod
        def forward(ctx, h):
            squared = ops.group_start(h) * h
            handle = torch.empty(pages(slot.max_stash_tokens) + 1, dtype=torch.int64)
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
# Overflow verdict under GraphTrainer's CUDA graph pass
# ---------------------------------------------------------------------------


def _runner(
    manager: PagedStashManager,
    mode: str,
    *,
    optimizers: Any = None,
    **kwargs: Any,
) -> PagedStashRunner:
    """A runner as ``build_paged_stash_runner`` builds it for GraphTrainer."""
    if manager.overflow is None:
        manager.overflow = torch.zeros(1, dtype=torch.int64, device="cuda")
        manager.host_spill = torch.zeros(1, dtype=torch.int64, device="cuda")
    return PagedStashRunner(
        optimizers,
        overflow_check=mode,
        config_path="compile.paged_stash",
        **kwargs,
    )


@requires_cuda
def test_blocking_mode(manager, caplog):
    """The verdict is read before the step returns, as in Megatron.

    GraphTrainer captures the whole step in the trainer's CUDA graph and runs
    the runner outside it, as here, so one test covers both. A HybridEP token
    drop comes from the capacity factor, which is compiled into the graph, so
    it is reported and the step kept; the flag accumulates over every dispatch
    and is reset in place for the next step. A stash overflow instead resets
    the graph through ``reset_cuda_graphs``, along with the pages it points
    into, and reruns the step eagerly with paging disabled. That rerun is the
    one eager call the reset asks for, and it measures the step's live pages,
    so the next step pages again over buffers grown to fit them and captures.
    Every step's gradient is exact, including the replay that adds into its own
    output buffer.
    """
    from torchtitan.distributed.deepep import hybridep

    slots = make_slots(manager, 1)
    tensors, counts, _ = setup_stash(
        manager, slots, [1000], stash=_stash, reload=_reload
    )
    tensors[0].copy_(torch.randn_like(tensors[0]))  # finite, so sums compare
    holder = nn.Module()
    holder.weight = nn.Parameter(torch.empty(HIDDEN, device="cuda"))
    weight = holder.weight
    expected = tensors[0][:PAGE_SIZE].float().sum(0)
    # Whether each eager or captured run of the step paged. A replay runs no
    # Python, so it appends nothing.
    paged: list[bool] = []

    def forward_backward(tensor: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        paged.append(manager.active)
        reloaded = round_trip(slots, [tensor], [count], stash=_stash, reload=_reload)
        grad = reloaded[0][:PAGE_SIZE].float().sum(0)
        accumulate_param_grads_([weight], [grad])
        return grad

    graphed = wrap_fwd_bwd_with_cuda_graph(
        forward_backward,
        parameters=[weight],
        num_warmup_iterations=NUM_CUDA_GRAPH_WARMUP_STEPS,
    )
    probe = OverbudgetProbe(
        hybridep.check_hybridep_over_budget, hybridep.reset_hybridep_over_budget
    )
    runner = _runner(
        manager,
        "blocking",
        optimizers=optimizers(holder),
        overbudget_probes=[probe],
        # The whole-step graph is a CUDAGraphWrapper, so the reset that keeps
        # the shared pool is the one that reaches it.
        reset_cuda_graphs=reset_cuda_graphs,
    )

    def step(dropped: tuple[int, ...] = ()) -> list[bool]:
        """One step from the engine's starting state: every gradient None.
        ``dropped`` is the over-budget flag of each of its dispatches, recorded
        outside the captured region as the dispatch kernels' writes are."""
        paged.clear()
        weight.grad = None

        def attempt() -> torch.Tensor:
            for flag in dropped:
                hybridep._record_over_budget(
                    torch.tensor([flag], dtype=torch.int32, device="cuda")
                )
            return graphed(tensors[0], counts[0])

        runner(attempt)
        assert weight.grad is not None and torch.equal(weight.grad, expected)
        return list(paged)

    try:
        assert step() == [True] and step() == [True], "warmup runs eagerly"
        assert step() == [True], "the third step captures"
        assert step() == [], "and the fourth replays"

        with caplog.at_level("INFO"):
            assert step(dropped=(0, 1, 0)) == [], "a token drop must not rerun the step"
        assert "over budget" in caplog.text
        over_budget = hybridep.check_hybridep_over_budget()

        counts[0].fill_(1200)  # 19 pages; the buffer holds int(16 * 1.10) = 17
        assert step() == [False], "the rerun must run eagerly, with paging disabled"
        assert not over_budget.item(), "the drop flag must be reset in place"

        counts[0].fill_(1000)
        # The rerun was the one eager call the reset asked for, so the next
        # step pages again and captures, over buffers the rerun re-measured to
        # fit the step that overflowed.
        assert step() == [True], "the next step pages again, and captures"
        assert manager.get_buffer(BF16, HIDDEN).num_cuda_pages == int(
            pages(1200) * 1.10
        )
        assert step() == [], "a replay added into its own output"
    finally:
        cuda_graph_teardown()
