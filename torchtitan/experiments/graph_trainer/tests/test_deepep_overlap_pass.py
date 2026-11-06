# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the DeepEP dispatch overlap scheduling pass.

The graphs are built by hand with the registered ``deepep`` op targets; the pass
only reorders nodes, so no op runs and no GPU or communication is needed.
"""

import operator

import pytest
import torch
import torch.fx as fx

pytest.importorskip("deep_ep")
import torchtitan.distributed.deepep.deepep  # noqa: F401  (registers the ops)
from torchtitan.experiments.graph_trainer.deepep_overlap_pass import (
    deepep_dispatch_overlap_pass,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import ordered_nodes


def _build_moe_layer_graph(
    *, mutate_input: bool = False
) -> tuple[fx.GraphModule, dict]:
    """A forward MoE layer in eager order: dispatch, routed experts, combine, shared."""
    graph = fx.Graph()
    x = graph.placeholder("x")
    topk_idx = graph.placeholder("topk_idx")
    topk_scores = graph.placeholder("topk_scores")
    w_routed = graph.placeholder("w_routed")
    w_shared = graph.placeholder("w_shared")
    launch = graph.call_function(
        torch.ops.deepep.launch_dispatch.default, (x, topk_idx, topk_scores, 8, 128)
    )
    wait = graph.call_function(
        torch.ops.deepep.wait_dispatch.default, (x, topk_idx, topk_scores, launch, 2)
    )
    recv_x = graph.call_function(operator.getitem, (wait, 0))
    handle_id = graph.call_function(operator.getitem, (wait, 4))
    routed = graph.call_function(torch.ops.aten.mm.default, (recv_x, w_routed))
    combine = graph.call_function(
        torch.ops.deepep.combine.default, (routed, handle_id, True, 128)
    )
    mutation = None
    if mutate_input:
        mutation = graph.call_function(torch.ops.aten.add_.Scalar, (x, 1))
    shared = graph.call_function(torch.ops.aten.mm.default, (x, w_shared))
    out = graph.call_function(torch.ops.aten.add.Tensor, (combine, shared))
    graph.output(out)
    gm = fx.GraphModule(torch.nn.Module(), graph)
    names = dict(
        launch=launch,
        wait=wait,
        routed=routed,
        combine=combine,
        shared=shared,
        out=out,
        mutation=mutation,
    )
    return gm, names


def test_shared_experts_move_between_launch_and_wait():
    gm, n = _build_moe_layer_graph()
    before = ordered_nodes(gm)
    assert before[n["combine"]] < before[n["shared"]]

    gm = deepep_dispatch_overlap_pass(gm)

    after = ordered_nodes(gm)
    assert after[n["launch"]] < after[n["shared"]] < after[n["wait"]]
    assert after[n["wait"]] < after[n["routed"]] < after[n["combine"]] < after[n["out"]]
    records = gm.meta["deepep_dispatch_overlap"]
    assert len(records) == 1
    assert records[0].num_filler == 1
    assert records[0].filler_targets == ("aten.mm.default",)
    assert len(list(gm.graph.nodes)) == len(before)


def test_filler_does_not_cross_a_mutation_of_its_input():
    gm, n = _build_moe_layer_graph(mutate_input=True)
    before = ordered_nodes(gm)

    gm = deepep_dispatch_overlap_pass(gm)

    after = ordered_nodes(gm)
    assert [after[node] for node in n.values()] == [before[node] for node in n.values()]
    records = gm.meta["deepep_dispatch_overlap"]
    assert records[0].num_filler == 0


def test_backward_waits_are_ignored():
    gm, n = _build_moe_layer_graph()
    n["wait"].meta["autograd_backward"] = True
    before = ordered_nodes(gm)

    gm = deepep_dispatch_overlap_pass(gm)

    assert ordered_nodes(gm) == before
    assert gm.meta["deepep_dispatch_overlap"] == []
