# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.traceback import preserve_node_meta
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer
from torchtitan.experiments.graph_trainer.subgraph_regions import (
    apply_subgraph_region_annotations_pass,
    subgraph,
    SUBGRAPH_REGION,
    SUBGRAPH_REGION_ROLE,
)


def _annotate_region(nodes, region):
    for node in nodes:
        node.meta.setdefault("custom", {})
        node.meta["custom"][SUBGRAPH_REGION] = region
        node.meta["custom"][SUBGRAPH_REGION_ROLE] = "loss_chunk"


def _invoke_subgraph_nodes(gm):
    return list(
        gm.graph.find_nodes(
            op="call_function",
            target=torch.ops.higher_order.invoke_subgraph,
        )
    )


def _subgraph_modules(gm):
    return [
        module
        for module in gm.modules()
        if isinstance(module, fx.GraphModule) and module is not gm
    ]


class TestSubgraphRegions(TestCase):
    def test_dedup_is_scoped_to_owning_module(self):
        def make_submodule():
            graph = fx.Graph()
            x = graph.placeholder("x")
            sin = graph.call_function(torch.ops.aten.sin.default, args=(x,))
            cos = graph.call_function(torch.ops.aten.cos.default, args=(sin,))
            graph.output(cos)
            gm = fx.GraphModule(torch.nn.Module(), graph)
            _annotate_region((sin, cos), "chunk")
            return gm

        root = torch.nn.Module()
        root.left = make_submodule()
        root.right = make_submodule()
        graph = fx.Graph()
        x = graph.placeholder("x")
        left = graph.call_module("left", args=(x,))
        right = graph.call_module("right", args=(x,))
        graph.output((left, right))
        gm = fx.GraphModule(root, graph)

        apply_subgraph_region_annotations_pass(gm)

        for module in (gm.left, gm.right):
            invoke_node = _invoke_subgraph_nodes(module)[0]
            attr_node = invoke_node.args[0]
            self.assertTrue(hasattr(module, attr_node.target))

    def test_subgraph_context_outlines_forward_and_backward_regions(self):
        def train_step(x):
            x = x.detach().requires_grad_()
            with subgraph("loss_chunk_0", role="loss_chunk"):
                y = torch.ops.aten.sin.default(x)
                loss = torch.ops.aten.sum.default(y)
            (grad,) = torch.autograd.grad(loss, (x,))
            return loss.detach(), grad

        x = torch.randn(4)
        gm = minimal_fx_tracer(train_step)(x).gm

        apply_subgraph_region_annotations_pass(gm)

        invoke_nodes = _invoke_subgraph_nodes(gm)
        self.assertEqual(len(invoke_nodes), 2)
        for node in invoke_nodes:
            self.assertEqual(node.meta[SUBGRAPH_REGION], "loss_chunk_0_loss_chunk")
            self.assertEqual(node.meta["custom"]["subgraph_region_id"], "loss_chunk_0")
            self.assertEqual(node.meta["custom"]["subgraph_region_role"], "loss_chunk")

        subgraph_modules = _subgraph_modules(gm)
        self.assertEqual(len(subgraph_modules), 2)
        self.assertEqual(
            {
                any(node.meta.get("autograd_backward") for node in module.graph.nodes)
                for module in subgraph_modules
            },
            {False, True},
        )
        torch.testing.assert_close(gm(x), train_step(x))

    def test_preserve_order_outlines_higher_order_operator(self):
        def true_fn(x):
            return x.sin()

        def false_fn(x):
            return x.cos()

        def f(pred, x):
            with subgraph("ordered", preserve_order=True):
                x = x + 1
                x = torch.cond(pred, true_fn, false_fn, (x,))
                return x * 2

        pred = torch.tensor(True)
        x = torch.randn(4)
        with preserve_node_meta():
            gm = make_fx(f)(pred, x)

        apply_subgraph_region_annotations_pass(gm)

        invoke_nodes = _invoke_subgraph_nodes(gm)
        self.assertEqual(len(invoke_nodes), 1)
        nested_config = invoke_nodes[0].meta["custom"]["nested_region_config"]
        self.assertEqual(
            nested_config.inductor_config_patches,
            {
                "reorder_for_locality": False,
                "reorder_for_peak_memory": False,
                "reorder_for_compute_comm_overlap": False,
                "fusion_memory_timeline_peak_allowed_increase_mb": None,
                "aten_distributed_optimizations.enable_simple_overlap": False,
                "aten_distributed_optimizations.enable_overlap_scheduling": False,
            },
        )
        self.assertEqual(gm(pred, x), f(pred, x))

    def test_structurally_identical_subgraph_regions_reuse_submodule(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        a0 = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b0 = graph.call_function(torch.ops.aten.cos.default, args=(a0,))
        a1 = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b1 = graph.call_function(torch.ops.aten.cos.default, args=(a1,))
        out = graph.call_function(torch.ops.aten.add.Tensor, args=(b0, b1))
        graph.output((out,))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        _annotate_region((a0, b0), "chunk_0")
        _annotate_region((a1, b1), "chunk_1")
        symbolic_value = torch.empty(ShapeEnv().create_unbacked_symint(), device="meta")
        for node in (a0, b0, a1, b1):
            node.meta["val"] = symbolic_value

        apply_subgraph_region_annotations_pass(gm)

        invoke_nodes = _invoke_subgraph_nodes(gm)
        self.assertEqual(len(invoke_nodes), 2)
        self.assertEqual(invoke_nodes[0].args[1], invoke_nodes[1].args[1])
        self.assertEqual(len(_subgraph_modules(gm)), 1)

    def test_structurally_different_subgraph_regions_keep_submodules(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        a0 = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b0 = graph.call_function(torch.ops.aten.cos.default, args=(a0,))
        a1 = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b1 = graph.call_function(torch.ops.aten.neg.default, args=(a1,))
        out = graph.call_function(torch.ops.aten.add.Tensor, args=(b0, b1))
        graph.output((out,))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        _annotate_region((a0, b0), "chunk_0")
        _annotate_region((a1, b1), "chunk_1")

        apply_subgraph_region_annotations_pass(gm)

        invoke_nodes = _invoke_subgraph_nodes(gm)
        self.assertEqual(len(invoke_nodes), 2)
        self.assertNotEqual(invoke_nodes[0].args[1], invoke_nodes[1].args[1])
        self.assertEqual(len(_subgraph_modules(gm)), 2)


if __name__ == "__main__":
    run_tests()
