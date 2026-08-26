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
    def test_subgraph_contextmanager_example_outlines_region(self):
        def f(x):
            with subgraph("chunk_0", role="loss_chunk"):
                a = torch.ops.aten.sin.default(x)
                b = torch.ops.aten.cos.default(a)
            return torch.ops.aten.add.Tensor(b, x)

        x = torch.randn(4)
        with preserve_node_meta():
            gm = make_fx(f)(x)

        apply_subgraph_region_annotations_pass(gm)

        invoke_nodes = _invoke_subgraph_nodes(gm)
        self.assertEqual(len(invoke_nodes), 1)
        self.assertEqual(invoke_nodes[0].meta[SUBGRAPH_REGION], "chunk_0_loss_chunk")
        self.assertEqual(
            invoke_nodes[0].meta["custom"]["subgraph_region_id"], "chunk_0"
        )
        self.assertEqual(
            invoke_nodes[0].meta["custom"]["subgraph_region_role"], "loss_chunk"
        )
        self.assertEqual(len(_subgraph_modules(gm)), 1)
        torch.testing.assert_close(gm(x), f(x))

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
