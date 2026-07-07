# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.fx as fx
from torch._decomp import get_decompositions
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer.min_cut_rematerialization import (
    min_cut_rematerialization_pass,
)
from torchtitan.experiments.graph_trainer.subgraph_regions import (
    apply_subgraph_region_annotations_pass,
    SUBGRAPH_REGION,
    SUBGRAPH_REGION_ROLE,
)


def _fake_prop(gm, *inputs):
    with FakeTensorMode() as fake_mode:
        fake_inputs = [
            torch.empty(shape, device="cuda", dtype=dtype)
            for shape, dtype in inputs
        ]
        FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(
            *fake_inputs
        )


def _recomputed_nodes(gm):
    return [node for node in gm.graph.nodes if node.name.endswith("_recomputed")]


def _log_softmax_decomposition_table():
    return get_decompositions([torch.ops.aten._log_softmax.default])


def _annotate_region(nodes, region):
    for node in nodes:
        node.meta.setdefault("custom", {})
        node.meta["custom"][SUBGRAPH_REGION] = region
        node.meta["custom"][SUBGRAPH_REGION_ROLE] = "txt_unemb_chunk"


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


class TestMinCutRematerialization(TestCase):
    def test_applies_to_whole_graph(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.cos.default, args=(a,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(b,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(b,))
        bwd.meta["autograd_backward"] = True
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        _fake_prop(gm, ((64, 64), torch.float32))

        min_cut_rematerialization_pass(gm)

        self.assertGreaterEqual(len(_recomputed_nodes(gm)), 1)

    def test_decomposed_backward_input_is_recomputed(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        grad = graph.placeholder("grad")
        log_probs = graph.call_function(
            torch.ops.aten._log_softmax.default, args=(x, -1, False)
        )
        loss = graph.call_function(torch.ops.aten.sum.default, args=(log_probs,))
        bwd = graph.call_function(
            torch.ops.aten._log_softmax_backward_data.default,
            args=(grad, log_probs, -1, torch.float32),
        )
        bwd.meta["autograd_backward"] = True
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        _fake_prop(gm, ((128, 1024), torch.float32), ((128, 1024), torch.float32))

        min_cut_rematerialization_pass(
            gm,
            decomposition_table=_log_softmax_decomposition_table(),
        )

        bwd = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten._log_softmax_backward_data.default
        )
        self.assertIsInstance(bwd.args[1], fx.Node)
        self.assertTrue(bwd.args[1].name.endswith("_recomputed"))
        self.assertFalse(
            any(
                node.target == torch.ops.aten._log_softmax.default
                for node in gm.graph.nodes
            )
        )

    def test_subgraph_annotation_can_apply_min_cut(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        a = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        b = graph.call_function(torch.ops.aten.cos.default, args=(a,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(b,))
        graph.output((bwd,))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        _fake_prop(gm, ((64, 64), torch.float32))

        for node in (a, b, bwd):
            node.meta.setdefault("custom", {})
            node.meta["custom"][SUBGRAPH_REGION] = "region"
            node.meta["custom"][SUBGRAPH_REGION_ROLE] = "fw_bw_grad_accum"
        bwd.meta["autograd_backward"] = True

        apply_subgraph_region_annotations_pass(
            gm,
            min_cut_rematerialization=True,
        )

        submods = [
            module
            for module in gm.modules()
            if isinstance(module, fx.GraphModule) and module is not gm
        ]
        self.assertEqual(len(submods), 1)
        self.assertGreaterEqual(len(_recomputed_nodes(submods[0])), 1)

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
