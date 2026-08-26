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
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.decompositions import (
    apply_decompositions_pass,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    tag_with_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.subgraph_regions import (
    apply_subgraph_region_annotations_pass,
    SUBGRAPH_REGION,
    SUBGRAPH_REGION_ROLE,
)


def _fake_prop(gm, *inputs):
    with FakeTensorMode() as fake_mode:
        fake_inputs = [
            torch.empty(shape, device="cuda", dtype=dtype) for shape, dtype in inputs
        ]
        FakeTensorProp(gm, mode=fake_mode).propagate_dont_convert_inputs(*fake_inputs)


def _recomputed_nodes(gm):
    return [node for node in gm.graph.nodes if node.name.endswith("_recomputed")]


def _log_softmax_decomposition_table():
    return get_decompositions([torch.ops.aten._log_softmax.default])


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

        tag_with_memory_policy_pass(gm, memory_policy="min_cut")
        self.assertTrue(
            any(
                node.meta.get("recompute") == CheckpointPolicy.MUST_RECOMPUTE
                for node in gm.graph.nodes
            )
        )
        self.assertEqual(len(_recomputed_nodes(gm)), 0)
        selective_activation_remat_pass(gm)

        self.assertGreaterEqual(len(_recomputed_nodes(gm)), 1)

    def test_decomposition_is_a_standalone_pass_before_min_cut(self):
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

        apply_decompositions_pass(
            gm,
            decomposition_table=_log_softmax_decomposition_table(),
        )
        self.assertFalse(
            any(
                node.target == torch.ops.aten._log_softmax.default
                for node in gm.graph.nodes
            )
        )
        self.assertEqual(len(_recomputed_nodes(gm)), 0)

        tag_with_memory_policy_pass(gm, memory_policy="min_cut")
        selective_activation_remat_pass(gm)

        bwd = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten._log_softmax_backward_data.default
        )
        self.assertIsInstance(bwd.args[1], fx.Node)
        self.assertTrue(bwd.args[1].name.endswith("_recomputed"))

    def test_min_cut_policy_respects_existing_checkpoint_policy(self):
        graph = fx.Graph()
        x = graph.placeholder("x")
        saved = graph.call_function(torch.ops.aten.sin.default, args=(x,))
        recompute = graph.call_function(torch.ops.aten.cos.default, args=(saved,))
        loss = graph.call_function(torch.ops.aten.sum.default, args=(recompute,))
        bwd = graph.call_function(torch.ops.aten.neg.default, args=(recompute,))
        bwd.meta["autograd_backward"] = True
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        _fake_prop(gm, ((64, 64), torch.float32))
        saved.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        recompute.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE

        tag_with_memory_policy_pass(gm, memory_policy="min_cut")

        self.assertEqual(saved.meta["recompute"], CheckpointPolicy.MUST_SAVE)
        self.assertIn(
            recompute.meta["recompute"],
            (CheckpointPolicy.PREFER_RECOMPUTE, CheckpointPolicy.MUST_RECOMPUTE),
        )
        selective_activation_remat_pass(gm)
        recomputed_targets = {node.target for node in _recomputed_nodes(gm)}
        self.assertIn(torch.ops.aten.cos.default, recomputed_targets)
        self.assertNotIn(torch.ops.aten.sin.default, recomputed_targets)

    def test_explicit_subgraph_decomposition_and_min_cut_policy(self):
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
        graph.output((loss, bwd))
        gm = fx.GraphModule(torch.nn.Module(), graph)
        _fake_prop(gm, ((128, 1024), torch.float32), ((128, 1024), torch.float32))

        for node in (log_probs, loss, bwd):
            node.meta.setdefault("custom", {})
            node.meta["custom"][SUBGRAPH_REGION] = "region"
            node.meta["custom"][SUBGRAPH_REGION_ROLE] = "fw_bw_grad_accum"
        bwd.meta["autograd_backward"] = True

        apply_subgraph_region_annotations_pass(gm)
        apply_decompositions_pass(
            gm,
            decomposition_table=_log_softmax_decomposition_table(),
            recurse=True,
            apply_to_root=False,
        )
        tag_with_memory_policy_pass(
            gm,
            memory_policy="min_cut",
            recurse=True,
            apply_to_root=False,
        )
        selective_activation_remat_pass(
            gm,
            recurse=True,
            apply_to_root=False,
        )

        submods = [
            module
            for module in gm.modules()
            if isinstance(module, fx.GraphModule) and module is not gm
        ]
        self.assertEqual(len(submods), 1)
        submod = submods[0]
        self.assertFalse(
            any(
                node.target == torch.ops.aten._log_softmax.default
                for node in submod.graph.nodes
            )
        )
        bwd = next(
            node
            for node in submod.graph.nodes
            if node.target == torch.ops.aten._log_softmax_backward_data.default
        )
        self.assertIsInstance(bwd.args[1], fx.Node)
        self.assertTrue(bwd.args[1].name.endswith("_recomputed"))


if __name__ == "__main__":
    run_tests()
