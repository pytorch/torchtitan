# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.common_utils import (
    PARAMETER_GRADIENT_FQNS_META,
)
from torchtitan.experiments.graph_trainer.wgrad_accumulation import (
    fuse_wgrad_accumulation_pass,
    insert_graph_gradient_accumulation,
)


def _mm_graph(
    dtype: torch.dtype = torch.bfloat16,
    *,
    annotate_wgrad: bool = True,
) -> fx.GraphModule:
    graph = fx.Graph()
    lhs = graph.placeholder("lhs")
    lhs.meta["val"] = torch.empty(3, 4, dtype=dtype)
    rhs = graph.placeholder("rhs")
    rhs.meta["val"] = torch.empty(4, 2, dtype=dtype)
    wgrad = graph.call_function(torch.ops.aten.mm.default, args=(lhs, rhs))
    wgrad.meta["val"] = torch.empty(3, 2, dtype=dtype)
    if annotate_wgrad:
        wgrad.meta["custom"] = {PARAMETER_GRADIENT_FQNS_META: ("weight",)}
    graph.output((wgrad,))
    return fx.GraphModule(torch.nn.Module(), graph)


class TestWgradAccumulation(unittest.TestCase):
    def test_graph_accumulator_reuses_one_buffer(self) -> None:
        gm = _mm_graph(torch.float64)
        (accumulator,) = insert_graph_gradient_accumulation(
            gm,
            num_param_grads=1,
            device=torch.device("cpu"),
        )
        assert accumulator is not None
        accumulator.zero_()
        lhs = torch.randn(3, 4, dtype=torch.float64)
        rhs = torch.randn(4, 2, dtype=torch.float64)

        first = gm(lhs, rhs, accumulator)[0]
        second = gm(lhs, rhs, accumulator)[0]

        self.assertIs(first, accumulator)
        self.assertIs(second, accumulator)
        torch.testing.assert_close(accumulator, 2 * (lhs @ rhs))

    def test_duplicate_gradient_outputs_share_one_accumulator(self) -> None:
        gm = _mm_graph(torch.float64)
        output = gm.graph.find_nodes(op="output")[0]
        (wgrad,) = output.args[0]
        output.args = ((wgrad, wgrad),)
        gm.recompile()

        first, second = insert_graph_gradient_accumulation(
            gm,
            num_param_grads=2,
            device=torch.device("cpu"),
        )

        self.assertIs(first, second)
        self.assertEqual(len(gm.graph.find_nodes(op="placeholder")), 3)

    def test_bf16_mm_accumulation_fuses_to_addmm(self) -> None:
        gm = _mm_graph()
        (accumulator,) = insert_graph_gradient_accumulation(
            gm,
            num_param_grads=1,
            device=torch.device("cpu"),
        )

        fuse_wgrad_accumulation_pass(gm)

        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(torch.ops.aten.addmm_.default, targets)
        self.assertNotIn(torch.ops.aten.mm.default, targets)
        self.assertNotIn(torch.ops.aten.add_.Tensor, targets)

        assert accumulator is not None
        accumulator.zero_()
        lhs = torch.randn(3, 4, dtype=torch.bfloat16)
        rhs = torch.randn(4, 2, dtype=torch.bfloat16)
        expected = torch.addmm(torch.zeros_like(accumulator), lhs, rhs)
        (actual,) = gm(lhs, rhs, accumulator)
        self.assertIs(actual, accumulator)
        torch.testing.assert_close(actual, expected)

    def test_non_bf16_mm_keeps_explicit_accumulation(self) -> None:
        gm = _mm_graph(torch.float32)
        insert_graph_gradient_accumulation(
            gm,
            num_param_grads=1,
            device=torch.device("cpu"),
        )

        fuse_wgrad_accumulation_pass(gm)

        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(torch.ops.aten.mm.default, targets)
        self.assertIn(torch.ops.aten.add_.Tensor, targets)
        self.assertNotIn(torch.ops.aten.addmm_.default, targets)

    def test_unannotated_mm_keeps_explicit_accumulation(self) -> None:
        gm = _mm_graph(annotate_wgrad=False)
        insert_graph_gradient_accumulation(
            gm,
            num_param_grads=1,
            device=torch.device("cpu"),
        )

        fuse_wgrad_accumulation_pass(gm)

        targets = {node.target for node in gm.graph.nodes}
        self.assertIn(torch.ops.aten.mm.default, targets)
        self.assertIn(torch.ops.aten.add_.Tensor, targets)
        self.assertNotIn(torch.ops.aten.addmm_.default, targets)


if __name__ == "__main__":
    unittest.main()
