# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import TestCase

from torchtitan.experiments.graph_trainer.coda_passes import (
    fuse_b6_bf16_weight_grad_cast_pass,
    get_coda_pattern_passes,
)


class TestB6WeightGradCastPass(TestCase):
    def _trace(self, fn, *inputs):
        return make_fx(fn)(*inputs)

    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def _body(self, gm, fused):
        body_ref = fused.args[1]
        self.assertEqual(body_ref.op, "get_attr")
        return getattr(gm, body_ref.target)

    def test_fuses_bf16_mm_to_fp32(self):
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, 12, dtype=torch.bfloat16)
        gm = self._trace(lambda a, b: torch.mm(a, b).float(), x, weight)
        expected = gm(x, weight)

        fuse_b6_bf16_weight_grad_cast_pass(gm)

        self.assertEqual(gm(x, weight), expected, exact_dtype=True)
        fused = self._flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 1)
        body_targets = [node.target for node in self._body(gm, fused[0]).graph.nodes]
        self.assertEqual(body_targets.count(torch.ops.aten.mm.default), 1)
        self.assertEqual(body_targets.count(torch.ops.aten._to_copy.default), 3)

    def test_does_not_fuse_unsupported_fp32_router_round_trip(self):
        x = torch.randn(8, 16)
        weight = torch.randn(16, 12)
        gm = self._trace(lambda a, b: torch.mm(a, b).bfloat16().float(), x, weight)
        fuse_b6_bf16_weight_grad_cast_pass(gm)
        self.assertEqual(self._flex_gemm_nodes(gm), [])

    def test_does_not_fuse_multi_use_mm(self):
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, 12, dtype=torch.bfloat16)

        def fn(a, b):
            mm = torch.mm(a, b)
            return mm.float(), mm + 1

        gm = self._trace(fn, x, weight)
        fuse_b6_bf16_weight_grad_cast_pass(gm)
        self.assertEqual(self._flex_gemm_nodes(gm), [])

    def test_preserves_metadata_and_tags_regional_inductor(self):
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        weight = torch.randn(16, 12, dtype=torch.bfloat16)
        gm = self._trace(lambda a, b: torch.mm(a, b).float(), x, weight)
        mm = next(
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default
        )
        cast = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten._to_copy.default
        )
        mm.meta["custom"] = {"module_fqn": "layers.0.moe", "EP": "compute"}
        cast.meta["custom"] = {"autograd_backward": True}

        fuse_b6_bf16_weight_grad_cast_pass(gm)

        fused = self._flex_gemm_nodes(gm)[0]
        self.assertEqual(fused.meta["custom"]["module_fqn"], "layers.0.moe")
        self.assertEqual(fused.meta["custom"]["EP"], "compute")
        self.assertTrue(fused.meta["custom"]["autograd_backward"])
        self.assertIn("compile_with_inductor", fused.meta["custom"])
        output = next(iter(fused.users))
        self.assertEqual(output.target, operator.getitem)
        self.assertIn("compile_with_inductor", output.meta["custom"])
        body_ref = fused.args[1]
        self.assertIn("compile_with_inductor", body_ref.meta["custom"])
        for node in self._body(gm, fused).graph.nodes:
            self.assertIn("compile_with_inductor", node.meta["custom"])


class TestCodaPatternRegistry(TestCase):
    def test_resolves_configured_order(self):
        passes = get_coda_pattern_passes(["b6_bf16_weight_grad_cast"])
        self.assertEqual(passes, [fuse_b6_bf16_weight_grad_cast_pass])

    def test_rejects_unknown_pattern(self):
        with self.assertRaisesRegex(ValueError, "Unknown.*not_a_pattern"):
            get_coda_pattern_passes(["not_a_pattern"])

    def test_rejects_duplicate_pattern(self):
        with self.assertRaisesRegex(ValueError, "Duplicate.*b6_bf16_weight_grad_cast"):
            get_coda_pattern_passes(
                ["b6_bf16_weight_grad_cast", "b6_bf16_weight_grad_cast"]
            )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
