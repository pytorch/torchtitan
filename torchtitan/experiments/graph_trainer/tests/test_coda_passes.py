# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.testing._internal.common_utils import TestCase

from torchtitan.experiments.graph_trainer.coda_passes import (
    fuse_b2_dense_swiglu_backward_pass,
    fuse_b6_bf16_weight_grad_cast_pass,
    fuse_f2_q_rmsnorm_pass,
    fuse_f4_dense_swiglu_pass,
    fuse_f6_router_sigmoid_bias_pass,
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


class TestF6RouterSigmoidBiasPass(TestCase):
    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def test_fuses_sigmoid_and_bias_while_preserving_raw_scores(self):
        x = torch.randn(8, 16)
        weight = torch.randn(16, 5)
        bias = torch.randn(5)

        def fn(a, b, expert_bias):
            scores = torch.sigmoid(torch.mm(a, b).reshape(2, 4, 5))
            return scores, scores + expert_bias, scores * 2

        gm = make_fx(fn)(x, weight, bias)
        expected = gm(x, weight, bias)
        fuse_f6_router_sigmoid_bias_pass(gm)

        self.assertEqual(gm(x, weight, bias), expected, exact_dtype=True)
        fused = self._flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 1)
        self.assertEqual(len(fused[0].args[2]), 3)
        body_ref = fused[0].args[1]
        body = getattr(gm, body_ref.target)
        targets = [node.target for node in body.graph.nodes]
        self.assertEqual(targets.count(torch.ops.aten.mm.default), 1)
        self.assertEqual(targets.count(torch.ops.aten.sigmoid.default), 1)
        self.assertEqual(targets.count(torch.ops.aten.add.Tensor), 1)

    def test_fuses_recomputed_sigmoid_without_bias(self):
        x = torch.randn(8, 16)
        weight = torch.randn(16, 5)

        def fn(a, b):
            scores = torch.sigmoid(torch.mm(a, b).reshape(2, 4, 5))
            return scores, scores * 2

        gm = make_fx(fn)(x, weight)
        expected = gm(x, weight)
        fuse_f6_router_sigmoid_bias_pass(gm)

        self.assertEqual(gm(x, weight), expected, exact_dtype=True)
        fused = self._flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 1)
        self.assertEqual(len(fused[0].args[2]), 2)

    def test_does_not_fuse_multi_use_mm(self):
        x = torch.randn(8, 16)
        weight = torch.randn(16, 5)

        def fn(a, b):
            mm = torch.mm(a, b)
            return torch.sigmoid(mm.reshape(2, 4, 5)), mm + 1

        gm = make_fx(fn)(x, weight)
        fuse_f6_router_sigmoid_bias_pass(gm)
        self.assertEqual(self._flex_gemm_nodes(gm), [])


class TestF4DenseSwiGLUPass(TestCase):
    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def test_fuses_two_gemms_and_preserves_saved_activations(self):
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        w1 = torch.randn(16, 12, dtype=torch.bfloat16)
        w3 = torch.randn(16, 12, dtype=torch.bfloat16)

        def fn(a, first_weight, gate_weight):
            first = torch.mm(a, first_weight).reshape(2, 4, 12)
            activated = torch.nn.functional.silu(first)
            gate = torch.mm(a, gate_weight).reshape(2, 4, 12)
            product = activated * gate
            return first, activated, gate, product, product + 1

        gm = make_fx(fn)(x, w1, w3)
        expected = gm(x, w1, w3)
        fuse_f4_dense_swiglu_pass(gm)

        self.assertEqual(gm(x, w1, w3), expected, exact_dtype=True)
        fused = self._flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 2)
        self.assertEqual(len(fused[0].args[2]), 2)
        self.assertEqual(len(fused[1].args[2]), 3)
        self.assertEqual(len(fused[0].meta["val"]), 2)
        first_body = getattr(gm, fused[0].args[1].target)
        second_body = getattr(gm, fused[1].args[1].target)
        first_targets = [node.target for node in first_body.graph.nodes]
        second_targets = [node.target for node in second_body.graph.nodes]
        self.assertEqual(first_targets.count(torch.ops.aten.silu.default), 1)
        self.assertEqual(first_targets.count(torch.ops.aten._to_copy.default), 2)
        self.assertEqual(second_targets.count(torch.ops.aten.mul.Tensor), 1)
        self.assertEqual(second_targets.count(torch.ops.aten._to_copy.default), 2)

    def test_does_not_fuse_fp32_swiglu(self):
        x = torch.randn(8, 16)
        w1 = torch.randn(16, 12)
        w3 = torch.randn(16, 12)

        def fn(a, first_weight, gate_weight):
            activated = torch.nn.functional.silu(
                torch.mm(a, first_weight).reshape(2, 4, 12)
            )
            gate = torch.mm(a, gate_weight).reshape(2, 4, 12)
            return activated * gate

        gm = make_fx(fn)(x, w1, w3)
        fuse_f4_dense_swiglu_pass(gm)
        self.assertEqual(self._flex_gemm_nodes(gm), [])

    def test_does_not_fuse_multi_use_first_gemm(self):
        x = torch.randn(8, 16, dtype=torch.bfloat16)
        w1 = torch.randn(16, 12, dtype=torch.bfloat16)
        w3 = torch.randn(16, 12, dtype=torch.bfloat16)

        def fn(a, first_weight, gate_weight):
            first_mm = torch.mm(a, first_weight)
            activated = torch.nn.functional.silu(first_mm.reshape(2, 4, 12))
            gate = torch.mm(a, gate_weight).reshape(2, 4, 12)
            return activated * gate, first_mm + 1

        gm = make_fx(fn)(x, w1, w3)
        fuse_f4_dense_swiglu_pass(gm)
        self.assertEqual(self._flex_gemm_nodes(gm), [])


class TestB2DenseSwiGLUBackwardPass(TestCase):
    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def test_fuses_branch_derivatives_and_input_gradient_add(self):
        grad_out = torch.randn(8, 16, dtype=torch.bfloat16)
        w2 = torch.randn(16, 12, dtype=torch.bfloat16)
        saved_silu = torch.randn(2, 4, 12, dtype=torch.bfloat16)
        saved_gate = torch.randn(2, 4, 12, dtype=torch.bfloat16)
        saved_preactivation = torch.randn(2, 4, 12, dtype=torch.bfloat16)
        w3 = torch.randn(12, 7, dtype=torch.bfloat16)
        w1 = torch.randn(12, 7, dtype=torch.bfloat16)

        def fn(
            grad,
            output_weight,
            silu,
            gate,
            preactivation,
            gate_weight,
            first_weight,
        ):
            branch_grad = torch.mm(grad, output_weight).reshape(2, 4, 12)
            gate_grad = branch_grad * silu
            silu_grad = torch.ops.aten.silu_backward.default(
                branch_grad * gate, preactivation
            )
            w3_input_grad = torch.mm(gate_grad.reshape(8, 12), gate_weight)
            w1_input_grad = torch.mm(silu_grad.reshape(8, 12), first_weight)
            input_grad = w3_input_grad.reshape(2, 4, 7) + w1_input_grad.reshape(2, 4, 7)
            return gate_grad, silu_grad, input_grad

        inputs = (
            grad_out,
            w2,
            saved_silu,
            saved_gate,
            saved_preactivation,
            w3,
            w1,
        )
        gm = make_fx(fn)(*inputs)
        expected = gm(*inputs)
        mm_nodes = [
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default
        ]
        self.assertEqual(len(mm_nodes), 3)
        mm_nodes[1].meta["custom"] = {
            "module_fqn": "layers.0.feed_forward.w3",
            "autograd_backward": True,
        }
        mm_nodes[2].meta["custom"] = {
            "module_fqn": "layers.0.feed_forward.w1",
            "autograd_backward": True,
        }

        fuse_b2_dense_swiglu_backward_pass(gm)

        self.assertEqual(gm(*inputs), expected, exact_dtype=True)
        fused = self._flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 2)
        self.assertEqual(len(fused[0].args[2]), 5)
        self.assertEqual(len(fused[1].args[2]), 3)

        branch_body = getattr(gm, fused[0].args[1].target)
        branch_targets = [node.target for node in branch_body.graph.nodes]
        self.assertEqual(branch_targets.count(torch.ops.aten.mul.Tensor), 2)
        self.assertEqual(branch_targets.count(torch.ops.aten.silu_backward.default), 1)
        self.assertEqual(branch_targets.count(torch.ops.aten._to_copy.default), 2)

        input_add_body = getattr(gm, fused[1].args[1].target)
        input_add_targets = [node.target for node in input_add_body.graph.nodes]
        self.assertEqual(input_add_targets.count(torch.ops.aten.add.Tensor), 1)
        self.assertEqual(input_add_targets.count(torch.ops.aten._to_copy.default), 2)

    def test_does_not_fuse_unrelated_input_gradient_add(self):
        x = torch.randn(8, 12, dtype=torch.bfloat16)
        first_weight = torch.randn(12, 7, dtype=torch.bfloat16)
        second_weight = torch.randn(12, 7, dtype=torch.bfloat16)

        def fn(a, first, second):
            lhs = torch.mm(a, first).reshape(2, 4, 7)
            rhs = torch.mm(a, second).reshape(2, 4, 7)
            return lhs + rhs

        gm = make_fx(fn)(x, first_weight, second_weight)
        for index, node in enumerate(
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default
        ):
            node.meta["custom"] = {"module_fqn": f"layers.0.linear{index}"}

        fuse_b2_dense_swiglu_backward_pass(gm)

        self.assertEqual(self._flex_gemm_nodes(gm), [])


class TestF2QRMSNormPass(TestCase):
    def _trace_q_chain(self, *, return_rstd=False):
        m, k, n, p = 8, 16, 512, 64

        def fn(a, first_weight, gamma, second_weight):
            first = torch.ops.aten.mm.default(a, first_weight)
            first = torch.ops.aten.reshape.default(first, [2, 4, n])
            normalized, rstd = torch.ops.aten._fused_rms_norm.default(
                first, [n], gamma, 1e-5
            )
            normalized = torch.ops.aten.reshape.default(normalized, [m, n])
            output = torch.ops.aten.mm.default(normalized, second_weight)
            return (output, rstd, first, normalized) if return_rstd else output

        inputs = (
            torch.randn(m, k, dtype=torch.bfloat16) * 0.02,
            torch.randn(k, n, dtype=torch.bfloat16) * 0.02,
            torch.ones(n, dtype=torch.bfloat16),
            torch.randn(n, p, dtype=torch.bfloat16) * 0.02,
        )
        gm = torch.fx.symbolic_trace(fn)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        FakeTensorProp(gm).propagate(*inputs)
        mm_nodes = [
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default
        ]
        norm = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten._fused_rms_norm.default
        )
        mm_nodes[0].meta["custom"] = {"module_fqn": "layers.0.attention.wq_a"}
        norm.meta["custom"] = {"module_fqn": "layers.0.attention.q_norm"}
        mm_nodes[1].meta["custom"] = {"module_fqn": "layers.0.attention.wq_b"}
        return gm, inputs

    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def test_reparameterizes_original_forward_q_norm(self):
        gm, inputs = self._trace_q_chain()
        a, first_weight, gamma, second_weight = inputs
        first = torch.mm(a, first_weight)
        weighted = (first.float() * gamma).bfloat16()
        partial_mean_square = first.float().reshape(8, -1, 512).square().mean(-1)
        rstd = (partial_mean_square.mean(-1, keepdim=True) + 1e-5).rsqrt()
        expected = (torch.mm(weighted, second_weight).float() * rstd).bfloat16()

        fuse_f2_q_rmsnorm_pass(gm)

        actual = gm(*inputs)
        self.assertEqual(actual.dtype, torch.bfloat16)
        torch.testing.assert_close(actual, expected, atol=1e-3, rtol=2e-2)
        fused = self._flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 2)
        self.assertEqual(len(fused[0].args[2]), 3)
        self.assertEqual(len(fused[1].args[2]), 3)
        self.assertFalse(
            any(
                node.target == torch.ops.aten._fused_rms_norm.default
                for node in gm.graph.nodes
            )
        )
        first_body = getattr(gm, fused[0].args[1].target)
        first_targets = [node.target for node in first_body.graph.nodes]
        self.assertEqual(first_targets.count(torch.ops.aten.mean.dim), 1)
        self.assertEqual(first_targets.count(torch.ops.aten.pow.Tensor_Scalar), 1)

    def test_rewrites_recomputed_norm_and_preserves_saved_values(self):
        gm, inputs = self._trace_q_chain(return_rstd=True)
        a, first_weight, gamma, second_weight = inputs
        first = torch.mm(a, first_weight)
        partial_mean_square = first.float().reshape(8, -1, 512).square().mean(-1)
        rstd_2d = (partial_mean_square.mean(-1, keepdim=True) + 1e-5).rsqrt()
        weighted = (first.float() * gamma).bfloat16()
        output = (torch.mm(weighted, second_weight).float() * rstd_2d).bfloat16()
        normalized = (first.float() * rstd_2d * gamma).bfloat16()

        fuse_f2_q_rmsnorm_pass(gm)

        actual_output, actual_rstd, actual_first, actual_normalized = gm(*inputs)
        torch.testing.assert_close(actual_output, output, atol=1e-3, rtol=2e-2)
        self.assertEqual(actual_rstd, rstd_2d.reshape(2, 4, 1))
        self.assertEqual(actual_first, first.reshape(2, 4, 512))
        self.assertEqual(actual_normalized, normalized)
        self.assertEqual(len(self._flex_gemm_nodes(gm)), 2)
        self.assertFalse(
            any(
                node.target == torch.ops.aten._fused_rms_norm.default
                for node in gm.graph.nodes
            )
        )


class TestCodaPatternRegistry(TestCase):
    def test_resolves_configured_order(self):
        passes = get_coda_pattern_passes(
            [
                "f6_router_sigmoid_bias",
                "f4_dense_swiglu",
                "b2_dense_swiglu_backward",
                "f2_q_rmsnorm",
                "b6_bf16_weight_grad_cast",
            ]
        )
        self.assertEqual(
            passes,
            [
                fuse_f6_router_sigmoid_bias_pass,
                fuse_f4_dense_swiglu_pass,
                fuse_b2_dense_swiglu_backward_pass,
                fuse_f2_q_rmsnorm_pass,
                fuse_b6_bf16_weight_grad_cast_pass,
            ],
        )

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
