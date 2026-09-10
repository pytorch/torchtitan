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
    fuse_b1_lm_head_input_grad_cast_pass,
    fuse_b2_dense_swiglu_backward_pass,
    fuse_b4_router_input_grad_add_pass,
    fuse_b5_mla_rmsnorm_backward_pass,
    fuse_b6_bf16_weight_grad_cast_pass,
    fuse_b7_attention_grad_merge_pass,
    fuse_f2_kv_rmsnorm_pass,
    fuse_f2_q_rmsnorm_pass,
    fuse_f3_residual_rmsnorm_pass,
    fuse_f4_dense_swiglu_pass,
    fuse_f6_router_sigmoid_bias_pass,
    get_coda_pattern_passes,
)


class TestB1LMHeadInputGradCastPass(TestCase):
    def _trace(self, *, module_fqn="lm_head", backward=True):
        m, k, n = 8, 16, 32

        def fn(grad, weight, destination):
            input_grad = torch.ops.aten.mm.default(grad, weight)
            input_grad = torch.ops.aten.reshape.default(input_grad, [2, 4, n])
            input_grad = torch.ops.aten.alias.default(input_grad)
            input_grad = torch.ops.aten._to_copy.default(
                input_grad, dtype=torch.float32
            )
            target = torch.ops.aten.slice.Tensor(destination, 1, 0, 4)
            torch.ops.aten.copy_.default(target, input_grad)
            return destination

        inputs = (
            torch.randn(m, k, dtype=torch.bfloat16) * 0.02,
            torch.randn(k, n, dtype=torch.bfloat16) * 0.02,
            torch.zeros(2, 8, n, dtype=torch.float32),
        )
        gm = make_fx(fn)(*inputs)
        mm = next(
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default
        )
        mm.meta["custom"] = {"module_fqn": module_fqn}
        if backward:
            mm.meta["autograd_backward"] = True
        return gm, inputs

    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def test_fuses_lm_head_input_gradient_cast(self):
        gm, inputs = self._trace()
        expected = gm(*(value.clone() for value in inputs))

        fuse_b1_lm_head_input_grad_cast_pass(gm)

        actual = gm(*(value.clone() for value in inputs))
        self.assertEqual(actual, expected, exact_dtype=True)
        fused = self._flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 1)
        fused_output = next(iter(fused[0].users))
        self.assertEqual(fused_output.meta["val"].shape, (8, 32))
        self.assertEqual(fused_output.meta["val"].dtype, torch.float32)
        body = getattr(gm, fused[0].args[1].target)
        body_targets = [node.target for node in body.graph.nodes]
        self.assertEqual(body_targets.count(torch.ops.aten._to_copy.default), 3)
        root_targets = [node.target for node in gm.graph.nodes]
        self.assertEqual(root_targets.count(torch.ops.aten.copy_.default), 1)

    def test_does_not_fuse_unrelated_module(self):
        gm, _ = self._trace(module_fqn="layers.3.attention.wq_a")

        fuse_b1_lm_head_input_grad_cast_pass(gm)

        self.assertEqual(self._flex_gemm_nodes(gm), [])

    def test_does_not_fuse_forward_graph(self):
        gm, _ = self._trace(backward=False)

        fuse_b1_lm_head_input_grad_cast_pass(gm)

        self.assertEqual(self._flex_gemm_nodes(gm), [])


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


class TestF3ResidualRMSNormPass(TestCase):
    def _trace_chain(self, *, cross_layer=False, return_saved=False):
        m, k, n, p, q = 8, 16, 512, 64, 96

        def fn(a, first_weight, residual, gamma, second_weight, third_weight):
            first = torch.ops.aten.mm.default(a, first_weight)
            first = torch.ops.aten.reshape.default(first, [2, 4, n])
            total = torch.ops.aten.add.Tensor(residual, first)
            gamma_for_norm = torch.ops.aten.clone.default(gamma)
            normalized, rstd = torch.ops.aten._fused_rms_norm.default(
                total, [n], gamma_for_norm, 1e-5
            )
            second_input = torch.ops.aten.reshape.default(normalized, [m, n])
            second = torch.ops.aten.mm.default(second_input, second_weight)
            third_input = torch.ops.aten.reshape.default(normalized, [m, n])
            third = torch.ops.aten.mm.default(third_input, third_weight)
            if return_saved:
                return second, third, total, rstd, second_input, third_input
            return second, third, total

        inputs = (
            torch.randn(m, k, dtype=torch.bfloat16) * 0.02,
            torch.randn(k, n, dtype=torch.bfloat16) * 0.02,
            torch.randn(2, 4, n, dtype=torch.bfloat16) * 0.02,
            torch.ones(n, dtype=torch.bfloat16),
            torch.randn(n, p, dtype=torch.bfloat16) * 0.02,
            torch.randn(n, q, dtype=torch.bfloat16) * 0.02,
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
        if cross_layer:
            mm_nodes[0].meta["custom"] = {"module_fqn": "layers.2.feed_forward.w2"}
            norm.meta["custom"] = {"module_fqn": "layers.3.attention_norm"}
            mm_nodes[1].meta["custom"] = {"module_fqn": "layers.3.attention.wq_a"}
            mm_nodes[2].meta["custom"] = {"module_fqn": "layers.3.attention.wkv_a"}
        else:
            mm_nodes[0].meta["custom"] = {"module_fqn": "layers.2.attention.wo"}
            norm.meta["custom"] = {"module_fqn": "layers.2.ffn_norm"}
            mm_nodes[1].meta["custom"] = {"module_fqn": "layers.2.feed_forward.w1"}
            mm_nodes[2].meta["custom"] = {"module_fqn": "layers.2.feed_forward.w3"}
        return gm, inputs

    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def _plain_mm_nodes(self, gm):
        return [
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default
        ]

    def _assert_outputs_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                atol=1e-3,
                rtol=2e-2,
            )

    def _trace_swiglu_chain(self, *, return_saved=False):
        m, k, n, p = 8, 16, 512, 64

        def fn(a, first_weight, residual, gamma, w1_weight, w3_weight):
            first = torch.ops.aten.mm.default(a, first_weight)
            first = torch.ops.aten.reshape.default(first, [2, 4, n])
            total = torch.ops.aten.add.Tensor(residual, first)
            normalized, rstd = torch.ops.aten._fused_rms_norm.default(
                total, [n], gamma, 1e-5
            )
            w1_input = torch.ops.aten.reshape.default(normalized, [m, n])
            w1 = torch.ops.aten.mm.default(w1_input, w1_weight)
            w1 = torch.ops.aten.reshape.default(w1, [2, 4, p])
            activated = torch.ops.aten.silu.default(w1)
            w3_input = torch.ops.aten.reshape.default(normalized, [m, n])
            w3 = torch.ops.aten.mm.default(w3_input, w3_weight)
            w3 = torch.ops.aten.reshape.default(w3, [2, 4, p])
            product = torch.ops.aten.mul.Tensor(activated, w3)
            return (product, total, rstd, w1, w3) if return_saved else (product, total)

        inputs = (
            torch.randn(m, k, dtype=torch.bfloat16) * 0.02,
            torch.randn(k, n, dtype=torch.bfloat16) * 0.02,
            torch.randn(2, 4, n, dtype=torch.bfloat16) * 0.02,
            torch.ones(n, dtype=torch.bfloat16),
            torch.randn(n, p, dtype=torch.bfloat16) * 0.02,
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
        mm_nodes[0].meta["custom"] = {"module_fqn": "layers.2.attention.wo"}
        norm.meta["custom"] = {"module_fqn": "layers.2.ffn_norm"}
        mm_nodes[1].meta["custom"] = {"module_fqn": "layers.2.feed_forward.w1"}
        mm_nodes[2].meta["custom"] = {"module_fqn": "layers.2.feed_forward.w3"}
        return gm, inputs

    def test_fuses_attention_output_residual_into_ffn_norm(self):
        gm, inputs = self._trace_chain()
        expected = gm(*inputs)

        fuse_f3_residual_rmsnorm_pass(gm)

        actual = gm(*inputs)
        self._assert_outputs_close(actual, expected)
        self.assertEqual(actual[2], expected[2])
        self.assertEqual(len(self._flex_gemm_nodes(gm)), 1)
        self.assertEqual(len(self._plain_mm_nodes(gm)), 2)
        self.assertFalse(
            any(
                node.target == torch.ops.aten._fused_rms_norm.default
                for node in gm.graph.nodes
            )
        )

    def test_composes_dense_swiglu_epilogues(self):
        gm, inputs = self._trace_swiglu_chain()
        expected = gm(*inputs)

        fuse_f3_residual_rmsnorm_pass(gm)
        fuse_f4_dense_swiglu_pass(gm)

        actual = gm(*inputs)
        self._assert_outputs_close(actual, expected)
        self.assertEqual(actual[1], expected[1])
        self.assertEqual(len(self._flex_gemm_nodes(gm)), 3)
        body_targets = [
            [node.target for node in getattr(gm, fused.args[1].target).graph.nodes]
            for fused in self._flex_gemm_nodes(gm)
        ]
        self.assertTrue(
            any(
                targets.count(torch.ops.aten.silu.default) == 1
                for targets in body_targets
            )
        )
        self.assertTrue(
            any(
                targets.count(torch.ops.aten.mul.Tensor) == 1
                for targets in body_targets
            )
        )

    def test_composes_recomputed_dense_swiglu_and_preserves_saved_values(self):
        gm, inputs = self._trace_swiglu_chain(return_saved=True)
        expected = gm(*inputs)

        fuse_f3_residual_rmsnorm_pass(gm)

        actual = gm(*inputs)
        self._assert_outputs_close(actual, expected)
        self.assertEqual(actual[1], expected[1])
        self.assertEqual(len(self._flex_gemm_nodes(gm)), 1)
        self.assertEqual(len(self._plain_mm_nodes(gm)), 2)

    def test_fuses_feed_forward_output_into_next_attention_norm(self):
        gm, inputs = self._trace_chain(cross_layer=True, return_saved=True)
        expected = gm(*inputs)

        fuse_f3_residual_rmsnorm_pass(gm)

        actual = gm(*inputs)
        self._assert_outputs_close(actual, expected)
        self.assertEqual(actual[2], expected[2])
        self.assertEqual(len(self._flex_gemm_nodes(gm)), 1)
        self.assertEqual(len(self._plain_mm_nodes(gm)), 2)

    def test_fuses_shared_expert_output_into_next_attention_norm(self):
        m, k, n, p, q = 8, 16, 512, 64, 96

        def fn(
            a,
            first_weight,
            routed_output,
            residual,
            gamma,
            second_weight,
            third_weight,
        ):
            shared_output = torch.ops.aten.mm.default(a, first_weight)
            shared_output = torch.ops.aten.reshape.default(shared_output, [2, 4, n])
            moe_output = torch.ops.aten.add.Tensor(routed_output, shared_output)
            total = torch.ops.aten.add.Tensor(residual, moe_output)
            gamma_for_norm = torch.ops.aten.clone.default(gamma)
            normalized, _ = torch.ops.aten._fused_rms_norm.default(
                total, [n], gamma_for_norm, 1e-5
            )
            second_input = torch.ops.aten.reshape.default(normalized, [m, n])
            second = torch.ops.aten.mm.default(second_input, second_weight)
            third_input = torch.ops.aten.reshape.default(normalized, [m, n])
            third = torch.ops.aten.mm.default(third_input, third_weight)
            return second, third, total

        inputs = (
            torch.randn(m, k, dtype=torch.bfloat16) * 0.02,
            torch.randn(k, n, dtype=torch.bfloat16) * 0.02,
            torch.randn(2, 4, n, dtype=torch.bfloat16) * 0.02,
            torch.randn(2, 4, n, dtype=torch.bfloat16) * 0.02,
            torch.ones(n, dtype=torch.bfloat16),
            torch.randn(n, p, dtype=torch.bfloat16) * 0.02,
            torch.randn(n, q, dtype=torch.bfloat16) * 0.02,
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
        mm_nodes[0].meta["custom"] = {"module_fqn": "layers.3.moe.shared_experts.w2"}
        norm.meta["custom"] = {"module_fqn": "layers.4.attention_norm"}
        mm_nodes[1].meta["custom"] = {"module_fqn": "layers.4.attention.wq_a"}
        mm_nodes[2].meta["custom"] = {"module_fqn": "layers.4.attention.wkv_a"}
        expected = gm(*inputs)

        fuse_f3_residual_rmsnorm_pass(gm)

        actual = gm(*inputs)
        self._assert_outputs_close(actual, expected)
        self.assertEqual(actual[2], expected[2])
        self.assertEqual(len(self._flex_gemm_nodes(gm)), 1)
        self.assertEqual(len(self._plain_mm_nodes(gm)), 2)

    def test_does_not_require_downstream_projection_roles(self):
        gm, inputs = self._trace_chain()
        mm_nodes = [
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default
        ]
        mm_nodes[1].meta["custom"] = {"module_fqn": "unrelated.first"}
        mm_nodes[2].meta["custom"] = {"module_fqn": "layers.2.feed_forward.w2"}
        expected = gm(*inputs)

        fuse_f3_residual_rmsnorm_pass(gm)

        self._assert_outputs_close(gm(*inputs), expected)
        self.assertEqual(len(self._flex_gemm_nodes(gm)), 1)
        self.assertEqual(len(self._plain_mm_nodes(gm)), 2)

    def test_does_not_fuse_mismatched_boundary_roles(self):
        gm, _ = self._trace_chain()
        norm = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten._fused_rms_norm.default
        )
        norm.meta["custom"] = {"module_fqn": "layers.3.attention_norm"}

        fuse_f3_residual_rmsnorm_pass(gm)

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


class TestF2KVRMSNormPass(TestCase):
    def _trace_kv_chain(self, *, return_saved=False):
        m, k, full_width, active_width, output_width = 8, 16, 576, 512, 96

        def fn(a, first_weight, gamma, second_weight):
            first = torch.ops.aten.mm.default(a, first_weight)
            first = torch.ops.aten.reshape.default(first, [2, 4, full_width])
            active, tail = torch.ops.aten.split_with_sizes.default(
                first, [active_width, full_width - active_width], -1
            )
            normalized, rstd = torch.ops.aten._fused_rms_norm.default(
                active, [active_width], gamma, 1e-5
            )
            second_input = torch.ops.aten.reshape.default(normalized, [m, active_width])
            output = torch.ops.aten.mm.default(second_input, second_weight)
            if return_saved:
                return output, tail, rstd, active, second_input
            return output, tail

        inputs = (
            torch.randn(m, k, dtype=torch.bfloat16) * 0.02,
            torch.randn(k, full_width, dtype=torch.bfloat16) * 0.02,
            torch.ones(active_width, dtype=torch.bfloat16),
            torch.randn(active_width, output_width, dtype=torch.bfloat16) * 0.02,
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
        mm_nodes[0].meta["custom"] = {"module_fqn": "layers.0.attention.wkv_a"}
        norm.meta["custom"] = {"module_fqn": "layers.0.attention.kv_norm"}
        mm_nodes[1].meta["custom"] = {"module_fqn": "layers.0.attention.wkv_b"}
        return gm, inputs

    def _expected(self, inputs):
        a, first_weight, gamma, second_weight = inputs
        first = torch.mm(a, first_weight)
        active = first[:, :512]
        tail = first[:, 512:].reshape(2, 4, 64)
        gamma_full = torch.nn.functional.pad(gamma.reshape(1, 512), (0, 64), value=1)
        weighted = (first.float() * gamma_full).bfloat16()[:, :512]
        partial = first.float().reshape(8, -1, 64).square().mean(-1)[:, :8]
        rstd = (partial.mean(-1, keepdim=True) + 1e-5).rsqrt()
        output = (torch.mm(weighted, second_weight).float() * rstd).bfloat16()
        normalized = (active.float() * rstd * gamma).bfloat16()
        return output, tail, rstd, active, normalized

    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def test_reparameterizes_segmented_kv_norm(self):
        gm, inputs = self._trace_kv_chain()
        expected_output, expected_tail, _, _, _ = self._expected(inputs)

        fuse_f2_kv_rmsnorm_pass(gm)

        actual_output, actual_tail = gm(*inputs)
        torch.testing.assert_close(actual_output, expected_output, atol=1e-3, rtol=2e-2)
        self.assertEqual(actual_tail, expected_tail)
        self.assertEqual(len(self._flex_gemm_nodes(gm)), 2)
        self.assertFalse(
            any(
                node.target == torch.ops.aten._fused_rms_norm.default
                for node in gm.graph.nodes
            )
        )

    def test_preserves_recomputed_values_and_raw_tail(self):
        gm, inputs = self._trace_kv_chain(return_saved=True)
        expected_output, expected_tail, rstd, active, normalized = self._expected(
            inputs
        )

        fuse_f2_kv_rmsnorm_pass(gm)

        actual_output, actual_tail, actual_rstd, actual_active, actual_normalized = gm(
            *inputs
        )
        torch.testing.assert_close(actual_output, expected_output, atol=1e-3, rtol=2e-2)
        self.assertEqual(actual_tail, expected_tail)
        self.assertEqual(actual_rstd, rstd.reshape(2, 4, 1))
        self.assertEqual(actual_active, active.reshape(2, 4, 512))
        self.assertEqual(actual_normalized, normalized)
        self.assertEqual(len(self._flex_gemm_nodes(gm)), 2)


class TestB4RouterInputGradAddPass(TestCase):
    def _trace(self, module_fqn):
        x = torch.randn(8, 16)
        weight = torch.randn(16, 12)
        residual = torch.randn(2, 4, 12, dtype=torch.bfloat16)

        def fn(a, b, other_grad):
            router_grad = torch.mm(a, b).reshape(2, 4, 12).bfloat16()
            return other_grad + router_grad

        gm = make_fx(fn)(x, weight, residual)
        mm = next(
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default
        )
        mm.meta["custom"] = {
            "module_fqn": module_fqn,
            "autograd_backward": True,
        }
        return gm, (x, weight, residual)

    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def test_fuses_router_cast_and_expert_gradient_add(self):
        gm, inputs = self._trace("layers.3.moe.router.gate")
        expected = gm(*inputs)

        fuse_b4_router_input_grad_add_pass(gm)

        self.assertEqual(gm(*inputs), expected, exact_dtype=True)
        fused = self._flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 1)
        self.assertEqual(len(fused[0].args[2]), 3)
        body = getattr(gm, fused[0].args[1].target)
        targets = [node.target for node in body.graph.nodes]
        self.assertEqual(targets.count(torch.ops.aten._to_copy.default), 1)
        self.assertEqual(targets.count(torch.ops.aten.add.Tensor), 1)

    def test_does_not_fuse_unrelated_fp32_linear(self):
        gm, _ = self._trace("layers.3.attention.wo")

        fuse_b4_router_input_grad_add_pass(gm)

        self.assertEqual(self._flex_gemm_nodes(gm), [])


class TestB5MLARMSNormBackwardPass(TestCase):
    def _trace(self, *, module_fqn="layers.3.attention.wkv_b"):
        m, k, n = 8, 16, 512

        def fn(a, b, x, rstd, gamma):
            grad = torch.ops.aten.mm.default(a, b)
            grad = torch.ops.aten.reshape.default(grad, [2, 4, n])
            return torch.ops.aten._fused_rms_norm_backward.default(
                grad, x, [n], rstd, gamma, [True, True]
            )

        inputs = (
            torch.randn(m, k, dtype=torch.bfloat16) * 0.02,
            torch.randn(k, n, dtype=torch.bfloat16) * 0.02,
            torch.randn(2, 4, n, dtype=torch.bfloat16) * 0.02,
            torch.rand(2, 4, 1, dtype=torch.float32) + 0.5,
            torch.randn(n, dtype=torch.bfloat16) * 0.02,
        )
        gm = make_fx(fn, tracing_mode="fake")(*inputs)
        mm = next(
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default
        )
        mm.meta["custom"] = {"module_fqn": module_fqn}
        mm.meta["autograd_backward"] = True
        norm = next(
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.aten._fused_rms_norm_backward.default
        )
        norm.meta["autograd_backward"] = True
        return gm, inputs

    def _expected(self, inputs):
        a, b, x, rstd, gamma = inputs
        n = x.shape[-1]
        grad = torch.mm(a, b).reshape_as(x).float()
        x_hat = x.float() * rstd
        grad_x_hat = grad * gamma.float()
        row_dot = (x_hat * grad_x_hat).sum(-1, keepdim=True)
        grad_input = ((grad_x_hat - (x_hat / n) * row_dot) * rstd).bfloat16()
        grad_weight = (grad * x_hat).reshape(-1, n).sum(0).bfloat16()
        return grad_input, grad_weight

    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def test_fuses_mla_projection_and_rmsnorm_backward_partials(self):
        gm, inputs = self._trace()
        expected = self._expected(inputs)

        fuse_b5_mla_rmsnorm_backward_pass(gm)

        actual = gm(*inputs)
        self.assertEqual(actual, expected, exact_dtype=True)
        fused = self._flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 1)
        self.assertEqual(len(fused[0].args[2]), 5)
        self.assertFalse(
            any(
                node.target == torch.ops.aten._fused_rms_norm_backward.default
                for node in gm.graph.nodes
            )
        )
        body = getattr(gm, fused[0].args[1].target)
        targets = [node.target for node in body.graph.nodes]
        self.assertEqual(targets.count(torch.ops.aten.sum.dim_IntList), 1)

    def test_does_not_fuse_non_mla_projection(self):
        gm, _ = self._trace(module_fqn="layers.3.feed_forward.w2")

        fuse_b5_mla_rmsnorm_backward_pass(gm)

        self.assertEqual(self._flex_gemm_nodes(gm), [])


class TestB7AttentionGradMergePass(TestCase):
    def _trace(self, *, q_layer=3):
        m, kv_width, q_width, model_width = 8, 16, 24, 512

        def fn(kv_grad, kv_weight, q_grad, q_weight):
            kv_input_grad = torch.ops.aten.mm.default(kv_grad, kv_weight)
            kv_input_grad = torch.ops.aten.reshape.default(
                kv_input_grad, [2, 4, model_width]
            )
            q_input_grad = torch.ops.aten.mm.default(q_grad, q_weight)
            q_input_grad = torch.ops.aten.reshape.default(
                q_input_grad, [2, 4, model_width]
            )
            return torch.ops.aten.add.Tensor(kv_input_grad, q_input_grad)

        inputs = (
            torch.randn(m, kv_width, dtype=torch.bfloat16) * 0.02,
            torch.randn(kv_width, model_width, dtype=torch.bfloat16) * 0.02,
            torch.randn(m, q_width, dtype=torch.bfloat16) * 0.02,
            torch.randn(q_width, model_width, dtype=torch.bfloat16) * 0.02,
        )
        gm = make_fx(fn)(*inputs)
        mm_nodes = [
            node for node in gm.graph.nodes if node.target == torch.ops.aten.mm.default
        ]
        mm_nodes[0].meta["custom"] = {
            "module_fqn": "layers.3.attention.wkv_a",
        }
        mm_nodes[0].meta["autograd_backward"] = True
        mm_nodes[1].meta["custom"] = {
            "module_fqn": f"layers.{q_layer}.attention.wq_a",
        }
        mm_nodes[1].meta["autograd_backward"] = True
        return gm, inputs

    def _flex_gemm_nodes(self, gm):
        return [
            node
            for node in gm.graph.nodes
            if node.target == torch.ops.higher_order.flex_gemm
        ]

    def test_fuses_q_kv_input_gradient_add(self):
        gm, inputs = self._trace()
        expected = gm(*inputs)

        fuse_b7_attention_grad_merge_pass(gm)

        self.assertEqual(gm(*inputs), expected)
        fused = self._flex_gemm_nodes(gm)
        self.assertEqual(len(fused), 1)
        self.assertEqual(len(fused[0].args[2]), 3)
        fused_output = next(iter(fused[0].users))
        self.assertEqual(fused_output.meta["val"].shape, (8, 512))
        body = getattr(gm, fused[0].args[1].target)
        targets = [node.target for node in body.graph.nodes]
        self.assertEqual(targets.count(torch.ops.aten.add.Tensor), 1)

    def test_does_not_fuse_different_layers(self):
        gm, _ = self._trace(q_layer=4)

        fuse_b7_attention_grad_merge_pass(gm)

        self.assertEqual(self._flex_gemm_nodes(gm), [])

    def test_does_not_fuse_forward_graph(self):
        gm, _ = self._trace()
        for node in gm.graph.nodes:
            node.meta.pop("autograd_backward", None)

        fuse_b7_attention_grad_merge_pass(gm)

        self.assertEqual(self._flex_gemm_nodes(gm), [])


class TestCodaPatternRegistry(TestCase):
    def test_resolves_configured_order(self):
        passes = get_coda_pattern_passes(
            [
                "b1_lm_head_input_grad_cast",
                "f6_router_sigmoid_bias",
                "f3_residual_rmsnorm",
                "f4_dense_swiglu",
                "b2_dense_swiglu_backward",
                "f2_q_rmsnorm",
                "f2_kv_rmsnorm",
                "b4_router_input_grad_add",
                "b5_mla_rmsnorm_backward",
                "b6_bf16_weight_grad_cast",
                "b7_attention_grad_merge",
            ]
        )
        self.assertEqual(
            passes,
            [
                fuse_b1_lm_head_input_grad_cast_pass,
                fuse_f6_router_sigmoid_bias_pass,
                fuse_f3_residual_rmsnorm_pass,
                fuse_f4_dense_swiglu_pass,
                fuse_b2_dense_swiglu_backward_pass,
                fuse_f2_q_rmsnorm_pass,
                fuse_f2_kv_rmsnorm_pass,
                fuse_b4_router_input_grad_add_pass,
                fuse_b5_mla_rmsnorm_backward_pass,
                fuse_b6_bf16_weight_grad_cast_pass,
                fuse_b7_attention_grad_merge_pass,
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

    def test_allows_f4_before_f3(self):
        passes = get_coda_pattern_passes(["f4_dense_swiglu", "f3_residual_rmsnorm"])

        self.assertEqual(
            passes,
            [fuse_f4_dense_swiglu_pass, fuse_f3_residual_rmsnorm_pass],
        )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
