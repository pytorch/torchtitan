# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import unittest
from unittest import mock

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.regional_inductor import _create_inductor_marked_regions
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer import coda_passes, compile_time_benchmark
from torchtitan.experiments.graph_trainer.coda_passes import (
    _apply_coda_candidate,
    _CODA_PATTERNS,
    coda_flex_gemm_pass,
    CODA_PATTERN_NAMES,
    get_coda_pattern_passes,
    materialize_coda_inductor_regions_pass,
)
from torchtitan.experiments.graph_trainer.compile_time_benchmark import (
    apply_benchmarked_rewrites,
    clear_compile_time_benchmark_cache,
    CompileTimeBenchmarkResult,
)


aten = torch.ops.aten
flex_gemm_hop = torch.ops.higher_order.flex_gemm


def _matmuls(gm):
    return [node for node in gm.graph.nodes if node.target is aten.mm.default]


def _flex_gemm_patterns(gm):
    return [
        node.meta.get("custom", {}).get("coda_pattern")
        for node in gm.graph.nodes
        if node.target is flex_gemm_hop
    ]


def _apply_benchmarked_coda_pattern(
    gm,
    pattern,
    *,
    coda_autotune,
    strict=False,
    benchmark_region=None,
):
    return apply_benchmarked_rewrites(
        gm,
        rewrite_name=f"CODA {pattern.name}",
        apply_candidate=functools.partial(_apply_coda_candidate, pattern=pattern),
        namespace=("coda_flex_gemm", pattern.name, coda_autotune),
        strict=strict,
        benchmark_region=benchmark_region,
        report_title=f"CODA benchmark results for {pattern.name}",
        artifact_name=f"coda_benchmark_{pattern.name}",
        candidate_prefix=f"{pattern.name}:",
        candidate_label="FlexGEMM",
        finalize=coda_passes._prepare_coda_graph,
        batch_candidates=True,
    )


def _tag_graph(gm, fqns, *, backward=False, kimi=False):
    for node in gm.graph.nodes:
        if backward:
            node.meta["autograd_backward"] = True
        if kimi:
            node.meta.setdefault("custom", {})["coda_model"] = "kimi_k3"
    matmuls = _matmuls(gm)
    if len(matmuls) != len(fqns):
        raise AssertionError(f"expected {len(fqns)} matmuls, found {len(matmuls)}")
    for node, fqn in zip(matmuls, fqns):
        node.meta.setdefault("custom", {})["module_fqn"] = fqn


def _run_pass(
    module,
    inputs,
    fqns,
    *,
    backward=False,
    kimi=False,
    kimi_stack=False,
    atol=2e-2,
    rtol=2e-2,
):
    gm = make_fx(module)(*inputs)
    reference = gm(*inputs)
    _tag_graph(gm, fqns, backward=backward, kimi=kimi)
    if kimi_stack:
        for node in gm.graph.nodes:
            if node.target is not aten.mm.default:
                node.meta["stack_trace"] = "modeling_kimi_linear.py"
    gm = coda_flex_gemm_pass(gm, inputs, compile_time_benchmark=False)
    fused_patterns = _flex_gemm_patterns(gm)
    for fused in (node for node in gm.graph.nodes if node.target is flex_gemm_hop):
        body_attr = fused.args[1]
        body = gm.get_submodule(body_attr.target)
        for body_node in body.graph.nodes:
            if body_node.op != "output" and "val" not in body_node.meta:
                raise AssertionError(
                    f"{body_attr.target}:{body_node.name} is missing value metadata"
                )
    for pattern, count in gm.meta["coda_pattern_counts"].items():
        if fused_patterns.count(pattern) < count:
            raise AssertionError(
                f"{pattern} counted {count} matches but emitted only "
                f"{fused_patterns.count(pattern)} FlexGEMM calls"
            )
    actual = gm(*inputs)
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    return gm


class _ProjectionNorm(torch.nn.Module):
    def __init__(self, split_projection):
        super().__init__()
        self.split_projection = split_projection

    def forward(self, x, w_a, norm_weight, w_b):
        projection = x @ w_a
        if self.split_projection:
            norm_input, rope = torch.split(projection, [128, 64], dim=-1)
        else:
            norm_input, rope = projection, projection
        norm, rstd = aten._fused_rms_norm.default(norm_input, [128], norm_weight, 1e-5)
        return projection, rope, norm @ w_b, rstd


class _ProjectionNormWithBackwardConsumer(torch.nn.Module):
    def forward(self, x, w_a, norm_weight, w_b, backward_weight):
        projection = x @ w_a
        norm, rstd = aten._fused_rms_norm.default(
            projection,
            [128],
            norm_weight,
            1e-5,
        )
        viewed = norm.view(norm.shape)
        return viewed @ w_b, viewed @ backward_weight, rstd


class _SegmentedProjectionNorm(torch.nn.Module):
    def forward(self, x, w_a, norm_weight, w_b):
        projection = x @ w_a
        norm_input, rope = torch.split(projection, [256, 32], dim=-1)
        norm, rstd = aten._fused_rms_norm.default(
            norm_input,
            [256],
            norm_weight,
            1e-5,
        )
        return projection, rope, norm @ w_b, rstd


class _PrimitiveProjectionNorm(torch.nn.Module):
    def forward(self, x, w_a, norm_weight, w_b, grad, backward_weight):
        projection = x @ w_a
        input_float = projection.float()
        rstd = torch.rsqrt(input_float.square().mean(-1, keepdim=True) + 1e-6)
        normalized = (input_float * rstd).to(projection.dtype)
        norm = norm_weight * normalized
        expanded = norm @ w_b

        norm_grad = grad @ backward_weight
        grad_weight = (norm_grad * normalized).sum(dim=0)
        grad_weighted = (norm_grad * norm_weight).float()
        direct = grad_weighted * rstd
        dot = (grad_weighted * input_float).sum(dim=-1, keepdim=True)
        correction = (-0.5 * dot * rstd.pow(3)).expand_as(input_float)
        correction = correction / projection.shape[-1] * (2.0 * input_float)
        grad_input = (direct + correction).to(projection.dtype)
        return projection, expanded, rstd, grad_input, grad_weight


class _ResidualNorm(torch.nn.Module):
    def forward(self, x, projection_weight, residual, norm_weight):
        hidden = x @ projection_weight + residual
        norm, rstd = aten._fused_rms_norm.default(hidden, [128], norm_weight, 1e-5)
        return hidden, norm, rstd


class _BatchedResidualNorm(torch.nn.Module):
    def forward(self, x, projection_weight, residual, norm_weight):
        projection = (x.view(-1, x.shape[-1]) @ projection_weight).view(
            *x.shape[:-1], projection_weight.shape[-1]
        )
        hidden = projection + residual
        norm, rstd = aten._fused_rms_norm.default(hidden, [128], norm_weight, 1e-5)
        return hidden, norm, rstd


class _WeightedResidualNorm(torch.nn.Module):
    def forward(self, probs, values, norm_weight):
        hidden = torch.bmm(probs, values).squeeze(1).to(torch.bfloat16)
        norm, rstd = aten._fused_rms_norm.default(hidden, [128], norm_weight, 1e-5)
        return hidden, norm, rstd


class _ResidualNormAlpha(torch.nn.Module):
    def forward(self, x, projection_weight, residual, norm_weight):
        hidden = aten.add.Tensor(x @ projection_weight, residual, alpha=2)
        norm, rstd = aten._fused_rms_norm.default(hidden, [128], norm_weight, 1e-5)
        return hidden, norm, rstd


class _ProjectionNormRightOperand(torch.nn.Module):
    def forward(self, x, projection_weight, norm_weight, expansion_weight):
        projection = x @ projection_weight
        norm, rstd = aten._fused_rms_norm.default(projection, [128], norm_weight, 1e-5)
        return expansion_weight @ norm, rstd


class _ProjectionNormWrongSplitDim(torch.nn.Module):
    def forward(self, x, projection_weight, norm_weight, expansion_weight):
        projection = x @ projection_weight
        norm_input, remainder = torch.split(projection, [64, 64], dim=0)
        norm, rstd = aten._fused_rms_norm.default(norm_input, [192], norm_weight, 1e-5)
        return projection, remainder, norm @ expansion_weight, rstd


class _TransposedProjectionNormBackward(torch.nn.Module):
    def forward(self, grad, projection_weight, norm_input, rstd, norm_weight):
        projected = (grad @ projection_weight).t()
        return aten._fused_rms_norm_backward.default(
            projected, norm_input, [128], rstd, norm_weight, [True, True]
        )


class _SwiGLU(torch.nn.Module):
    def forward(self, x, w1, w3):
        return torch.nn.functional.silu(x @ w1) * (x @ w3)


class _SingleActivation(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, x, weight):
        return self.activation(x @ weight)


class _SharedMatmulEpilogues(torch.nn.Module):
    def forward(self, x, weight):
        projection = x @ weight
        return torch.nn.functional.silu(projection), torch.sigmoid(projection)


class _SiTU(torch.nn.Module):
    def forward(self, x, gate_weight, up_weight):
        gate = (x @ gate_weight).float()
        up = (x @ up_weight).float()
        activated_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        transformed_up = 25.0 * torch.tanh(up / 25.0)
        return (activated_gate * transformed_up).bfloat16()


class _SiTUCat(torch.nn.Module):
    def forward(self, x, gate_weight, up_weight):
        gate_up = torch.cat([x @ gate_weight, x @ up_weight], dim=-1)
        width = gate_up.shape[-1] // 2
        gate = gate_up[..., :width].float()
        up = gate_up[..., width:].float()
        activated_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        transformed_up = 25.0 * torch.tanh(up / 25.0)
        return (activated_gate * transformed_up).bfloat16()


class _SigmoidEpilogue(torch.nn.Module):
    def __init__(self, add_bias):
        super().__init__()
        self.add_bias = add_bias

    def forward(self, x, weight, auxiliary):
        gate = torch.sigmoid(x @ weight)
        if self.add_bias:
            return gate, gate + auxiliary
        return gate, gate * auxiliary


class _ProjectionNormBackward(torch.nn.Module):
    def forward(self, grad, projection_weight, norm_input, rstd, norm_weight):
        projected = grad @ projection_weight
        return aten._fused_rms_norm_backward.default(
            projected, norm_input, [128], rstd, norm_weight, [True, True]
        )


class _BackwardCast(torch.nn.Module):
    def __init__(self, transpose_output):
        super().__init__()
        self.transpose_output = transpose_output

    def forward(self, lhs, rhs):
        output = lhs @ rhs
        if self.transpose_output:
            output = output.t()
        return output.float()


class _TwoBackwardCasts(torch.nn.Module):
    def forward(self, lhs_a, rhs_a, lhs_b, rhs_b):
        return (lhs_a @ rhs_a).float(), (lhs_b @ rhs_b).float()


class _FourBackwardCasts(torch.nn.Module):
    def forward(self, *inputs):
        return tuple(
            (inputs[index] @ inputs[index + 1]).float()
            for index in range(0, len(inputs), 2)
        )


class _BackwardReshapedCast(torch.nn.Module):
    def forward(self, lhs, rhs):
        return (lhs @ rhs).reshape(2, 4, -1).float()


class _BackwardMerge(torch.nn.Module):
    def forward(self, x1, w1, x2, w2):
        return x1 @ w1 + x2 @ w2


class _BackwardAccumulate(torch.nn.Module):
    def forward(self, x, weight, accumulated):
        return x @ weight + accumulated


class _SwiGLUBackward(torch.nn.Module):
    def forward(self, grad, w2, saved_silu, saved_gate, saved_w1, w3, w1):
        branch_grad = grad @ w2
        gate_grad = branch_grad * saved_silu
        silu_grad = aten.silu_backward.default(branch_grad * saved_gate, saved_w1)
        return gate_grad @ w3 + silu_grad @ w1


class _SiTUBackward(torch.nn.Module):
    def forward(self, grad, w2, saved_gate, saved_up, gate_weight, up_weight):
        branch_grad = (grad @ w2).float()
        gate = saved_gate.float()
        up = saved_up.float()
        sigmoid_gate = torch.sigmoid(gate)
        tanh_gate = torch.tanh(gate / 4.0)
        tanh_up = torch.tanh(up / 25.0)
        activated_gate = 4.0 * tanh_gate * sigmoid_gate
        transformed_up = 25.0 * tanh_up
        gate_derivative = (
            1.0 - tanh_gate.square()
        ) * sigmoid_gate + 4.0 * tanh_gate * sigmoid_gate * (1.0 - sigmoid_gate)
        gate_grad = (branch_grad * transformed_up * gate_derivative).bfloat16()
        up_grad = (branch_grad * activated_gate * (1.0 - tanh_up.square())).bfloat16()
        return gate_grad @ gate_weight + up_grad @ up_weight


class _SiTUCatBackward(torch.nn.Module):
    def forward(
        self,
        grad,
        down_weight,
        saved_tanh,
        saved_sigmoid,
        gate_weight,
        up_weight,
    ):
        branch_grad = (grad @ down_weight).float()
        up_grad = aten.tanh_backward.default(branch_grad, saved_tanh).bfloat16()
        gate_grad = aten.sigmoid_backward.default(branch_grad, saved_sigmoid).bfloat16()
        shape = [branch_grad.shape[0], branch_grad.shape[1] * 2]
        end = torch.iinfo(torch.int64).max
        joined = aten.slice_backward.default(
            up_grad, shape, 1, branch_grad.shape[1], end, 1
        ) + aten.slice_backward.default(gate_grad, shape, 1, 0, branch_grad.shape[1], 1)
        gate = joined[:, : branch_grad.shape[1]]
        up = joined[:, branch_grad.shape[1] :]
        return gate @ gate_weight + up @ up_weight


class _MlaOutputGateBackward(torch.nn.Module):
    def forward(self, grad, output_weight, attention, sigmoid_gate):
        gated_grad = grad @ output_weight
        attention_grad = gated_grad * sigmoid_gate
        saved_sigmoid_gate = aten.alias.default(sigmoid_gate)
        gate_grad = aten.sigmoid_backward.default(
            gated_grad * attention, saved_sigmoid_gate
        )
        return attention_grad, gate_grad


class _InterleavedSwiGLUBackward(torch.nn.Module):
    def forward(
        self,
        grad,
        w2,
        saved_gate,
        saved_w1,
        source,
        source_weight,
        w3,
        w1,
    ):
        branch_grad = grad @ w2
        silu_grad = aten.silu_backward.default(branch_grad * saved_gate, saved_w1)
        early_use = silu_grad.sum()
        saved_silu = source @ source_weight
        gate_grad = branch_grad * saved_silu
        merged = gate_grad @ w3 + silu_grad @ w1
        return early_use, merged


class TestCODAFlexGemmPass(TestCase):
    def test_flex_gemm_body_alias_detection(self):
        graph = torch.fx.Graph()
        placeholder = graph.placeholder("x")
        fake_mode = FakeTensorMode()
        with fake_mode:
            value = torch.randn(8)
            alias = value.view(2, 4)
            independent = value + 1
        placeholder.meta["val"] = value
        self.assertTrue(
            coda_passes._flex_gemm_body_has_aliasing(
                (placeholder,), (alias, independent)
            )
        )
        self.assertFalse(
            coda_passes._flex_gemm_body_has_aliasing(
                (placeholder,), (value + 1, value + 2)
            )
        )

        other = graph.placeholder("other")
        other.meta["val"] = alias
        self.assertTrue(
            coda_passes._flex_gemm_body_has_aliasing(
                (placeholder, other), (value + 1,)
            )
        )

    def test_structural_patterns_do_not_require_module_fqns(self):
        forward_inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
        )
        backward_inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 32, dtype=torch.bfloat16),
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 32, dtype=torch.bfloat16),
        )
        cases = (
            (_SwiGLU(), forward_inputs, "F_swiglu", False),
            (_SiTU(), forward_inputs, "F_situ", False),
            (
                _BackwardReshapedCast(),
                forward_inputs[:2],
                "B_reshape_bf16_to_fp32",
                True,
            ),
            (_BackwardMerge(), backward_inputs, "B_parallel_mm_dx_merge", True),
            (
                _BackwardAccumulate(),
                backward_inputs[:2] + (torch.randn(16, 32, dtype=torch.bfloat16),),
                "B_mm_dx_residual_add",
                True,
            ),
        )
        for module, inputs, pattern, backward in cases:
            with self.subTest(pattern=pattern):
                gm = make_fx(module)(*inputs)
                reference = gm(*inputs)
                if backward:
                    for node in gm.graph.nodes:
                        node.meta["autograd_backward"] = True

                gm = coda_flex_gemm_pass(
                    gm,
                    inputs,
                    patterns=[pattern],
                    compile_time_benchmark=False,
                )

                self.assertEqual(gm(*inputs), reference)
                self.assertEqual(gm.meta["coda_pattern_counts"][pattern], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_forward_projection_rmsnorm_patterns(self):
        x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
        norm_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        for split_projection, width, first_fqn, pattern in (
            (False, 128, "layers.0.attention.wq_a", "F_mla_qproj_rmsnorm_expand"),
            (
                True,
                192,
                "layers.0.attention.wkv_a",
                "F_mla_kvproj_rmsnorm_expand",
            ),
        ):
            with self.subTest(pattern=pattern):
                inputs = (
                    x,
                    torch.randn(64, width, device="cuda", dtype=torch.bfloat16),
                    norm_weight,
                    torch.randn(128, 256, device="cuda", dtype=torch.bfloat16),
                )
                gm = _run_pass(
                    _ProjectionNorm(split_projection),
                    inputs,
                    [first_fqn, first_fqn.replace("_a", "_b")],
                    atol=0.15,
                )
                self.assertEqual(gm.meta["coda_pattern_counts"][pattern], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_kimi_16b_segmented_kv_projection_compiles(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 288, device="cuda", dtype=torch.bfloat16),
            torch.randn(256, device="cuda", dtype=torch.bfloat16),
            torch.randn(256, 128, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_SegmentedProjectionNorm())(*inputs)
        reference = gm(*inputs)
        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["F_mla_kvproj_rmsnorm_expand"],
            compile_time_benchmark=False,
        )

        actual = torch.compile(gm, backend="inductor", fullgraph=True)(*inputs)
        self.assertEqual(actual, reference, atol=0.15, rtol=0.03)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_forward_projection_rmsnorm_ignores_backward_consumer(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, 256, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, 32, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_ProjectionNormWithBackwardConsumer())(*inputs)
        reference = gm(*inputs)
        _tag_graph(
            gm,
            [
                "layers.0.attention.wq_a",
                "layers.0.attention.wq_b",
                "layers.0.attention.wq_b",
            ],
        )
        _matmuls(gm)[-1].meta["autograd_backward"] = True

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["F_mla_qproj_rmsnorm_expand"],
            compile_time_benchmark=False,
        )

        for result, expected in zip(gm(*inputs), reference, strict=True):
            self.assertEqual(result, expected, atol=0.15, rtol=0.03)
        self.assertEqual(
            gm.meta["coda_pattern_counts"]["F_mla_qproj_rmsnorm_expand"],
            1,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_segmented_projection_rmsnorm_uses_normalized_width(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 288, device="cuda", dtype=torch.bfloat16),
            torch.randn(256, device="cuda", dtype=torch.bfloat16),
            torch.randn(256, 512, device="cuda", dtype=torch.bfloat16),
        )

        gm = _run_pass(
            _SegmentedProjectionNorm(),
            inputs,
            ["layers.0.attention.kv_a_proj", "layers.0.attention.kv_b_proj"],
            atol=0.15,
            rtol=0.03,
        )

        self.assertEqual(
            gm.meta["coda_pattern_counts"]["F_mla_kvproj_rmsnorm_expand"],
            1,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_kimi_primitive_projection_rmsnorm_forward_backward(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.02,
            torch.ones(128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) * 0.02,
            torch.randn(16, 256, device="cuda", dtype=torch.bfloat16) * 0.02,
            torch.randn(256, 128, device="cuda", dtype=torch.bfloat16) * 0.02,
        )
        gm = make_fx(_PrimitiveProjectionNorm())(*inputs)
        reference = gm(*inputs)
        _tag_graph(
            gm,
            [
                "layers.0.self_attn.q_a_proj",
                "layers.0.self_attn.q_b_proj",
                "layers.0.self_attn.q_b_proj",
            ],
        )
        matmuls = _matmuls(gm)
        backward_start = list(gm.graph.nodes).index(matmuls[2])
        for index, node in enumerate(gm.graph.nodes):
            if "aten" in str(node.target) and index < backward_start:
                node.meta["stack_trace"] = "modeling_kimi_linear.py"
            if index >= backward_start:
                node.meta["autograd_backward"] = True

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=[
                "F_mla_qproj_rmsnorm_expand",
                "B_mm_dx_rmsnorm",
            ],
            compile_time_benchmark=False,
        )

        actual = gm(*inputs)
        for result, expected in zip(actual, reference):
            torch.testing.assert_close(result, expected, atol=0.15, rtol=0.03)
        self.assertEqual(
            gm.meta["coda_pattern_counts"]["F_mla_qproj_rmsnorm_expand"], 0
        )
        self.assertEqual(gm.meta["coda_pattern_counts"]["B_mm_dx_rmsnorm"], 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_forward_residual_norm_patterns(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_ResidualNorm())(*inputs)
        reference = gm(*inputs)
        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["F_mm_residual_rmsnorm"],
            compile_time_benchmark=False,
        )

        self.assertEqual(gm(*inputs), reference, atol=2e-2, rtol=2e-2)
        self.assertEqual(gm.meta["coda_pattern_counts"]["F_mm_residual_rmsnorm"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_forward_residual_norm_accepts_batched_view(self):
        inputs = (
            torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )

        gm = _run_pass(_BatchedResidualNorm(), inputs, ["layers.0.attention.wo"])

        self.assertEqual(
            gm.meta["coda_pattern_counts"]["F_mm_residual_rmsnorm"],
            1,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_forward_weighted_residual_bmm_prenorm(self):
        inputs = (
            torch.softmax(torch.randn(16, 1, 4, device="cuda"), dim=-1),
            torch.randn(16, 4, 128, device="cuda"),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_WeightedResidualNorm())(*inputs)
        reference = gm(*inputs)

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["F_weighted_residual_bmm_prenorm"],
            compile_time_benchmark=False,
        )

        self.assertEqual(gm(*inputs), reference, atol=2e-2, rtol=2e-2)
        self.assertEqual(
            gm.meta["coda_pattern_counts"]["F_weighted_residual_bmm_prenorm"],
            1,
        )
        fused = next(node for node in gm.graph.nodes if node.target is flex_gemm_hop)
        self.assertIs(fused.args[0], aten.bmm.default)
        self.assertEqual(fused.args[4], {"backend": "TRITON"})
        self.assertEqual(
            torch.compile(gm, backend="inductor", fullgraph=True)(*inputs),
            reference,
            atol=2e-2,
            rtol=2e-2,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_weighted_residual_benchmark_region(self):
        inputs = (
            torch.softmax(torch.randn(16, 1, 4, device="cuda"), dim=-1),
            torch.randn(16, 4, 128, device="cuda"),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_WeightedResidualNorm())(*inputs)
        reference = gm(*inputs)
        benchmark_region = mock.Mock(return_value=CompileTimeBenchmarkResult(2.0, 1.0))

        clear_compile_time_benchmark_cache()
        result = _apply_benchmarked_coda_pattern(
            gm,
            _CODA_PATTERNS["F_weighted_residual_bmm_prenorm"],
            coda_autotune=False,
            benchmark_region=benchmark_region,
        )

        self.assertEqual(benchmark_region.call_count, 1)
        self.assertEqual(result(*inputs), reference, atol=2e-2, rtol=2e-2)
        self.assertEqual(
            result.meta["coda_pattern_counts"]["F_weighted_residual_bmm_prenorm"],
            1,
        )

    def test_kimi_mla_output_gate_backward(self):
        inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 64, dtype=torch.bfloat16),
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.sigmoid(torch.randn(16, 64, dtype=torch.bfloat16)),
        )
        gm = make_fx(_MlaOutputGateBackward())(*inputs)
        reference = gm(*inputs)
        for node in gm.graph.nodes:
            node.meta["autograd_backward"] = True

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["B_k3_mla_output_gate"],
            compile_time_benchmark=False,
        )

        self.assertEqual(gm(*inputs), reference)
        self.assertEqual(gm.meta["coda_pattern_counts"]["B_k3_mla_output_gate"], 1)
        self.assertEqual(_flex_gemm_patterns(gm), ["B_k3_mla_output_gate"])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_residual_norm_preserves_add_alpha(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )
        gm = _run_pass(_ResidualNormAlpha(), inputs, ["layers.0.attention.wo"])
        self.assertEqual(gm.meta["coda_pattern_counts"]["F_mm_residual_rmsnorm"], 1)
        fused = next(node for node in gm.graph.nodes if node.target is flex_gemm_hop)
        body = gm.get_submodule(fused.args[1].target)
        add = next(node for node in body.graph.nodes if node.target is aten.add.Tensor)
        self.assertEqual(add.kwargs, {"alpha": 2})

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_projection_rmsnorm_rejects_rhs_expansion(self):
        inputs = (
            torch.randn(128, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
            torch.randn(256, 128, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_ProjectionNormRightOperand())(*inputs)
        _tag_graph(
            gm,
            ["layers.0.attention.wq_a", "layers.0.attention.wq_b"],
        )

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["F_mla_qproj_rmsnorm_expand"],
            compile_time_benchmark=False,
        )

        self.assertEqual(
            gm.meta["coda_pattern_counts"]["F_mla_qproj_rmsnorm_expand"], 0
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_projection_rmsnorm_rejects_wrong_split_dim(self):
        inputs = (
            torch.randn(128, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 192, device="cuda", dtype=torch.bfloat16),
            torch.randn(192, device="cuda", dtype=torch.bfloat16),
            torch.randn(192, 256, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_ProjectionNormWrongSplitDim())(*inputs)
        _tag_graph(
            gm,
            ["layers.0.attention.wkv_a", "layers.0.attention.wkv_b"],
        )

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["F_mla_kvproj_rmsnorm_expand"],
            compile_time_benchmark=False,
        )

        self.assertEqual(
            gm.meta["coda_pattern_counts"]["F_mla_kvproj_rmsnorm_expand"], 0
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_residual_norm_rejects_broadcast_residual(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(1, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )
        gm = _run_pass(_ResidualNorm(), inputs, ["layers.0.attention.wo"])
        self.assertEqual(
            gm.meta["coda_pattern_counts"]["F_mm_residual_rmsnorm"],
            0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_residual_norm_does_not_depend_on_module_names(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_ResidualNorm())(*inputs)
        reference = gm(*inputs)
        _tag_graph(gm, ["layers.26.moe.shared_experts.w2"])
        norm = next(
            node
            for node in gm.graph.nodes
            if node.target is aten._fused_rms_norm.default
        )
        norm.meta.setdefault("custom", {})["module_fqn"] = "norm"

        gm = coda_flex_gemm_pass(gm, inputs, compile_time_benchmark=False)

        self.assertEqual(
            gm.meta["coda_pattern_counts"]["F_mm_residual_rmsnorm"],
            1,
        )
        self.assertEqual(gm(*inputs), reference)

    def test_forward_activation_patterns(self):
        inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
        )
        cases = (
            (_SwiGLU(), "layers.0.feed_forward", "F_swiglu", False),
            (
                _SwiGLU(),
                "layers.0.moe.shared_experts",
                "F_swiglu",
                False,
            ),
            (_SiTU(), "layers.0.feed_forward", "F_situ", True),
            (
                _SiTU(),
                "layers.0.moe.shared_experts",
                "F_situ",
                True,
            ),
        )
        for module, prefix, pattern, kimi in cases:
            with self.subTest(pattern=pattern):
                gm = _run_pass(
                    module,
                    inputs,
                    [f"{prefix}.w1", f"{prefix}.w3"],
                    kimi=kimi,
                )
                self.assertEqual(gm.meta["coda_pattern_counts"][pattern], 1)
                fused = [
                    node for node in gm.graph.nodes if node.target is flex_gemm_hop
                ]
                regions = [
                    node.meta["custom"]["compile_with_inductor"]["inductor_region"]
                    for node in fused
                ]
                self.assertEqual(len(regions), len(set(regions)))

    def test_compound_activations_reject_single_branch(self):
        inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
        )
        cases = (
            (
                _SingleActivation(torch.nn.functional.silu),
                "F_swiglu",
            ),
            (_SingleActivation(torch.tanh), "F_situ"),
        )
        for module, pattern in cases:
            with self.subTest(pattern=pattern):
                gm = make_fx(module)(*inputs)
                _tag_graph(gm, ["layers.0.feed_forward.w1"])
                gm = coda_flex_gemm_pass(
                    gm,
                    inputs,
                    patterns=[pattern],
                    compile_time_benchmark=False,
                )
                self.assertEqual(gm.meta["coda_pattern_counts"][pattern], 0)
                self.assertEqual(_flex_gemm_patterns(gm), [])

    def test_compound_activation_does_not_require_semantic_role(self):
        inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
        )
        gm = make_fx(_SwiGLU())(*inputs)

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["F_swiglu"],
            compile_time_benchmark=False,
        )

        self.assertEqual(gm.meta["coda_pattern_counts"]["F_swiglu"], 1)
        self.assertEqual(_flex_gemm_patterns(gm), ["F_swiglu", "F_swiglu"])

    def test_kimi_cat_situ_requires_layout_preserving_match(self):
        inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
        )
        gm = _run_pass(
            _SiTUCat(),
            inputs,
            ["layers.0.mlp.gate_proj", "layers.0.mlp.up_proj"],
            kimi_stack=True,
        )
        self.assertEqual(gm.meta["coda_pattern_counts"]["F_situ"], 0)

    def test_forward_sigmoid_patterns(self):
        x = torch.randn(16, 64, dtype=torch.bfloat16)
        weight = torch.randn(64, 32, dtype=torch.bfloat16)
        auxiliary = torch.randn(16, 32, dtype=torch.bfloat16)
        for add_bias, fqn, pattern in (
            (True, "layers.0.moe.router.gate", "F_router_sigmoid_bias"),
            (False, "layers.0.attention.g_proj", "F_k3_mla_output_gate"),
            (True, "layers.0.mlp.gate", "F_router_sigmoid_bias"),
            (False, "layers.0.self_attn.g_proj", "F_k3_mla_output_gate"),
        ):
            with self.subTest(pattern=pattern):
                gm = _run_pass(
                    _SigmoidEpilogue(add_bias),
                    (x, weight, auxiliary),
                    [fqn],
                    kimi_stack="self_attn" in fqn or fqn.endswith("mlp.gate"),
                )
                self.assertEqual(gm.meta["coda_pattern_counts"][pattern], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_backward_rmsnorm_patterns(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16),
            torch.rand(16, 1, device="cuda"),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_ProjectionNormBackward())(*inputs)
        reference = gm(*inputs)
        for node in gm.graph.nodes:
            node.meta["autograd_backward"] = True
        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["B_mm_dx_rmsnorm"],
            compile_time_benchmark=False,
        )

        self.assertEqual(gm(*inputs), reference, atol=0.15, rtol=0.05)
        self.assertEqual(gm.meta["coda_pattern_counts"]["B_mm_dx_rmsnorm"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_routed_up_rmsnorm_benchmark(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16),
            torch.rand(16, 1, device="cuda"),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_ProjectionNormBackward())(*inputs)
        reference = gm(*inputs)
        _tag_graph(gm, ["layers.0.moe.routed_up"], backward=True)
        benchmark_region = mock.Mock(return_value=CompileTimeBenchmarkResult(2.0, 1.0))

        clear_compile_time_benchmark_cache()
        result = _apply_benchmarked_coda_pattern(
            gm,
            _CODA_PATTERNS["B_mm_dx_rmsnorm"],
            coda_autotune=False,
            benchmark_region=benchmark_region,
        )

        self.assertEqual(benchmark_region.call_count, 1)
        self.assertEqual(result(*inputs), reference, atol=0.15, rtol=0.05)
        self.assertEqual(
            result.meta["coda_pattern_counts"]["B_mm_dx_rmsnorm"],
            1,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA RMSNorm")
    def test_backward_rmsnorm_rejects_transposed_projection(self):
        inputs = (
            torch.randn(128, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 16, device="cuda", dtype=torch.bfloat16),
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16),
            torch.rand(16, 1, device="cuda"),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )
        gm = _run_pass(
            _TransposedProjectionNormBackward(),
            inputs,
            ["layers.0.attention.wq_b"],
            backward=True,
        )
        self.assertEqual(gm.meta["coda_pattern_counts"]["B_mm_dx_rmsnorm"], 0)

    def test_backward_cast_patterns(self):
        inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
        )
        for module, fqn, pattern in (
            (
                _BackwardReshapedCast(),
                "model.lm_head",
                "B_reshape_bf16_to_fp32",
            ),
            (_BackwardCast(False), "unrelated", "B_linear_dw_bf16_to_fp32"),
            (_BackwardCast(True), "unrelated", "B_linear_dw_bf16_to_fp32"),
        ):
            with self.subTest(pattern=pattern):
                gm = _run_pass(module, inputs, [fqn], backward=True)
                self.assertEqual(gm.meta["coda_pattern_counts"][pattern], 1)

    def test_backward_merge_patterns(self):
        inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 32, dtype=torch.bfloat16),
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 32, dtype=torch.bfloat16),
        )
        cases = (
            (
                _BackwardAccumulate(),
                inputs[:2] + (torch.randn(16, 32, dtype=torch.bfloat16),),
                ["layers.0.moe.router.gate"],
                "B_mm_dx_residual_add",
            ),
            (
                _BackwardMerge(),
                inputs,
                ["layers.0.attention.wq", "layers.0.attention.wkv_a"],
                "B_parallel_mm_dx_merge",
            ),
        )
        for module, case_inputs, fqns, pattern in cases:
            with self.subTest(pattern=pattern):
                gm = _run_pass(module, case_inputs, fqns, backward=True)
                self.assertEqual(gm.meta["coda_pattern_counts"][pattern], 1)

    def test_backward_merge_rejects_fp32_gemm(self):
        inputs = (
            torch.randn(16, 64),
            torch.randn(64, 32),
            torch.randn(16, 64),
            torch.randn(64, 32),
        )
        gm = _run_pass(
            _BackwardMerge(),
            inputs,
            ["layers.0.attention.wkv_a", "layers.0.attention.wq"],
            backward=True,
        )
        self.assertEqual(gm.meta["coda_pattern_counts"]["B_parallel_mm_dx_merge"], 0)
        self.assertEqual(_flex_gemm_patterns(gm), [])

    def test_backward_merge_rejects_same_numel_broadcast(self):
        inputs = (
            torch.randn(2, 4, dtype=torch.bfloat16),
            torch.randn(4, 1, dtype=torch.bfloat16),
            torch.randn(1, 2, dtype=torch.bfloat16),
        )
        gm = make_fx(_BackwardAccumulate())(*inputs)
        reference = gm(*inputs)
        _tag_graph(gm, ["layers.0.moe.router.gate"], backward=True)

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["B_mm_dx_residual_add"],
            compile_time_benchmark=False,
        )

        self.assertEqual(gm.meta["coda_pattern_counts"]["B_mm_dx_residual_add"], 0)
        self.assertEqual(gm(*inputs), reference)

    def test_backward_activation_patterns(self):
        m, d, h = 16, 64, 128
        swiglu_inputs = (
            torch.randn(m, d, dtype=torch.bfloat16),
            torch.randn(d, h, dtype=torch.bfloat16),
            torch.randn(m, h, dtype=torch.bfloat16),
            torch.randn(m, h, dtype=torch.bfloat16),
            torch.randn(m, h, dtype=torch.bfloat16),
            torch.randn(h, d, dtype=torch.bfloat16),
            torch.randn(h, d, dtype=torch.bfloat16),
        )
        gm = _run_pass(
            _SwiGLUBackward(),
            swiglu_inputs,
            [
                "layers.0.feed_forward.w2",
                "layers.0.feed_forward.w3",
                "layers.0.feed_forward.w1",
            ],
            backward=True,
        )
        self.assertEqual(
            gm.meta["coda_pattern_counts"]["B_swiglu_backward_activation"], 1
        )
        self.assertEqual(gm.meta["coda_pattern_counts"]["B_parallel_mm_dx_merge"], 1)

        situ_inputs = (
            torch.randn(m, d, dtype=torch.bfloat16),
            torch.randn(d, h, dtype=torch.bfloat16),
            torch.randn(m, h, dtype=torch.bfloat16),
            torch.randn(m, h, dtype=torch.bfloat16),
            torch.randn(h, d, dtype=torch.bfloat16),
            torch.randn(h, d, dtype=torch.bfloat16),
        )
        gm = _run_pass(
            _SiTUBackward(),
            situ_inputs,
            [
                "layers.0.feed_forward.w2",
                "layers.0.feed_forward.w1",
                "layers.0.feed_forward.w3",
            ],
            backward=True,
            kimi=True,
        )
        self.assertEqual(
            gm.meta["coda_pattern_counts"]["B_k3_situ_backward_activation"], 0
        )
        self.assertEqual(gm.meta["coda_pattern_counts"]["B_parallel_mm_dx_merge"], 1)

        cat_inputs = (
            torch.randn(m, d, dtype=torch.bfloat16),
            torch.randn(d, h, dtype=torch.bfloat16),
            torch.tanh(torch.randn(m, h)),
            torch.sigmoid(torch.randn(m, h)),
            torch.randn(h, d, dtype=torch.bfloat16),
            torch.randn(h, d, dtype=torch.bfloat16),
        )
        gm = _run_pass(
            _SiTUCatBackward(),
            cat_inputs,
            [
                "layers.0.mlp.down_proj",
                "layers.0.mlp.gate_proj",
                "layers.0.mlp.up_proj",
            ],
            backward=True,
            kimi=True,
        )
        self.assertEqual(
            gm.meta["coda_pattern_counts"]["B_k3_situ_backward_activation"], 1
        )
        self.assertEqual(gm.meta["coda_pattern_counts"]["B_parallel_mm_dx_merge"], 1)

    def test_interleaved_backward_outputs_are_topologically_sorted(self):
        m, d, h = 16, 64, 128
        inputs = (
            torch.randn(m, d, dtype=torch.bfloat16),
            torch.randn(d, h, dtype=torch.bfloat16),
            torch.randn(m, h, dtype=torch.bfloat16),
            torch.randn(m, h, dtype=torch.bfloat16),
            torch.randn(m, d, dtype=torch.bfloat16),
            torch.randn(d, h, dtype=torch.bfloat16),
            torch.randn(h, d, dtype=torch.bfloat16),
            torch.randn(h, d, dtype=torch.bfloat16),
        )
        gm = _run_pass(
            _InterleavedSwiGLUBackward(),
            inputs,
            [
                "layers.0.feed_forward.w2",
                "layers.0.unrelated",
                "layers.0.feed_forward.w3",
                "layers.0.feed_forward.w1",
            ],
            backward=True,
        )

        gm.graph.lint()
        self.assertEqual(
            gm.meta["coda_pattern_counts"]["B_swiglu_backward_activation"], 1
        )
        self.assertEqual(gm.meta["coda_pattern_counts"]["B_parallel_mm_dx_merge"], 1)

    def test_all_structural_patterns_have_matchers(self):
        self.assertEqual(
            CODA_PATTERN_NAMES,
            (
                "F_mla_qproj_rmsnorm_expand",
                "F_mla_kvproj_rmsnorm_expand",
                "F_weighted_residual_bmm_prenorm",
                "F_mm_residual_rmsnorm",
                "F_swiglu",
                "F_situ",
                "F_k3_mla_output_gate",
                "F_router_sigmoid_bias",
                "B_reshape_bf16_to_fp32",
                "B_swiglu_backward_activation",
                "B_parallel_mm_dx_merge",
                "B_k3_mla_output_gate",
                "B_k3_situ_backward_activation",
                "B_mm_dx_residual_add",
                "B_mm_dx_rmsnorm",
                "B_linear_dw_bf16_to_fp32",
            ),
        )

    def test_epilogue_fusions_cannot_share_matmul_ownership(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_SharedMatmulEpilogues())(*inputs)
        root = next(node for node in gm.graph.nodes if node.target is aten.mm.default)
        silu = next(node for node in gm.graph.nodes if node.target is aten.silu.default)
        sigmoid = next(
            node for node in gm.graph.nodes if node.target is aten.sigmoid.default
        )

        first = coda_passes._insert_flex_gemm(
            gm,
            root=root,
            body_nodes={root, silu},
            pattern="F_k3_mla_output_gate",
        )
        second = coda_passes._insert_flex_gemm(
            gm,
            root=root,
            body_nodes={root, sigmoid},
            pattern="F_router_sigmoid_bias",
        )

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(root.meta["coda_owner"], "F_k3_mla_output_gate")

    def test_independent_coda_groups_use_distinct_inductor_regions(self):
        inputs = tuple(
            tensor
            for _ in range(4)
            for tensor in (
                torch.randn(8, 16, dtype=torch.bfloat16),
                torch.randn(16, 12, dtype=torch.bfloat16),
            )
        )
        gm = make_fx(_FourBackwardCasts())(*inputs)
        for node in gm.graph.nodes:
            node.meta["autograd_backward"] = True

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["B_linear_dw_bf16_to_fp32"],
            compile_time_benchmark=False,
        )

        regions = [
            node.meta["custom"]["compile_with_inductor"]["inductor_region"]
            for node in gm.graph.nodes
            if node.target is flex_gemm_hop
        ]
        self.assertEqual(len(regions), 4)
        self.assertEqual(len(regions), len(set(regions)))

        scooped = _create_inductor_marked_regions(gm)
        scooped.graph.lint()
        self.assertEqual(
            len(scooped.graph.find_nodes(op="call_module")),
            4,
        )

    def test_materializes_coda_regions_without_generic_partition_discovery(self):
        inputs = tuple(
            tensor
            for _ in range(4)
            for tensor in (
                torch.randn(8, 16, dtype=torch.bfloat16),
                torch.randn(16, 12, dtype=torch.bfloat16),
            )
        )
        gm = make_fx(_FourBackwardCasts())(*inputs)
        for node in gm.graph.nodes:
            node.meta["autograd_backward"] = True
        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["B_linear_dw_bf16_to_fp32"],
            compile_time_benchmark=False,
        )

        gm = materialize_coda_inductor_regions_pass(gm)

        gm.graph.lint()
        regions = [
            node
            for node in gm.graph.find_nodes(op="call_module")
            if str(node.target).startswith("__marked_inductor_submod_coda_")
        ]
        self.assertEqual(len(regions), 4)

    def test_compile_time_benchmark_accepts_only_faster_rewrite(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )

        def make_graph():
            gm = make_fx(_BackwardCast(False))(*inputs)
            _tag_graph(gm, ["layers.0.attention.wo"], backward=True)
            return gm

        clear_compile_time_benchmark_cache()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch.object(
                compile_time_benchmark,
                "infer_rewrite_regions",
                side_effect=AssertionError("whole-graph inference was used"),
            ),
        ):
            accepted = _apply_benchmarked_coda_pattern(
                make_graph(),
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=lambda *args: CompileTimeBenchmarkResult(2.0, 1.0),
            )
        self.assertEqual(len(_flex_gemm_patterns(accepted)), 1)

        clear_compile_time_benchmark_cache()
        original = make_graph()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch.object(
                compile_time_benchmark,
                "infer_rewrite_regions",
                side_effect=AssertionError("whole-graph inference was used"),
            ),
        ):
            rejected = _apply_benchmarked_coda_pattern(
                original,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=lambda *args: CompileTimeBenchmarkResult(1.0, 2.0),
            )
        self.assertIs(rejected, original)

    def test_transposed_weight_grad_uses_explicit_benchmark_region(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_BackwardCast(True))(*inputs)
        _tag_graph(gm, ["layers.0.attention.wo"], backward=True)
        benchmark_region = mock.Mock(
            return_value=CompileTimeBenchmarkResult(2.0, 1.0)
        )

        clear_compile_time_benchmark_cache()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch.object(
                compile_time_benchmark,
                "infer_rewrite_regions",
                side_effect=AssertionError("whole-graph inference was used"),
            ),
        ):
            result = _apply_benchmarked_coda_pattern(
                gm,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=benchmark_region,
            )

        benchmark_region.assert_called_once()
        self.assertEqual(
            result.meta["coda_pattern_counts"]["B_linear_dw_bf16_to_fp32"], 1
        )

    def test_compile_time_benchmark_applies_occurrences_independently(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
            torch.randn(4, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_TwoBackwardCasts())(*inputs)
        _tag_graph(gm, ["linear_a", "linear_b"], backward=True)
        benchmark_region = mock.Mock(
            side_effect=(
                CompileTimeBenchmarkResult(2.0, 1.0),
                CompileTimeBenchmarkResult(1.0, 2.0),
            )
        )

        clear_compile_time_benchmark_cache()
        copy_graph = coda_passes._copy_graph_module_for_rewrite
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch.object(
                coda_passes,
                "_copy_graph_module_for_rewrite",
                wraps=copy_graph,
            ) as copy_graph_mock,
        ):
            result = _apply_benchmarked_coda_pattern(
                gm,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=benchmark_region,
            )

        self.assertEqual(benchmark_region.call_count, 2)
        self.assertEqual(copy_graph_mock.call_count, 2)
        self.assertEqual(
            _flex_gemm_patterns(result),
            ["B_linear_dw_bf16_to_fp32"],
        )
        self.assertEqual(
            result.meta["coda_pattern_counts"]["B_linear_dw_bf16_to_fp32"], 1
        )

    def test_compile_time_benchmark_applies_swiglu_regions_independently(self):
        m, d, h = 16, 64, 128
        inputs = (
            torch.randn(m, d, dtype=torch.bfloat16),
            torch.randn(d, h, dtype=torch.bfloat16),
            torch.randn(m, h, dtype=torch.bfloat16),
            torch.randn(m, h, dtype=torch.bfloat16),
            torch.randn(m, h, dtype=torch.bfloat16),
            torch.randn(h, d, dtype=torch.bfloat16),
            torch.randn(h, d, dtype=torch.bfloat16),
        )
        gm = make_fx(_SwiGLUBackward())(*inputs)
        _tag_graph(
            gm,
            [
                "layers.0.feed_forward.w2",
                "layers.0.feed_forward.w3",
                "layers.0.feed_forward.w1",
            ],
            backward=True,
        )
        benchmark_region = mock.Mock(
            side_effect=(
                CompileTimeBenchmarkResult(1.0, 2.0),
                CompileTimeBenchmarkResult(2.0, 1.0),
            )
        )

        clear_compile_time_benchmark_cache()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            result = gm
            for pattern in (
                "B_swiglu_backward_activation",
                "B_parallel_mm_dx_merge",
            ):
                result = _apply_benchmarked_coda_pattern(
                    result,
                    _CODA_PATTERNS[pattern],
                    coda_autotune=False,
                    benchmark_region=benchmark_region,
                )

        self.assertEqual(benchmark_region.call_count, 2)
        self.assertEqual(
            _flex_gemm_patterns(result),
            ["B_parallel_mm_dx_merge"],
        )
        self.assertEqual(
            result.meta["coda_pattern_counts"]["B_swiglu_backward_activation"],
            0,
        )
        self.assertEqual(
            result.meta["coda_pattern_counts"]["B_parallel_mm_dx_merge"],
            1,
        )

    def test_compile_time_benchmark_reuses_equivalent_measurements(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_TwoBackwardCasts())(*inputs)
        _tag_graph(gm, ["linear_a", "linear_b"], backward=True)
        benchmark_region = mock.Mock(return_value=CompileTimeBenchmarkResult(2.0, 1.0))

        clear_compile_time_benchmark_cache()
        copy_graph = coda_passes._copy_graph_module_for_rewrite
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch.object(
                coda_passes,
                "_copy_graph_module_for_rewrite",
                wraps=copy_graph,
            ) as copy_graph_mock,
        ):
            result = _apply_benchmarked_coda_pattern(
                gm,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=benchmark_region,
            )

        benchmark_region.assert_called_once()
        copy_graph_mock.assert_called_once()
        self.assertEqual(
            result.meta["coda_pattern_counts"]["B_linear_dw_bf16_to_fp32"], 2
        )

    def test_compile_time_benchmark_rejects_invalid_rewrite(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        original = make_fx(_BackwardCast(False))(*inputs)
        _tag_graph(original, ["layers.0.attention.wo"], backward=True)
        benchmark_region = mock.Mock()

        def failing_apply(
            gm,
            pattern,
            *,
            log_matches=True,
            benchmark_regions=None,
            finalize=True,
        ):
            del log_matches, benchmark_regions, finalize
            cast = next(
                node for node in gm.graph.nodes if node.target is aten._to_copy.default
            )
            if coda_passes._claim_coda_match(pattern.name, cast):
                raise AssertionError("invalid rewrite")
            return gm

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(
                coda_passes, "_apply_coda_pattern", side_effect=failing_apply
            ),
        ):
            result = _apply_benchmarked_coda_pattern(
                original,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=benchmark_region,
            )

        self.assertIs(result, original)
        benchmark_region.assert_not_called()

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(
                coda_passes, "_apply_coda_pattern", side_effect=failing_apply
            ),
            self.assertRaisesRegex(RuntimeError, "candidate.*rewrite failed"),
        ):
            _apply_benchmarked_coda_pattern(
                original,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                strict=True,
                benchmark_region=benchmark_region,
            )

    def test_compile_time_benchmark_logs_grouped_results(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
            torch.randn(4, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_TwoBackwardCasts())(*inputs)
        _tag_graph(gm, ["linear_a", "linear_b"], backward=True)
        benchmark_region = mock.Mock(
            side_effect=(
                CompileTimeBenchmarkResult(2.0, 1.0),
                CompileTimeBenchmarkResult(1.0, 2.0),
            )
        )

        clear_compile_time_benchmark_cache()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch.object(compile_time_benchmark.logger, "info") as log,
            mock.patch.object(compile_time_benchmark, "trace_structured") as trace,
        ):
            _apply_benchmarked_coda_pattern(
                gm,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=benchmark_region,
            )

        summary = log.call_args.args[0]
        self.assertExpectedInline(
            summary,
            """CODA benchmark results for B_linear_dw_bf16_to_fp32: candidates=2, applied=1, rejected=1 (slower=1, failed=0)
  APPLIED (1):
    candidate _to_copy: region 0: eager=2000.0 us, FlexGEMM=1000.0 us, speedup=2.000x, cache=miss
  REJECTED (1):
    candidate _to_copy_1: region 0: eager=1000.0 us, FlexGEMM=2000.0 us, speedup=0.500x, cache=miss; FlexGEMM was not faster for every changed region""",  # noqa: B950
        )
        trace.assert_called_once()
        self.assertEqual(trace.call_args.args, ("artifact",))
        self.assertEqual(
            trace.call_args.kwargs["metadata_fn"](),
            {
                "name": "coda_benchmark_B_linear_dw_bf16_to_fp32",
                "encoding": "string",
            },
        )
        self.assertEqual(trace.call_args.kwargs["payload_fn"](), summary)

    def test_body_output_metadata_must_match_propagated_body(self):
        body = make_fx(lambda x: (x,))(torch.randn(2, 1))
        with self.assertRaisesRegex(AssertionError, "output 0 has spec"):
            coda_passes._validate_body_outputs(
                body,
                (torch.randn(1, 2),),
                "test_pattern",
            )

    def test_failed_direct_rewrite_does_not_mutate_original(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        original = make_fx(_BackwardCast(False))(*inputs)
        _tag_graph(original, ["layers.0.attention.wo"], backward=True)
        original_graph = str(original.graph)

        def failing_matcher(gm, counts, benchmark_regions=None):
            del counts, benchmark_regions
            output = next(node for node in gm.graph.nodes if node.op == "output")
            with gm.graph.inserting_before(output):
                gm.graph.call_function(aten.neg.default, (output.args[0],))
            raise RuntimeError("rewrite failed")

        pattern = coda_passes.CodaPattern("test_pattern", failing_matcher, 0, {})
        with self.assertRaisesRegex(RuntimeError, "rewrite failed"):
            coda_passes._apply_coda_pattern(original, pattern)

        self.assertEqual(str(original.graph), original_graph)

    def test_transactional_rewrite_does_not_deepcopy_module_state(self):
        class NonCopyableLeaf(torch.nn.Module):
            def forward(self, x):
                return x.neg()

            def __deepcopy__(self, memo):
                del memo
                raise AssertionError("module state was deep-copied")

        root = torch.nn.Module()
        root.add_module("leaf", NonCopyableLeaf())
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        output = graph.call_module("leaf", (x,))
        graph.output(output)
        original = torch.fx.GraphModule(root, graph)
        original_graph = str(original.graph)

        def matcher(gm, counts, benchmark_regions=None):
            del benchmark_regions
            counts["test_pattern"] += 1

        pattern = coda_passes.CodaPattern("test_pattern", matcher, 0, {})
        result = coda_passes._apply_coda_pattern(original, pattern)

        self.assertIsNot(result, original)
        self.assertIs(result.leaf, original.leaf)
        self.assertEqual(str(original.graph), original_graph)

    def test_configured_weight_grad_cast_autotunes(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_BackwardCast(False))(*inputs)
        _tag_graph(gm, ["layers.0.attention.wo"], backward=True)
        pattern = get_coda_pattern_passes(
            ["B_linear_dw_bf16_to_fp32"],
            compile_time_benchmark=False,
            coda_autotune=True,
        )[0]

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            gm = pattern(gm, inputs)

        fused = next(node for node in gm.graph.nodes if node.target is flex_gemm_hop)
        self.assertEqual(
            fused.args[4],
            {"backend": "QUACK", "tuned": True, "tune_split_k": True},
        )

    def test_configured_pass_uses_generic_compile_time_benchmark(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_BackwardCast(False))(*inputs)
        _tag_graph(gm, ["layers.0.attention.wo"], backward=True)
        pattern = get_coda_pattern_passes(
            ["B_linear_dw_bf16_to_fp32"],
            compile_time_benchmark=True,
            coda_autotune=False,
        )[0]

        with mock.patch.object(
            coda_passes,
            "apply_benchmarked_rewrites",
            return_value=gm,
        ) as benchmark:
            result = pattern(gm, inputs)

        self.assertIs(result, gm)
        self.assertEqual(benchmark.call_args.args, (gm,))
        self.assertEqual(
            benchmark.call_args.kwargs["rewrite_name"],
            "CODA B_linear_dw_bf16_to_fp32",
        )
        self.assertEqual(
            benchmark.call_args.kwargs["namespace"],
            ("coda_flex_gemm", "B_linear_dw_bf16_to_fp32", False),
        )

    def test_configured_pass_autotunes_without_pinned_config(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
            torch.randn(8, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_SigmoidEpilogue(False))(*inputs)
        pattern = get_coda_pattern_passes(
            ["F_k3_mla_output_gate"],
            compile_time_benchmark=False,
            coda_autotune=True,
        )[0]

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            gm = pattern(gm, inputs)

        fused = next(node for node in gm.graph.nodes if node.target is flex_gemm_hop)
        self.assertEqual(
            fused.args[4],
            {"backend": "QUACK", "tuned": True, "fast_math": True},
        )

    def test_resolves_registry_order_and_legacy_names(self):
        requested = [
            "B_lmhead_dx_bf16_to_fp32",
            "F_router_sigmoid_bias",
            "F_k3_dense_ffn_situ",
        ]
        passes = get_coda_pattern_passes(
            requested,
            compile_time_benchmark=False,
        )
        self.assertEqual(
            [pattern.__name__ for pattern in passes],
            [
                "F_situ",
                "F_router_sigmoid_bias",
                "B_reshape_bf16_to_fp32",
            ],
        )

        legacy = get_coda_pattern_passes(
            ["f3-dense", "k3-b-output-gate", "b6-weight-grad-cast"],
            compile_time_benchmark=False,
        )
        self.assertEqual(
            [pattern.__name__ for pattern in legacy],
            [
                "F_mm_residual_rmsnorm",
                "B_k3_mla_output_gate",
                "B_linear_dw_bf16_to_fp32",
            ],
        )

        split = get_coda_pattern_passes(
            ["B_swiglu_backward_dx_merge"],
            compile_time_benchmark=False,
        )
        self.assertEqual(
            [pattern.__name__ for pattern in split],
            ["B_swiglu_backward_activation", "B_parallel_mm_dx_merge"],
        )

        collapsed = get_coda_pattern_passes(
            ["b5-routed-rmsnorm", "b5-q-rmsnorm", "b5-kv-rmsnorm"],
            compile_time_benchmark=False,
        )
        self.assertEqual(
            [pattern.__name__ for pattern in collapsed], ["B_mm_dx_rmsnorm"]
        )

    def test_rejects_unknown_and_duplicate_patterns(self):
        with self.assertRaisesRegex(ValueError, "Unknown.*not_a_pattern"):
            get_coda_pattern_passes(["not_a_pattern"])
        with self.assertRaisesRegex(ValueError, "Duplicate.*F_situ"):
            get_coda_pattern_passes(["F_situ", "F_situ"])


if __name__ == "__main__":
    run_tests()
