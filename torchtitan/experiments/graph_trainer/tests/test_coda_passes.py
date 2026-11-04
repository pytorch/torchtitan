# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from collections import Counter
from unittest import mock

import torch
from torch._inductor.pattern_matcher import PatternExpr
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.regional_inductor import _create_inductor_marked_regions
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer import coda_passes, compile_time_benchmark
from torchtitan.experiments.graph_trainer.coda_passes import (
    _CODA_PATTERNS,
    coda_flex_gemm_pass,
    CODA_PATTERN_NAMES,
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
        name=pattern.name,
        prepare_candidates=lambda candidate: coda_passes._prepare_coda_candidates(
            candidate,
            pattern,
            autotune=coda_autotune,
        ),
        cache_key=("coda_flex_gemm", coda_autotune),
        strict=strict,
        benchmark_region=benchmark_region,
    )


def _configured_pattern_pass(
    name,
    *,
    compile_time_benchmark,
    coda_autotune=False,
    benchmark_graph_processor=None,
):
    return coda_passes._configured_coda_pass(
        _CODA_PATTERNS[name],
        compile_time_benchmark=compile_time_benchmark,
        benchmark_strict=False,
        coda_autotune=coda_autotune,
        benchmark_graph_processor=benchmark_graph_processor,
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
    rewrite_state = [
        node.name
        for node in gm.graph.nodes
        if {"coda_consumed", "coda_owner"} & node.meta.keys()
    ]
    if rewrite_state:
        raise AssertionError(f"CODA stored rewrite state on nodes: {rewrite_state}")
    if "_coda_nodes_to_erase" in gm.meta:
        raise AssertionError("CODA stored rewrite state on the GraphModule")
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
    def __init__(self, output_dtype=torch.bfloat16):
        super().__init__()
        self.output_dtype = output_dtype

    def forward(self, x, gate_weight, up_weight):
        gate = (x @ gate_weight).float()
        up = (x @ up_weight).float()
        activated_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
        transformed_up = 25.0 * torch.tanh(up / 25.0)
        return (activated_gate * transformed_up).to(self.output_dtype)


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
    def test_recompiles_once_after_all_patterns(self):
        inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
        )
        gm = make_fx(_SwiGLU())(*inputs)

        with mock.patch.object(gm, "recompile", wraps=gm.recompile) as recompile:
            coda_flex_gemm_pass(
                gm,
                inputs,
                patterns=["F_swiglu"],
                compile_time_benchmark=False,
            )

        recompile.assert_called_once_with()

    def test_new_node_metadata_reuses_input_fake_mode(self):
        fake_mode = torch._subclasses.FakeTensorMode()
        with fake_mode:
            fake_input = torch.empty(4, 8, device="cuda")
        other_mode = torch._subclasses.FakeTensorMode()
        with other_mode:
            other_input = torch.empty(4, 8, device="cuda")

        graph = torch.fx.Graph()
        input_node = graph.placeholder("x")
        input_node.meta["val"] = fake_input
        other_node = graph.placeholder("y")
        other_node.meta["val"] = other_input
        new_node = graph.call_function(aten.add.Tensor, (input_node, other_node))

        coda_passes._infer_new_node_meta(new_node)

        self.assertIs(new_node.meta["val"].fake_mode, fake_mode)

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

    def test_patterns_accept_fp16_storage_dtype(self):
        dtype = torch.float16
        forward_inputs = (
            torch.randn(16, 64, dtype=dtype),
            torch.randn(64, 128, dtype=dtype),
            torch.randn(64, 128, dtype=dtype),
        )
        backward_inputs = (
            torch.randn(16, 64, dtype=dtype),
            torch.randn(64, 32, dtype=dtype),
            torch.randn(16, 64, dtype=dtype),
            torch.randn(64, 32, dtype=dtype),
        )
        cases = (
            (_SiTU(dtype), forward_inputs, "F_situ", False),
            (
                _BackwardReshapedCast(),
                forward_inputs[:2],
                "B_reshape_bf16_to_fp32",
                True,
            ),
            (
                _BackwardCast(False),
                forward_inputs[:2],
                "B_linear_dw_bf16_to_fp32",
                True,
            ),
            (_BackwardMerge(), backward_inputs, "B_parallel_mm_dx_merge", True),
        )
        for module, inputs, pattern, backward in cases:
            with self.subTest(pattern=pattern):
                traced = make_fx(module)(*inputs)
                gm = _run_pass(
                    module,
                    inputs,
                    ["linear"] * len(_matmuls(traced)),
                    backward=backward,
                )
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
    def test_projection_rmsnorm_benchmark_uses_local_region(self):
        x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
        norm_weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
        for split_projection, width, first_fqn, pattern_name in (
            (False, 128, "layers.0.attention.wq_a", "F_mla_qproj_rmsnorm_expand"),
            (
                True,
                192,
                "layers.0.attention.wkv_a",
                "F_mla_kvproj_rmsnorm_expand",
            ),
        ):
            with self.subTest(pattern=pattern_name):
                inputs = (
                    x,
                    torch.randn(64, width, device="cuda", dtype=torch.bfloat16),
                    norm_weight,
                    torch.randn(128, 256, device="cuda", dtype=torch.bfloat16),
                )
                gm = make_fx(_ProjectionNorm(split_projection))(*inputs)
                reference = gm(*inputs)
                _tag_graph(gm, [first_fqn, first_fqn.replace("_a", "_b")])
                benchmark_region = mock.Mock(
                    return_value=CompileTimeBenchmarkResult(2.0, 1.0)
                )
                clear_compile_time_benchmark_cache()

                result = _apply_benchmarked_coda_pattern(
                    gm,
                    _CODA_PATTERNS[pattern_name],
                    coda_autotune=False,
                    benchmark_region=benchmark_region,
                )

                benchmark_region.assert_called_once()
                self.assertIs(result, gm)
                self.assertEqual(result(*inputs), reference, atol=0.15, rtol=0.03)
                self.assertEqual(result.meta["coda_pattern_counts"][pattern_name], 1)

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
    def test_forward_residual_norm_benchmark_uses_declared_region(self):
        inputs = (
            torch.randn(16, 64, device="cuda", dtype=torch.bfloat16),
            torch.randn(64, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(16, 128, device="cuda", dtype=torch.bfloat16),
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
        )
        gm = make_fx(_ResidualNorm())(*inputs)
        _tag_graph(gm, ["layers.0.attention.wo"])
        benchmark_region = mock.Mock(return_value=CompileTimeBenchmarkResult(2.0, 1.0))

        clear_compile_time_benchmark_cache()
        result = _apply_benchmarked_coda_pattern(
            gm,
            _CODA_PATTERNS["F_mm_residual_rmsnorm"],
            coda_autotune=False,
            benchmark_region=benchmark_region,
        )

        benchmark_region.assert_called_once()
        self.assertEqual(
            result.meta["coda_pattern_counts"]["F_mm_residual_rmsnorm"],
            1,
        )

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

    def test_compound_activation_validates_all_sites_before_rewriting(self):
        inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
        )
        gm = make_fx(_SwiGLU())(*inputs)
        matmuls = _matmuls(gm)
        counts = Counter()

        _CODA_PATTERNS["F_swiglu"].apply(
            gm,
            counts,
            {matmuls[1]},
        )

        self.assertEqual(counts["F_swiglu"], 0)
        self.assertEqual(_flex_gemm_patterns(gm), [])

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

    def test_transposed_cast_benchmark_uses_local_region(self):
        inputs = (
            torch.randn(16, 64, dtype=torch.bfloat16),
            torch.randn(64, 128, dtype=torch.bfloat16),
        )
        gm = make_fx(_BackwardCast(True))(*inputs)
        _tag_graph(gm, ["unrelated"], backward=True)
        benchmark_region = mock.Mock(return_value=CompileTimeBenchmarkResult(2.0, 1.0))

        clear_compile_time_benchmark_cache()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            result = _apply_benchmarked_coda_pattern(
                gm,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=False,
                benchmark_region=benchmark_region,
            )

        benchmark_region.assert_called_once()
        self.assertIs(result, gm)
        self.assertEqual(
            _flex_gemm_patterns(result),
            ["B_linear_dw_bf16_to_fp32"],
        )

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

    def test_backward_residual_benchmark_does_not_claim_fp32_gemm(self):
        inputs = (
            torch.randn(16, 64),
            torch.randn(64, 32),
            torch.randn(16, 32),
        )
        gm = make_fx(_BackwardAccumulate())(*inputs)
        _tag_graph(gm, ["layers.0.moe.router.gate"], backward=True)
        candidates = coda_passes._prepare_coda_candidates(
            gm,
            _CODA_PATTERNS["B_mm_dx_residual_add"],
            autotune=False,
        )

        self.assertEqual(candidates, [])

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

    def test_situ_backward_requires_exact_topology(self):
        def unrelated_sigmoid_tanh(x, weight):
            output = x @ weight
            return torch.sigmoid(output) + torch.tanh(output)

        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(unrelated_sigmoid_tanh)(*inputs)
        reference = gm(*inputs)
        for node in gm.graph.nodes:
            node.meta["autograd_backward"] = True

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["B_k3_situ_backward_activation"],
            compile_time_benchmark=False,
        )

        self.assertEqual(gm(*inputs), reference)
        self.assertEqual(
            gm.meta["coda_pattern_counts"]["B_k3_situ_backward_activation"],
            0,
        )
        self.assertEqual(_flex_gemm_patterns(gm), [])

    def test_situ_backward_rejects_an_extra_branch_output(self):
        def situ_backward_with_extra_user(
            grad,
            weight,
            saved_tanh,
            saved_sigmoid,
        ):
            branch_grad = (grad @ weight).float()
            tanh_grad = aten.tanh_backward.default(branch_grad, saved_tanh)
            sigmoid_grad = aten.sigmoid_backward.default(branch_grad, saved_sigmoid)
            return (
                tanh_grad.bfloat16(),
                sigmoid_grad.bfloat16(),
                branch_grad.square(),
            )

        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
            torch.tanh(torch.randn(8, 12)),
            torch.sigmoid(torch.randn(8, 12)),
        )
        gm = make_fx(situ_backward_with_extra_user)(*inputs)
        for node in gm.graph.nodes:
            node.meta["autograd_backward"] = True

        gm = coda_flex_gemm_pass(
            gm,
            inputs,
            patterns=["B_k3_situ_backward_activation"],
            compile_time_benchmark=False,
        )

        self.assertEqual(
            gm.meta["coda_pattern_counts"]["B_k3_situ_backward_activation"],
            0,
        )
        self.assertEqual(_flex_gemm_patterns(gm), [])

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

    def test_pattern_registrations_are_self_describing(self):
        patterns = tuple(coda_passes._CODA_PATTERNS.values())
        self.assertEqual(len({pattern.priority for pattern in patterns}), len(patterns))
        for pattern in patterns:
            self.assertEqual(pattern.name, pattern.matcher.__name__)
            self.assertIsInstance(pattern.search, PatternExpr)
            self.assertTrue(pattern.kernels)
            self.assertIn("Input::", pattern.matcher.__doc__)
            self.assertIn("Output::", pattern.matcher.__doc__)
            self.assertIn("Sites:", pattern.matcher.__doc__)

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
        nodes_to_remove: set[torch.fx.Node] = set()

        first = coda_passes._rewrite_matched_epilogue_as_flex_gemm(
            gm,
            root=root,
            body_nodes={root, silu},
            nodes_to_remove=nodes_to_remove,
            pattern="F_k3_mla_output_gate",
        )
        second = coda_passes._rewrite_matched_epilogue_as_flex_gemm(
            gm,
            root=root,
            body_nodes={root, sigmoid},
            nodes_to_remove=nodes_to_remove,
            pattern="F_router_sigmoid_bias",
        )

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertIn(root, nodes_to_remove)
        self.assertNotIn("coda_owner", root.meta)

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

    def test_compile_time_benchmark_accepts_only_faster_rewrite(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )

        def make_graph():
            gm = make_fx(_BackwardCast(False))(*inputs)
            _tag_graph(gm, ["layers.0.attention.wo"], backward=True)
            return gm

        def accept_region(_baseline, _baseline_inputs, candidate, _candidate_inputs):
            fused = next(
                node for node in candidate.graph.nodes if node.target is flex_gemm_hop
            )
            self.assertEqual(
                fused.meta["custom"]["compile_with_inductor"]["inductor_region"],
                "coda_flex_gemm_0",
            )
            return CompileTimeBenchmarkResult(2.0, 1.0)

        clear_compile_time_benchmark_cache()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            accepted = _apply_benchmarked_coda_pattern(
                make_graph(),
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=accept_region,
            )
        self.assertEqual(len(_flex_gemm_patterns(accepted)), 1)

        clear_compile_time_benchmark_cache()
        original = make_graph()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            rejected = _apply_benchmarked_coda_pattern(
                original,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=lambda *args: CompileTimeBenchmarkResult(1.0, 2.0),
            )
        self.assertIs(rejected, original)

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
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            result = _apply_benchmarked_coda_pattern(
                gm,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=benchmark_region,
            )

        self.assertEqual(benchmark_region.call_count, 2)
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
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            result = _apply_benchmarked_coda_pattern(
                gm,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=benchmark_region,
            )

        benchmark_region.assert_called_once()
        self.assertEqual(
            result.meta["coda_pattern_counts"]["B_linear_dw_bf16_to_fp32"], 2
        )

    def test_compile_time_benchmark_propagates_invalid_rewrite(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        original = make_fx(_BackwardCast(False))(*inputs)
        _tag_graph(original, ["layers.0.attention.wo"], backward=True)
        benchmark_region = mock.Mock()

        def failing_prepare(
            gm,
            pattern,
            *,
            autotune=True,
        ):
            del gm, pattern, autotune
            raise AssertionError("invalid rewrite")

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.object(
                coda_passes, "_prepare_coda_candidates", side_effect=failing_prepare
            ),
            self.assertRaisesRegex(AssertionError, "invalid rewrite"),
        ):
            _apply_benchmarked_coda_pattern(
                original,
                _CODA_PATTERNS["B_linear_dw_bf16_to_fp32"],
                coda_autotune=True,
                benchmark_region=benchmark_region,
            )

        benchmark_region.assert_not_called()

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
            """Compile-time benchmark results for B_linear_dw_bf16_to_fp32: candidates=2, applied=1, rejected=1 (slower=1, failed=0)
  APPLIED (1):
    candidate _to_copy: region 0: baseline=2000.0 us, candidate=1000.0 us, speedup=2.000x, cache=miss
  REJECTED (1):
    candidate _to_copy_1: region 0: baseline=1000.0 us, candidate=2000.0 us, speedup=0.500x, cache=miss; candidate was not faster for every changed region""",  # noqa: B950
        )
        trace.assert_called_once()
        self.assertEqual(trace.call_args.args, ("artifact",))
        self.assertEqual(
            trace.call_args.kwargs["metadata_fn"](),
            {
                "name": "compile_time_benchmark_B_linear_dw_bf16_to_fp32",
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

    def test_direct_rewrite_mutates_original(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        original = make_fx(_BackwardCast(False))(*inputs)
        _tag_graph(original, ["layers.0.attention.wo"], backward=True)

        def matcher(
            gm,
            candidates,
            counts,
            nodes_to_remove,
            prepared_candidates=None,
            autotune=True,
        ):
            del nodes_to_remove, prepared_candidates, autotune
            self.assertEqual(len(candidates), 1)
            output = next(node for node in gm.graph.nodes if node.op == "output")
            with gm.graph.inserting_before(output):
                gm.graph.call_function(aten.neg.default, (output.args[0],))
            counts["test_pattern"] += 1

        pattern = coda_passes.CodaPattern(
            name="test_pattern",
            matcher=matcher,
            priority=0,
            search=coda_passes.CallFunction(
                aten._to_copy.default,
                coda_passes.KeywordArg("input"),
                dtype=torch.float32,
                _users=coda_passes.MULTIPLE,
            ),
            kernels={"main": coda_passes.CodaKernel()},
        )
        result = coda_passes._apply_coda_pattern(original, pattern)

        self.assertIs(result, original)
        self.assertEqual(result.meta["coda_pattern_counts"]["test_pattern"], 1)

    def test_local_benchmark_region_does_not_share_mutable_metadata(self):
        gm = make_fx(lambda x: x.neg())(torch.randn(8))
        neg = next(node for node in gm.graph.nodes if node.target is aten.neg.default)
        neg.meta["custom"] = {"compile_with_inductor": {"inductor_region": "original"}}

        region = coda_passes._make_local_rewrite_benchmark_region(
            gm,
            (neg,),
            "test_pattern",
            lambda _candidate, _candidate_nodes, _nodes_to_remove: None,
        )
        for extracted in (region.baseline, region.candidate):
            extracted_neg = next(
                node
                for node in extracted.graph.nodes
                if node.target is aten.neg.default
            )
            extracted_neg.meta["custom"]["compile_with_inductor"][
                "inductor_region"
            ] = "changed"

        self.assertEqual(
            neg.meta["custom"]["compile_with_inductor"]["inductor_region"],
            "original",
        )

    def test_configured_weight_grad_cast_autotunes(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_BackwardCast(False))(*inputs)
        _tag_graph(gm, ["layers.0.attention.wo"], backward=True)
        pattern = _configured_pattern_pass(
            "B_linear_dw_bf16_to_fp32",
            compile_time_benchmark=False,
            coda_autotune=True,
        )

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            gm = pattern(gm, inputs)

        fused = next(node for node in gm.graph.nodes if node.target is flex_gemm_hop)
        self.assertEqual(
            fused.args[4],
            {"backend": "QUACK", "tuned": True},
        )

    def test_configured_pass_uses_generic_compile_time_benchmark(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_BackwardCast(False))(*inputs)
        _tag_graph(gm, ["layers.0.attention.wo"], backward=True)
        process_graph = mock.Mock()
        pattern = _configured_pattern_pass(
            "B_linear_dw_bf16_to_fp32",
            compile_time_benchmark=True,
            coda_autotune=False,
            benchmark_graph_processor=process_graph,
        )

        def apply_benchmarked(*args, **kwargs):
            del args, kwargs
            gm.meta["coda_pattern_counts"] = {
                "B_linear_dw_bf16_to_fp32": 1,
            }
            return gm

        with (
            mock.patch.object(
                coda_passes,
                "apply_benchmarked_rewrites",
                side_effect=apply_benchmarked,
            ) as benchmark,
            mock.patch.object(coda_passes.logger, "info") as log,
        ):
            result = pattern(gm, inputs)

        self.assertIs(result, gm)
        self.assertEqual(benchmark.call_args.args, (gm,))
        self.assertEqual(
            benchmark.call_args.kwargs["name"],
            "B_linear_dw_bf16_to_fp32",
        )
        self.assertEqual(
            benchmark.call_args.kwargs["cache_key"],
            ("coda_flex_gemm", False, process_graph),
        )
        self.assertIs(
            benchmark.call_args.kwargs["process_baseline"],
            process_graph,
        )
        self.assertIs(
            benchmark.call_args.kwargs["process_candidate"],
            process_graph,
        )
        log.assert_any_call("CODA B_linear_dw_bf16_to_fp32 applied 1 groups")

    def test_benchmark_graph_processing_defaults_to_eager(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_BackwardCast(False))(*inputs)
        _tag_graph(gm, ["layers.0.attention.wo"], backward=True)
        pattern = _configured_pattern_pass(
            "B_linear_dw_bf16_to_fp32",
            compile_time_benchmark=True,
            coda_autotune=False,
        )

        with mock.patch.object(
            coda_passes,
            "apply_benchmarked_rewrites",
            return_value=gm,
        ) as benchmark:
            pattern(gm, inputs)

        self.assertIsNone(benchmark.call_args.kwargs["process_baseline"])
        self.assertIsNone(benchmark.call_args.kwargs["process_candidate"])
        self.assertEqual(benchmark.call_args.kwargs["name"], pattern.__name__)

    def test_configured_pass_autotunes_without_pinned_config(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
            torch.randn(8, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_SigmoidEpilogue(False))(*inputs)
        pattern = _configured_pattern_pass(
            "F_k3_mla_output_gate",
            compile_time_benchmark=False,
            coda_autotune=True,
        )

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

    def test_configured_pass_autotune_overrides_pinned_config(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
            torch.randn(8, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_SigmoidEpilogue(True))(*inputs)
        _tag_graph(gm, ["layers.0.moe.router.gate"])
        pattern = _configured_pattern_pass(
            "F_router_sigmoid_bias",
            compile_time_benchmark=False,
            coda_autotune=True,
        )

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

    def test_shape_specific_config_overrides_device_default(self):
        root = mock.Mock()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch.object(
                coda_passes,
                "_gemm_shape",
                return_value=(8192, 2048, 10944),
            ),
        ):
            options = coda_passes._kernel_options(
                "B_swiglu_backward_activation",
                root=root,
            )

        self.assertEqual(options["config"]["tile_m"], 256)
        self.assertEqual(options["config"]["tile_n"], 224)
        self.assertTrue(options["config"]["is_dynamic_persistent"])
        self.assertFalse(options["tuned"])

    def test_multi_gemm_configs_are_selected_by_kernel_name(self):
        root = mock.Mock()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch.object(
                coda_passes,
                "_gemm_shape",
                return_value=(8192, 2048, 10944),
            ),
        ):
            gate = coda_passes._kernel_options("F_swiglu", kernel="gate", root=root)
            up = coda_passes._kernel_options("F_swiglu", kernel="up", root=root)

        self.assertEqual(gate["config"]["cluster_n"], 2)
        self.assertEqual(up["config"]["cluster_n"], 1)

    def test_shape_specific_config_falls_back_to_device_default(self):
        root = mock.Mock()
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.get_device_capability", return_value=(10, 0)),
            mock.patch.object(
                coda_passes,
                "_gemm_shape",
                return_value=(8, 16, 12),
            ),
        ):
            options = coda_passes._kernel_options(
                "B_swiglu_backward_activation",
                root=root,
            )

        self.assertEqual(options["config"]["tile_m"], 128)
        self.assertEqual(options["config"]["tile_n"], 128)
        self.assertFalse(options["tuned"])

    def test_configured_pass_does_not_autotune_by_default(self):
        inputs = (
            torch.randn(8, 16, dtype=torch.bfloat16),
            torch.randn(16, 12, dtype=torch.bfloat16),
            torch.randn(8, 12, dtype=torch.bfloat16),
        )
        gm = make_fx(_SigmoidEpilogue(False))(*inputs)
        pattern = _configured_pattern_pass(
            "F_k3_mla_output_gate",
            compile_time_benchmark=False,
        )
        gm = pattern(gm, inputs)

        fused = next(node for node in gm.graph.nodes if node.target is flex_gemm_hop)
        self.assertEqual(
            fused.args[4],
            {"backend": "QUACK", "tuned": False, "fast_math": True},
        )

    def test_selects_patterns_in_registry_order(self):
        requested = [
            "B_reshape_bf16_to_fp32",
            "F_router_sigmoid_bias",
            "F_situ",
            "F_situ",
        ]
        configured = []

        def configure(pattern, **_kwargs):
            configured.append(pattern.name)
            return lambda gm, _inputs: gm

        gm = mock.Mock()
        with mock.patch.object(
            coda_passes, "_configured_coda_pass", side_effect=configure
        ):
            coda_flex_gemm_pass(gm, patterns=requested)

        self.assertEqual(
            configured,
            [
                "F_situ",
                "F_router_sigmoid_bias",
                "B_reshape_bf16_to_fp32",
            ],
        )

    def test_empty_pattern_list_disables_coda(self):
        gm = mock.Mock()
        with mock.patch.object(coda_passes, "_configured_coda_pass") as configure:
            coda_flex_gemm_pass(gm, patterns=[])
        configure.assert_not_called()

    def test_rejects_unknown_patterns(self):
        gm = mock.Mock()
        with self.assertRaisesRegex(ValueError, "Unknown.*not_a_pattern"):
            coda_flex_gemm_pass(gm, patterns=["not_a_pattern"])
        with self.assertRaisesRegex(ValueError, "Unknown.*f3-dense"):
            coda_flex_gemm_pass(gm, patterns=["f3-dense"])


if __name__ == "__main__":
    run_tests()
