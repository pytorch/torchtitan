# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Original tests by Yidi Wu (@ydwu4), pytorch/torchtitan#5258.
# Source revision: a5d8990c5e9b46e3a87007b33de8e03c09fb97b7.

import unittest

import pytest

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch_remat as remat
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.decoder_sharding import dense_sequence_parallel_placement
from torchtitan.models.common.linear.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router
from torchtitan_recipes.overrides.fused_dsv3_router import (
    fused_dsv3_router,
    FusedDSv3RouterFunction,
)


class _MegatronSigmoid(torch.autograd.Function):
    """Reference NVIDIA's FP32 score math and per-branch BF16 logit gradient."""

    @staticmethod
    def forward(ctx, logits):
        scores = logits.float().sigmoid()
        ctx.save_for_backward(scores)
        return scores

    @staticmethod
    def backward(ctx, grad_scores):
        (scores,) = ctx.saved_tensors
        return (grad_scores * scores * (1 - scores)).bfloat16()


def _config(dim=256):
    return DeepSeekV3Router.Config(
        num_experts=256,
        gate=HiMidLoLinear.Config(
            in_features=dim,
            out_features=256,
            backward_mode="hi_mid_lo",
            inplace_wgrad_accum=False,
        ),
        score_func=Sigmoid.Config(),
        top_k=8,
        route_norm=True,
        route_scale=2.5,
        num_expert_groups=8,
        num_limited_groups=4,
        aux_loss=MicrobatchWiseLoadBalanceLoss.Config(coeff=0.001),
    )


def _reference(router, logits, bias):
    scores = logits.sigmoid()
    ids = router._select_experts(scores, bias)
    weights = scores.gather(-1, ids)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * 2.5
    routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, ids, True)
    counts = routing_map.sum(0)
    return weights, ids, routing_map, counts, scores


@pytest.fixture
def deterministic_algorithms():
    """Preserve caller state while checking the deterministic numerical contract."""
    prior = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prior)


@pytest.mark.usefixtures("deterministic_algorithms")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "tokens,experts,top_k,groups,limited,forced,strided",
    [
        (1, 64, 8, 8, 4, False, False),
        (31, 128, 4, 8, 4, False, False),
        (65, 96, 8, 8, 4, False, False),
        (4097, 256, 8, 8, 4, False, False),
        (257, 512, 8, 8, 4, False, False),
        (8193, 64, 8, 8, 4, False, False),
        (7, 12, 3, 1, 1, False, False),
        (19, 64, 6, 8, 4, False, False),
        (17, 512, 64, 8, 4, False, False),
        (65, 12, 5, 1, 1, True, True),
        (128, 512, 64, 8, 4, False, True),
        (129, 128, 8, 8, 4, False, True),
        (17, 512, 129, 1, 1, False, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_general_router_shapes_match_native(
    tokens, experts, top_k, groups, limited, forced, strided, dtype
):
    """Exercise real non-specialized forward/backward, including odd expert/token tails."""
    torch.manual_seed(42)
    config = _config(experts)
    config.num_experts = config.gate.out_features = experts
    config.top_k, config.num_expert_groups, config.num_limited_groups = (
        top_k,
        groups,
        limited,
    )
    router = config.build()
    logits = torch.randn(
        tokens, experts, device="cuda", dtype=dtype, requires_grad=True
    )
    bias = torch.randn(experts, device="cuda") * 0.025
    counter = torch.zeros(1, device="cuda", dtype=torch.int32)
    padding = torch.arange(tokens, device="cuda") % 7 == 0
    if dtype == torch.bfloat16:
        scores = _MegatronSigmoid.apply(logits)
        aux_scores = _MegatronSigmoid.apply(logits)
    else:
        scores = logits.sigmoid()
        aux_scores = scores
    ids = (
        torch.arange(tokens * top_k, device="cuda").reshape(tokens, top_k) % experts
        if forced
        else router._select_experts(scores, bias)
    )
    selected = scores.gather(-1, ids)
    weights = (selected / (selected.sum(-1, keepdim=True) + 1e-20) * 2.5).to(dtype)
    expected_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, ids, True)
    expected_counts = (expected_map & ~padding[:, None]).sum(0)
    upstream = (
        torch.randn(top_k, tokens, device="cuda", dtype=dtype).t()
        if strided
        else torch.randn_like(weights)
    )
    score_upstream = torch.randn_like(scores) * 0.001
    expected_gradient = torch.autograd.grad(
        (weights, aux_scores), logits, (upstream, score_upstream)
    )[0]
    actual = FusedDSv3RouterFunction.apply(
        logits, bias, counter, forced, padding, top_k, groups, limited
    )
    actual_gradient = torch.autograd.grad(
        (actual[0], actual[4]), logits, (upstream, score_upstream)
    )[0]
    assert torch.equal(actual[2], expected_map)
    assert torch.equal(actual[3], expected_counts)
    assert torch.equal(actual[1], ids)
    torch.testing.assert_close(actual[4], scores, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], weights, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(
        actual_gradient,
        expected_gradient,
        rtol=0,
        atol=0,
    )


@pytest.mark.usefixtures("deterministic_algorithms")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_general_router_override_compiles_and_replays():
    """Run the actual general-shape module, autograd and capture interfaces."""
    torch.manual_seed(42)
    config = _config(64)
    config.num_experts = config.gate.out_features = 64
    config.top_k = 6
    config.aux_loss = None
    native, fused = (
        config.build().cuda(),
        fused_dsv3_router(config, require_fusion=True).build().cuda(),
    )
    for module in (native, fused):
        module.gate.to(torch.bfloat16)
    with torch.no_grad():
        native.gate.weight.normal_(0, 0.01)
    fused.load_state_dict(native.state_dict())
    inputs = torch.randn(
        129, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    bias = torch.randn(64, device="cuda") * 0.025
    expected = native(inputs, bias)
    compiled = torch.compile(fused, fullgraph=True)
    actual = compiled(inputs, bias)
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    upstream = torch.randn_like(expected[0])
    expected_gradient = torch.autograd.grad(expected[0], inputs, upstream)[0]
    actual_gradient = torch.autograd.grad(actual[0], inputs, upstream)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
    # Release eager graphs before changing streams; their leaf edges retain
    # AccumulateGrad nodes bound to the eager stream even after backward.
    expected = tuple(value.detach().clone() for value in expected)
    del actual, actual_gradient
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = compiled(inputs, bias)
            torch.autograd.grad(output[0], inputs, upstream)
            del output
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = compiled(inputs, bias)
            gradient = torch.autograd.grad(captured[0], inputs, upstream)[0]
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(captured[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3),
    "GB300 specialization",
)
class TestFusedDSv3RouterGPU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        torch.use_deterministic_algorithms(True)

    def test_spmd_rows_remain_sharded_and_counts_remain_partial(self):
        if dist.is_initialized():
            self.skipTest(
                "standalone FakePG typecheck requires an unused default group"
            )
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cuda", (2, 1, 1), mesh_dim_names=("dp", "cp", "tp"))
        logits = torch.randn(4096, 256, device="cuda", requires_grad=True)
        bias = torch.randn(256, device="cuda") * 0.025
        counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        with set_current_spmd_mesh(mesh), typecheck(local=False):
            spmd.assert_type(logits, dense_sequence_parallel_placement())
            spmd.assert_type(bias, spmd.R)
            outputs = FusedDSv3RouterFunction.apply(logits, bias, counter)
            for output in (*outputs[:3], outputs[4]):
                spmd.assert_type(output, dense_sequence_parallel_placement())
            spmd.assert_type(outputs[3], {"dp": spmd.P, "cp": spmd.P, "tp": spmd.P})

    def test_whole_override_compiles_without_graph_breaks(self):
        module = fused_dsv3_router(_config()).build().cuda()
        module.gate.to(torch.bfloat16)
        with torch.no_grad():
            module.gate.weight.normal_(0, 0.01)
        inputs = torch.randn(
            4096, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        bias = torch.randn(256, device="cuda") * 0.025
        denominator = torch.tensor(4096.0, device="cuda")
        native = module(inputs, bias, aux_loss_denominator=denominator)
        actual = torch.compile(module, fullgraph=True)(
            inputs, bias, aux_loss_denominator=denominator
        )
        for expected, fused in zip(native, actual):
            self._assert_bits_equal(expected, fused)
        upstream = torch.randn_like(native[0])
        (grad_native,) = torch.autograd.grad(native[0], inputs, upstream)
        (grad_actual,) = torch.autograd.grad(actual[0], inputs, upstream)
        self._assert_bits_equal(grad_native, grad_actual)

    def _assert_bits_equal(self, expected, actual):
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(
            torch.equal(
                expected.contiguous().reshape(-1).view(torch.uint8),
                actual.contiguous().reshape(-1).view(torch.uint8),
            )
        )

    def test_bf16_megatron_boundaries_and_forced_padding_counts(self):
        router = _config().build()
        for force in (False, True):
            with self.subTest(force=force):
                logits = torch.randn(4096, 256, device="cuda", dtype=torch.bfloat16)
                bias = torch.randn(256, device="cuda") * 0.025
                mask = torch.arange(4096, device="cuda") % 13 == 0
                native = logits.clone().requires_grad_()
                fused = logits.clone().requires_grad_()
                route_scores = _MegatronSigmoid.apply(native)
                if force:
                    ids = torch.arange(4096 * 8, device="cuda").reshape(4096, 8) % 256
                else:
                    ids = router._select_experts(route_scores, bias)
                selected = route_scores.gather(-1, ids)
                weights = (
                    selected / (selected.sum(-1, keepdim=True) + 1e-20) * 2.5
                ).bfloat16()
                scores = _MegatronSigmoid.apply(native)
                routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                    -1, ids, True
                )
                counts = (routing_map & ~mask[:, None]).sum(0)
                expected = (weights, ids, routing_map, counts, scores)
                counter = torch.zeros(1, dtype=torch.int32, device="cuda")
                actual = FusedDSv3RouterFunction.apply(
                    fused, bias, counter, force, mask
                )
                for a, b in zip(expected, actual):
                    self._assert_bits_equal(a, b)
                grad_weights = torch.randn_like(weights)
                grad_scores = torch.randn_like(scores) * (0.01 / 62914560)
                for indices in ((0,), (4,), (0, 4)):
                    grads = tuple(
                        grad_weights if index == 0 else grad_scores for index in indices
                    )
                    (a,) = torch.autograd.grad(
                        tuple(expected[index] for index in indices),
                        native,
                        grads,
                        retain_graph=True,
                    )
                    (b,) = torch.autograd.grad(
                        tuple(actual[index] for index in indices),
                        fused,
                        grads,
                        retain_graph=True,
                    )
                    self._assert_bits_equal(a, b)

    def test_bitwise_values_decisions_and_all_gradient_paths(self):
        router = _config().build()
        for case in (
            "normal",
            "no_bias",
            "ties",
            "near_ties",
            "saturated",
            "unaligned",
        ):
            with self.subTest(case=case):
                logits = torch.randn(4096, 256, device="cuda")
                bias = torch.randn(256, device="cuda") * 0.025
                if case == "no_bias":
                    bias = None
                elif case == "ties":
                    logits.zero_()
                    bias.zero_()
                elif case == "near_ties":
                    logits.mul_(1e-7)
                    bias.zero_()
                elif case == "saturated":
                    logits.mul_(100)
                native = logits.clone().requires_grad_()
                fused = logits.clone().requires_grad_()
                if case == "unaligned":
                    fused = torch.empty(logits.numel() + 1, device="cuda")[1:].view_as(
                        logits
                    )
                    fused.copy_(logits).requires_grad_()
                    bias = torch.cat((bias.new_zeros(1), bias))[1:]
                expected = _reference(router, native, bias)
                counter = torch.zeros(1, dtype=torch.int32, device="cuda")
                actual = FusedDSv3RouterFunction.apply(fused, bias, counter)
                for a, b in zip(expected, actual):
                    self._assert_bits_equal(a, b)
                grad_weights = torch.randn_like(expected[0])
                grad_scores = torch.randn_like(expected[4]) * (0.001 / 4096)
                for indices in ((0,), (4,), (0, 4)):
                    grad_outputs = tuple(
                        grad_weights if i == 0 else grad_scores for i in indices
                    )
                    (grad_native,) = torch.autograd.grad(
                        tuple(expected[i] for i in indices),
                        native,
                        grad_outputs,
                        retain_graph=True,
                    )
                    (grad_fused,) = torch.autograd.grad(
                        tuple(actual[i] for i in indices),
                        fused,
                        grad_outputs,
                        retain_graph=True,
                    )
                    self._assert_bits_equal(grad_native, grad_fused)

    def test_strided_and_negative_upstream_gradients(self):
        router = _config().build()
        logits = torch.randn(4096, 256, device="cuda", requires_grad=True)
        bias = torch.randn(256, device="cuda") * 0.025
        expected = _reference(router, logits, bias)
        counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        actual = FusedDSv3RouterFunction.apply(logits, bias, counter)
        for grad in (
            torch.randn(8, 4096, device="cuda").T,
            torch.ones(1, device="cuda").expand(4096, 8),
            torch._neg_view(torch.randn(4096, 8, device="cuda")),
            torch.randn(4096 * 8 + 1, device="cuda")[1:].view(4096, 8),
        ):
            (a,) = torch.autograd.grad(expected[0], logits, grad, retain_graph=True)
            (b,) = torch.autograd.grad(actual[0], logits, grad, retain_graph=True)
            self._assert_bits_equal(a, b)

        upstream = torch.randn_like(expected[0])
        for grad_scores in (
            torch.randn(256, 4096, device="cuda").T,
            torch.ones(1, device="cuda").expand(4096, 256),
            torch._neg_view(torch.randn(4096, 256, device="cuda")),
            torch.randn(4096 * 256 + 1, device="cuda")[1:].view(4096, 256),
        ):
            for deterministic in (False, True):
                torch.use_deterministic_algorithms(deterministic)
                for indices in ((4,), (0, 4)):
                    gradients = tuple(
                        upstream if i == 0 else grad_scores for i in indices
                    )
                    (a,) = torch.autograd.grad(
                        tuple(expected[i] for i in indices),
                        logits,
                        gradients,
                        retain_graph=True,
                    )
                    (b,) = torch.autograd.grad(
                        tuple(actual[i] for i in indices),
                        logits,
                        gradients,
                        retain_graph=True,
                    )
                    self._assert_bits_equal(a, b)

    def test_native_gate_gradients_and_checkpoint_state(self):
        cfg = _config(7168)
        native = cfg.build().cuda()
        fused = fused_dsv3_router(cfg).build().cuda()
        for module in (native, fused):
            module.gate.to(torch.bfloat16)
        with torch.no_grad():
            native.gate.weight.normal_(0, 0.01)
        fused.load_state_dict(native.state_dict())
        inputs = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(256, device="cuda") * 0.025
        upstream = torch.randn(4096, 8, device="cuda")
        denominator = torch.tensor(4096.0, device="cuda")
        for checkpoint in (False, True):
            results = []
            for module in (native, fused):
                module.tokens_per_expert_E.zero_()
                module.aux_loss.instance_acc.zero_()
                x = inputs.clone().requires_grad_()

                def run(x):
                    return module(x, bias, aux_loss_denominator=denominator)

                outputs = remat.checkpoint()(run)(x) if checkpoint else run(x)
                counts = module.tokens_per_expert_E.clone()
                metric = module.aux_loss.instance_acc.clone()
                grads = torch.autograd.grad(
                    outputs[0], (x, module.gate.weight), upstream
                )
                self._assert_bits_equal(counts, module.tokens_per_expert_E)
                self._assert_bits_equal(metric, module.aux_loss.instance_acc)
                results.append((*outputs, *grads, counts, metric))
            for expected, actual in zip(*results):
                self._assert_bits_equal(expected, actual)

    def test_fullgraph_compile_and_cuda_graph(self):
        logits = torch.randn(4096, 256, device="cuda", requires_grad=True)
        bias = torch.randn(256, device="cuda") * 0.025
        counter = torch.zeros(1, dtype=torch.int32, device="cuda")

        def fused(logits, bias):
            return FusedDSv3RouterFunction.apply(logits, bias, counter)

        compiled = torch.compile(fused, fullgraph=True)
        native = fused(logits, bias)
        actual = compiled(logits, bias)
        for a, b in zip(native, actual):
            self._assert_bits_equal(a, b)
        upstream = torch.randn_like(native[0])
        grad_scores = torch.randn_like(native[4]) * (0.001 / 4096)
        (grad_native,) = torch.autograd.grad(
            (native[0], native[4]), logits, (upstream, grad_scores)
        )
        (grad_actual,) = torch.autograd.grad(
            (actual[0], actual[4]), logits, (upstream, grad_scores)
        )
        self._assert_bits_equal(grad_native, grad_actual)
        del native, actual
        logits = logits.detach().clone().requires_grad_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warm = fused(logits, bias)
            torch.autograd.grad((warm[0], warm[4]), logits, (upstream, grad_scores))
            del warm
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = fused(logits, bias)
            (gradient,) = torch.autograd.grad(
                (result[0], result[4]), logits, (upstream, grad_scores)
            )
        for _ in range(3):
            graph.replay()
            self._assert_bits_equal(grad_native, gradient)

    def test_independent_streams_and_counter_wrap(self):
        torch.manual_seed(42)
        router = _config().build()
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        cases = [
            (
                torch.randn(4096, 256, device="cuda"),
                torch.randn(256, device="cuda") * 0.025,
                torch.tensor([start], device="cuda", dtype=torch.int32),
            )
            for start in (0, 2**31 - 64)
        ]
        expected = [_reference(router, x, bias) for x, bias, _ in cases]
        for _ in range(3):
            outputs = []
            for stream, (x, bias, counter) in zip(streams, cases):
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    outputs.append(FusedDSv3RouterFunction.apply(x, bias, counter))
            for stream in streams:
                torch.cuda.current_stream().wait_stream(stream)
            for native, fused in zip(expected, outputs):
                for a, b in zip(native, fused):
                    self._assert_bits_equal(a, b)

    def test_router_without_auxiliary_loss(self):
        cfg = _config()
        cfg.aux_loss = None
        native = cfg.build().cuda()
        fused = fused_dsv3_router(cfg).build().cuda()
        for module in (native, fused):
            module.gate.to(torch.bfloat16)
        with torch.no_grad():
            native.gate.weight.normal_(0, 0.01)
        fused.load_state_dict(native.state_dict())
        inputs = torch.randn(4096, 256, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(256, device="cuda") * 0.025
        upstream = torch.randn(4096, 8, device="cuda")
        results = []
        for module in (native, fused):
            x = inputs.clone().requires_grad_()
            outputs = module(x, bias)
            gradients = torch.autograd.grad(
                outputs[0], (x, module.gate.weight), upstream
            )
            results.append((*outputs, *gradients, module.tokens_per_expert_E.clone()))
        for expected, actual in zip(*results):
            self._assert_bits_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
