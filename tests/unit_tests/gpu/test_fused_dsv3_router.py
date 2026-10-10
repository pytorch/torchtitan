# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_remat as remat
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.decoder_sharding import dense_sequence_parallel_placement
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan.models.deepseek_v3.moe import DeepSeekV3Router
from torchtitan_recipes.overrides.fused_dsv3_router import (
    fused_dsv3_router,
    FusedDSv3RouterFunction,
    kernels,
)


def _config(dim=256):
    return DeepSeekV3Router.Config(
        num_experts=256,
        gate=HiMidLoLinear.Config(
            in_features=dim, out_features=256, backward_mode="hi_mid_lo"
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
    frequencies = F.normalize(counts.float(), p=1, dim=0) * 256
    raw_sum = (frequencies * F.normalize(scores, p=1, dim=-1).sum(0)).sum()
    return weights, ids, routing_map, raw_sum, counts


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3),
    "GB300 specialization",
)
class TestFusedDSv3RouterGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kernels.prepare()

    def setUp(self):
        torch.manual_seed(7)

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
        with set_current_spmd_mesh(mesh), typecheck(local=False):
            spmd.assert_type(logits, dense_sequence_parallel_placement())
            spmd.assert_type(bias, spmd.R)
            outputs = FusedDSv3RouterFunction.apply(logits, bias)
            for output in outputs[:3]:
                spmd.assert_type(output, dense_sequence_parallel_placement())
            spmd.assert_type(outputs[4], {"dp": spmd.P, "cp": spmd.P, "tp": spmd.P})

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

    def test_bitwise_values_decisions_and_all_gradient_paths(self):
        router = _config().build()
        for case in ("normal", "no_bias", "ties", "near_ties", "saturated"):
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
                expected = _reference(router, native, bias)
                actual = FusedDSv3RouterFunction.apply(fused, bias)
                for a, b in zip(expected, actual):
                    self._assert_bits_equal(a, b)
                grad_weights = torch.randn_like(expected[0])
                grad_raw_sum = torch.tensor(0.001 / 4096, device="cuda")
                for indices in ((0,), (3,), (0, 3)):
                    grad_outputs = tuple(
                        grad_weights if i == 0 else grad_raw_sum for i in indices
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
        actual = FusedDSv3RouterFunction.apply(logits, bias)
        for grad in (
            torch.randn(8, 4096, device="cuda").T,
            torch.ones(1, device="cuda").expand(4096, 8),
            torch._neg_view(torch.randn(4096, 8, device="cuda")),
        ):
            (a,) = torch.autograd.grad(expected[0], logits, grad, retain_graph=True)
            (b,) = torch.autograd.grad(actual[0], logits, grad, retain_graph=True)
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
        fused = FusedDSv3RouterFunction.apply
        compiled = torch.compile(fused, fullgraph=True)
        native = fused(logits, bias)
        actual = compiled(logits, bias)
        for a, b in zip(native, actual):
            self._assert_bits_equal(a, b)
        upstream = torch.randn_like(native[0])
        grad_raw = torch.tensor(0.001 / 4096, device="cuda")
        (grad_native,) = torch.autograd.grad(
            (native[0], native[3]), logits, (upstream, grad_raw)
        )
        (grad_actual,) = torch.autograd.grad(
            (actual[0], actual[3]), logits, (upstream, grad_raw)
        )
        self._assert_bits_equal(grad_native, grad_actual)
        del native, actual
        logits = logits.detach().clone().requires_grad_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            warm = fused(logits, bias)
            torch.autograd.grad((warm[0], warm[3]), logits, (upstream, grad_raw))
            del warm
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = fused(logits, bias)
            (gradient,) = torch.autograd.grad(
                (result[0], result[3]), logits, (upstream, grad_raw)
            )
        for _ in range(3):
            graph.replay()
            self._assert_bits_equal(grad_native, gradient)


if __name__ == "__main__":
    unittest.main()
