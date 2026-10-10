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
from torchtitan.models.common.decoder_sharding import dense_sequence_parallel_placement
from torchtitan.models.common.moe import MicrobatchWiseLoadBalanceLoss
from torchtitan_recipes.overrides.fused_dsv3_seqwise_loss import (
    FusedDSv3SeqwiseLoss,
    FusedDSv3SeqwiseLossFunction,
)


def _reference(scores, routing_map):
    counts = routing_map.to(scores.dtype).sum(0)
    frequencies = F.normalize(counts, p=1, dim=0) * scores.shape[-1]
    return (frequencies * F.normalize(scores, p=1, dim=-1).sum(0)).sum()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3),
    "GB300 specialization",
)
class TestFusedDSv3SeqwiseLossGPU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_spmd_loss_is_dp_local_and_carrier_keeps_its_sharding(self):
        if dist.is_initialized():
            self.skipTest(
                "standalone FakePG typecheck requires an unused default group"
            )
        dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
        self.addCleanup(dist.destroy_process_group)
        mesh = init_device_mesh("cuda", (2, 1, 1), mesh_dim_names=("dp", "cp", "tp"))
        module = FusedDSv3SeqwiseLoss.Config(coeff=0.001).build().cuda()
        scores, routing_map, _ = self._inputs()
        carrier = torch.rand(4096, 8, device="cuda")
        denominator = torch.tensor(4096.0, device="cuda")
        with set_current_spmd_mesh(mesh), typecheck(local=False):
            for value in (scores, routing_map, carrier):
                spmd.assert_type(value, dense_sequence_parallel_placement())
            for buffer in module.buffers():
                spmd.assert_type(buffer, spmd.R)
            spmd.assert_type(denominator, spmd.R)
            result = module(
                scores, routing_map, carrier=carrier, denominator=denominator
            )
            spmd.assert_type(result, dense_sequence_parallel_placement())

    def test_whole_override_compiles_without_graph_breaks(self):
        module = FusedDSv3SeqwiseLoss.Config(coeff=0.001).build().cuda()
        scores, routing_map, _ = self._inputs()
        scores.requires_grad_()
        carrier = torch.rand(4096, 8, device="cuda", requires_grad=True)
        denominator = torch.tensor(4096.0, device="cuda")
        expected = module(scores, routing_map, carrier=carrier, denominator=denominator)
        actual = torch.compile(module, fullgraph=True)(
            scores, routing_map, carrier=carrier, denominator=denominator
        )
        self._assert_bits_equal(expected, actual)
        upstream = torch.randn_like(expected)
        (grad_native,) = torch.autograd.grad(expected, scores, upstream)
        (grad_actual,) = torch.autograd.grad(actual, scores, upstream)
        self._assert_bits_equal(grad_native, grad_actual)

    def _assert_bits_equal(self, expected, actual):
        self.assertEqual(expected.dtype, actual.dtype)
        self.assertEqual(expected.shape, actual.shape)
        self.assertTrue(
            torch.equal(
                expected.contiguous().reshape(-1).view(torch.uint8),
                actual.contiguous().reshape(-1).view(torch.uint8),
            ),
            f"max difference: {(expected - actual).abs().max().item()}",
        )

    def _inputs(self):
        scores = torch.rand(4096, 256, device="cuda")
        ids = torch.rand_like(scores).topk(8, dim=-1, sorted=False).indices
        routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, ids, True)
        counter = torch.zeros(1, dtype=torch.int32, device="cuda")
        return scores, routing_map, counter

    def test_bitwise_forward_and_backward(self):
        for case in (
            "positive",
            "signed",
            "zero",
            "tiny",
            "sparse_map",
            "all_map",
            "empty_map",
            "epsilon",
            "unaligned_gradient",
        ):
            with self.subTest(case=case):
                scores, routing_map, counter = self._inputs()
                if case == "signed":
                    scores = torch.randn_like(scores)
                elif case == "zero":
                    scores.zero_()
                elif case == "tiny":
                    scores.mul_(1e-15)
                elif case == "epsilon":
                    scores.zero_()
                    scores[:, 0] = 1e-12
                elif case == "sparse_map":
                    routing_map[::7].zero_()
                    routing_map[3, 17] = True
                elif case == "all_map":
                    routing_map.fill_(True)
                elif case == "empty_map":
                    routing_map.zero_()
                native = scores.clone().requires_grad_()
                fused = scores.clone().requires_grad_()
                expected = _reference(native, routing_map)
                actual = FusedDSv3SeqwiseLossFunction.apply(fused, routing_map, counter)
                grad = torch.tensor(0.013, device="cuda")
                if case == "unaligned_gradient":
                    grad = torch.full((2,), 0.013, device="cuda")[1]
                (expected_grad,) = torch.autograd.grad(expected, native, grad)
                (actual_grad,) = torch.autograd.grad(actual, fused, grad)
                self._assert_bits_equal(expected, actual)
                self._assert_bits_equal(expected_grad, actual_grad)

    def test_module_injection_and_checkpoint_metric(self):
        scores, routing_map, _ = self._inputs()
        carrier = torch.rand(4096, 8, device="cuda")
        for denominator in (0.0, 3.75, 4096.0):
            for checkpoint in (False, True):
                results = []
                for cls in (MicrobatchWiseLoadBalanceLoss, FusedDSv3SeqwiseLoss):
                    module = cls.Config(coeff=0.001).build().cuda()
                    x = scores.clone().requires_grad_()
                    c = carrier.clone().requires_grad_()
                    denom = torch.tensor(denominator, device="cuda")

                    def run(x, c):
                        return module(x, routing_map, carrier=c, denominator=denom)

                    output = remat.checkpoint()(run)(x, c) if checkpoint else run(x, c)
                    metric_before = module.instance_acc.clone()
                    grads = torch.autograd.grad(output, (x, c), torch.ones_like(output))
                    self._assert_bits_equal(metric_before, module.instance_acc)
                    results.append((output, *grads, module.instance_acc.clone()))
                for expected, actual in zip(*results):
                    self._assert_bits_equal(expected, actual)

    def test_fullgraph_compile_and_cuda_graph(self):
        scores, routing_map, counter = self._inputs()
        scores.requires_grad_()
        fused = FusedDSv3SeqwiseLossFunction.apply
        compiled = torch.compile(fused, fullgraph=True)
        expected = fused(scores, routing_map, counter)
        actual = compiled(scores, routing_map, counter)
        self._assert_bits_equal(expected, actual)
        (expected_grad,) = torch.autograd.grad(expected, scores)
        (actual_grad,) = torch.autograd.grad(actual, scores)
        self._assert_bits_equal(expected_grad, actual_grad)
        del expected, actual
        scores = scores.detach().clone().requires_grad_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fused(scores, routing_map, counter).backward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = fused(scores, routing_map, counter)
            (gradient,) = torch.autograd.grad(result, scores)
        for _ in range(3):
            graph.replay()
            self._assert_bits_equal(_reference(scores, routing_map), result)
            self._assert_bits_equal(expected_grad, gradient)

    def test_independent_streams_have_independent_counters(self):
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        cases = [self._inputs() for _ in streams]
        expected = [_reference(x, m) for x, m, _ in cases]
        outputs = []
        for stream, (x, m, counter) in zip(streams, cases):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs.append(FusedDSv3SeqwiseLossFunction.apply(x, m, counter))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for native, fused in zip(expected, outputs):
            self._assert_bits_equal(native, fused)

    def test_counter_wrap_preserves_loss(self):
        torch.manual_seed(42)
        scores, routing_map, counter = self._inputs()
        counter.fill_(2**31 - 64)
        expected = _reference(scores, routing_map)
        for _ in range(3):
            actual = FusedDSv3SeqwiseLossFunction.apply(scores, routing_map, counter)
            self._assert_bits_equal(expected, actual)


if __name__ == "__main__":
    unittest.main()
