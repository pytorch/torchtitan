# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Original tests by Yidi Wu (@ydwu4), pytorch/torchtitan#5259.
# Source revision: 25dafa9a6db1e7d5327377a0e28913485e704a1f.

import itertools
import unittest

import pytest

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
    fused_dsv3_router_and_seqwise_loss,
    FusedDSv3SeqwiseLoss,
    FusedDSv3SeqwiseLossFunction,
)

from tests.unit_tests.gpu.test_fused_dsv3_router import (  # noqa: F401 -- pytest fixture discovery.
    _config,
    deterministic_algorithms,
)


def _reference(scores, routing_map):
    counts = routing_map.to(scores.dtype).sum(0)
    frequencies = F.normalize(counts, p=1, dim=0) * scores.shape[-1]
    return (frequencies * F.normalize(scores, p=1, dim=-1).sum(0)).sum()


@pytest.mark.usefixtures("deterministic_algorithms")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "tokens,experts,top_k,variant",
    [
        (31, 64, 6, "random"),
        (257, 128, 8, "random"),
        (65, 96, 8, "random"),
        (1023, 256, 8, "random"),
        (1024, 256, 8, "random"),
        (4097, 512, 8, "random"),
        (8193, 64, 6, "random"),
        (33, 129, 8, "random"),
        (33, 64, 8, "signed"),
        (17, 128, 8, "tiny"),
        (9, 64, 8, "boundary"),
        (9, 96, 6, "zero_valid"),
        (8193, 64, 6, "post_hot_seed"),
    ],
)
@pytest.mark.parametrize("unbiased", [False, True])
def test_general_sequence_loss_matches_native(
    tokens, experts, top_k, variant, unbiased
):
    """Check real row/column/CTA reductions and gradients across their boundaries."""
    torch.manual_seed(42)
    if variant == "post_hot_seed":
        torch.manual_seed(14536)
        torch.randn(4096, 256, device="cuda")
    scores = torch.randn(tokens, experts, device="cuda")
    if variant != "signed":
        scores = scores.sigmoid()
    if variant == "tiny":
        scores *= 1e-18
    elif variant == "boundary":
        scores.zero_()
        scores[:, 0] = 1e-12
    scores.requires_grad_()
    padding = torch.arange(tokens, device="cuda") % 7 == 0
    if variant == "zero_valid":
        padding.fill_(True)
    elif variant == "post_hot_seed":
        padding.fill_(False)
    ids = scores.topk(top_k, dim=-1, sorted=False).indices
    routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, ids, True)
    normalized = (
        scores / (scores.sum(-1, keepdim=True) + 1e-20)
        if unbiased
        else F.normalize(scores, p=1, dim=-1)
    )
    if unbiased:
        ids = normalized.topk(top_k, dim=-1, sorted=False).indices
        selected = torch.zeros_like(routing_map).scatter_(-1, ids, True)
    else:
        selected = routing_map
    selected = selected & ~padding[:, None]
    normalized = normalized * ~padding[:, None]
    frequencies = F.normalize(selected.float().sum(0), p=1, dim=0) * experts
    expected = (frequencies * normalized.sum(0)).sum()
    actual = FusedDSv3SeqwiseLossFunction.apply(
        scores,
        routing_map,
        torch.zeros(1, device="cuda", dtype=torch.int32),
        unbiased,
        padding,
        top_k,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.tensor(
        0.01 / 62914560 if variant == "post_hot_seed" else 1e-5, device="cuda"
    )
    expected_gradient = torch.autograd.grad(expected, scores, upstream)[0]
    actual_gradient = torch.autograd.grad(actual, scores, upstream)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.usefixtures("deterministic_algorithms")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_general_sequence_module_compiles_and_replays():
    """Exercise real general-shape injection, compilation, capture and replay."""
    from torchtitan_recipes.overrides.dsv3_mlperf import UnbiasedTopKLoadBalanceLoss

    torch.manual_seed(42)
    native = UnbiasedTopKLoadBalanceLoss.Config(coeff=0.01, top_k=6).build().cuda()
    fused = (
        FusedDSv3SeqwiseLoss.Config(coeff=0.01, top_k=6, require_fusion=True)
        .build()
        .cuda()
    )
    scores = torch.rand(65, 64, device="cuda", requires_grad=True)
    mapping = torch.zeros_like(scores, dtype=torch.bool)
    carrier = torch.rand(65, 6, device="cuda", requires_grad=True)
    denominator = torch.tensor(65.0, device="cuda")
    expected = native(scores, mapping, carrier=carrier, denominator=denominator)
    upstream = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, (scores, carrier), upstream)
    compiled = torch.compile(fused, fullgraph=True)
    actual = compiled(scores, mapping, carrier=carrier, denominator=denominator)
    actual_gradients = torch.autograd.grad(actual, (scores, carrier), upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    expected = expected.detach().clone()
    del actual, actual_gradients
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            output = compiled(scores, mapping, carrier=carrier, denominator=denominator)
            torch.autograd.grad(output, (scores, carrier), upstream)
            del output
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = compiled(
                scores, mapping, carrier=carrier, denominator=denominator
            )
            gradients = torch.autograd.grad(captured, (scores, carrier), upstream)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        for actual_grad, expected_grad in zip(
            gradients, expected_gradients, strict=True
        ):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3),
    "GB300 specialization",
)
class TestFusedDSv3SeqwiseLossGPU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        torch.use_deterministic_algorithms(True)

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
            f"max difference: {(expected - actual).abs().max().item()}"
            if expected.is_floating_point()
            else "non-floating values differ",
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

    def test_unbiased_normalized_topk_and_padding_ignore_dispatch(self):
        for padded, scale in itertools.product((False, True), (1.0, 1e-18, 1e-24)):
            with self.subTest(padded=padded, scale=scale):
                scores = torch.randn(4096, 256, device="cuda").sigmoid() * scale
                native = scores.clone().requires_grad_()
                fused = scores.clone().requires_grad_()
                dispatch_ids = (
                    torch.arange(4096 * 8, device="cuda").reshape(4096, 8) % 256
                )
                dispatch_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                    -1, dispatch_ids, True
                )
                mask = torch.arange(4096, device="cuda") % 13 == 0 if padded else None
                normalized = native / (native.sum(-1, keepdim=True) + 1e-20)
                aux_ids = normalized.topk(8, dim=-1, sorted=False).indices
                aux_map = torch.zeros_like(scores, dtype=torch.bool).scatter_(
                    -1, aux_ids, True
                )
                if mask is not None:
                    aux_map = aux_map & ~mask[:, None]
                    normalized = normalized * ~mask[:, None]
                frequencies = F.normalize(aux_map.float().sum(0), p=1, dim=0) * 256
                expected = (frequencies * normalized.sum(0)).sum()
                self.assertFalse(torch.equal(aux_map.sum(0), dispatch_map.sum(0)))
                counter = torch.zeros(1, dtype=torch.int32, device="cuda")
                actual = FusedDSv3SeqwiseLossFunction.apply(
                    fused, dispatch_map, counter, True, mask
                )
                gradient = torch.tensor(0.01 / 62914560, device="cuda")
                (a,) = torch.autograd.grad(expected, native, gradient)
                (b,) = torch.autograd.grad(actual, fused, gradient)
                self.assertGreater(a.abs().sum().item(), 0)
                self._assert_bits_equal(expected, actual)
                self._assert_bits_equal(a, b)

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

    def test_unaligned_scores_preserve_native_reduction_order(self):
        for offset in (1, 2, 3):
            for signed in (False, True):
                with self.subTest(offset=offset, signed=signed):
                    values = torch.randn(4096 * 256 + offset, device="cuda")
                    if not signed:
                        values = values.sigmoid()
                    scores = values[offset:].view(4096, 256).requires_grad_()
                    _, routing_map, counter = self._inputs()
                    expected = _reference(scores, routing_map)
                    actual = FusedDSv3SeqwiseLossFunction.apply(
                        scores, routing_map, counter
                    )
                    upstream = torch.tensor(-0.013, device="cuda")
                    (expected_grad,) = torch.autograd.grad(expected, scores, upstream)
                    (actual_grad,) = torch.autograd.grad(actual, scores, upstream)
                    self._assert_bits_equal(expected, actual)
                    self._assert_bits_equal(expected_grad, actual_grad)

    def test_router_composition_preserves_model_and_checkpoint_state(self):
        cfg = _config(7168)
        native = cfg.build().cuda()
        fused = fused_dsv3_router_and_seqwise_loss(cfg).build().cuda()
        for module in (native, fused):
            module.gate.to(torch.bfloat16)
        with torch.no_grad():
            native.gate.weight.normal_(0, 0.01)
        fused.load_state_dict(native.state_dict())
        self.assertEqual(native.state_dict().keys(), fused.state_dict().keys())
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
                gradients = torch.autograd.grad(
                    outputs[0], (x, module.gate.weight), upstream
                )
                self._assert_bits_equal(counts, module.tokens_per_expert_E)
                self._assert_bits_equal(metric, module.aux_loss.instance_acc)
                results.append((*outputs, *gradients, counts, metric))
            for expected, actual in zip(*results):
                self._assert_bits_equal(expected, actual)

    def test_router_composition_compiles_without_graph_breaks(self):
        module = fused_dsv3_router_and_seqwise_loss(_config()).build().cuda()
        module.gate.to(torch.bfloat16)
        with torch.no_grad():
            module.gate.weight.normal_(0, 0.01)
        inputs = torch.randn(
            4096, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        bias = torch.randn(256, device="cuda") * 0.025
        denominator = torch.tensor(4096.0, device="cuda")
        expected = module(inputs, bias, aux_loss_denominator=denominator)
        actual = torch.compile(module, fullgraph=True)(
            inputs, bias, aux_loss_denominator=denominator
        )
        upstream = torch.randn_like(expected[0])
        expected_gradients = torch.autograd.grad(
            expected[0], (inputs, module.gate.weight), upstream
        )
        actual_gradients = torch.autograd.grad(
            actual[0], (inputs, module.gate.weight), upstream
        )
        for a, b in zip((*expected, *expected_gradients), (*actual, *actual_gradients)):
            self._assert_bits_equal(a, b)


if __name__ == "__main__":
    unittest.main()
