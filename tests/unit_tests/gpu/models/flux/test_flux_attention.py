# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import spmd_types as spmd
import torch
import torch.nn.functional as F
from spmd_types.checker import typecheck
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.flux.attention import FluxInnerAttention
from torchtitan.models.flux.sharding import flux_inner_attention_sharding_config


def _reference_attention(q, k, v):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        ).transpose(1, 2)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFluxVarlenAttention(unittest.TestCase):
    def test_forward_backward_and_batch_isolation(self):
        torch.manual_seed(42)
        attention = FluxInnerAttention.Config().build()
        for query_len in (17, 32):
            with self.subTest(query_len=query_len):
                inputs = [
                    torch.randn(
                        2, length, 4, 64, device="cuda", dtype=torch.bfloat16
                    ).requires_grad_()
                    for length in (query_len, 32, 32)
                ]
                reference_inputs = [x.detach().float().requires_grad_() for x in inputs]
                actual = attention(*inputs)
                expected = _reference_attention(*reference_inputs)
                torch.testing.assert_close(
                    actual.float(), expected, atol=0.01, rtol=0.01
                )
                grad_output = torch.randn_like(actual)
                grads = torch.autograd.grad(actual, inputs, grad_output)
                reference_grads = torch.autograd.grad(
                    expected, reference_inputs, grad_output.float()
                )
                for actual_grad, expected_grad in zip(grads, reference_grads):
                    torch.testing.assert_close(
                        actual_grad.float(), expected_grad, atol=0.02, rtol=0.02
                    )

                changed = [x.detach().clone().requires_grad_() for x in inputs]
                with torch.no_grad():
                    for x in changed:
                        x[1].add_(3)
                independent = attention(*changed)
                torch.testing.assert_close(actual[0], independent[0], atol=0, rtol=0)
                independent[0].float().sum().backward()
                for x in changed:
                    self.assertEqual(torch.count_nonzero(x.grad[1]).item(), 0)

    def test_compile_forward_and_backward(self):
        attention = FluxInnerAttention.Config().build()
        compiled = torch.compile(attention, fullgraph=True)
        inputs = [
            torch.randn(
                2, 32, 4, 64, device="cuda", dtype=torch.bfloat16
            ).requires_grad_()
            for _ in range(3)
        ]
        expected = attention(*inputs)
        actual = compiled(*inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        actual.float().square().mean().backward()
        for x in inputs:
            self.assertTrue(torch.isfinite(x.grad).all().item())


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestFluxVarlenContextParallel(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_gathered_kv_preserves_batch_boundaries(self):
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=2,
            tp=1,
            pp=1,
            ep=1,
            world_size=2,
            enable_sequence_parallel=False,
        )
        config = FluxInnerAttention.Config(
            sharding_config=flux_inner_attention_sharding_config()
        )
        attention = config.build()
        with parallelism_context.activate_spmd():
            attention._parallelize(parallelism_context)

        torch.manual_seed(42)
        global_inputs = [
            torch.randn(2, 32, 4, 64, device=self.device_type, dtype=torch.bfloat16)
            for _ in range(3)
        ]
        local_inputs = [
            x.chunk(self.world_size, dim=1)[self.rank].clone().requires_grad_()
            for x in global_inputs
        ]
        reference_inputs = [x.float().requires_grad_() for x in global_inputs]
        expected = _reference_attention(*reference_inputs)
        expected.square().sum().backward()

        with parallelism_context.activate_spmd(), typecheck(local=False):
            for x in local_inputs:
                spmd.assert_type(x, {"dp": spmd.S(0), "cp": spmd.S(1)})
            actual = attention(*local_inputs)
            actual.float().square().sum().backward()

        self.assertEqual(
            actual.float(),
            expected.chunk(self.world_size, dim=1)[self.rank],
            atol=0.01,
            rtol=0.01,
        )
        for actual_input, expected_input in zip(local_inputs, reference_inputs):
            self.assertEqual(
                actual_input.grad.float(),
                expected_input.grad.chunk(self.world_size, dim=1)[self.rank],
                atol=0.03,
                rtol=0.03,
            )
