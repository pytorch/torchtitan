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
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.flux.layers import _varlen_attention
from torchtitan.models.flux.model import _get_attention_metadata
from torchtitan.models.flux.sharding import flux_inner_attention_sharding_config


def _reference_attention(q, k, v):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        ).transpose(1, 2)


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
        config = VarlenInnerAttention.Config(
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
        expected[0].square().sum().backward()

        with parallelism_context.activate_spmd(), typecheck(local=False):
            for x in local_inputs:
                spmd.assert_type(x, {"dp": spmd.S(0), "cp": spmd.S(1)})
            metadata = _get_attention_metadata(
                config, 2, 16, device=local_inputs[0].device
            )
            actual = _varlen_attention(
                attention, *local_inputs, attention_metadata=metadata
            )
            actual[0].float().square().sum().backward()

        self.assertEqual(
            actual.float(),
            expected.chunk(self.world_size, dim=1)[self.rank],
            atol=0.01,
            rtol=0.01,
        )
        for actual_input, expected_input in zip(local_inputs, reference_inputs):
            assert actual_input.grad is not None
            assert expected_input.grad is not None
            self.assertEqual(
                actual_input.grad.float(),
                expected_input.grad.chunk(self.world_size, dim=1)[self.rank],
                atol=0.03,
                rtol=0.03,
            )
            self.assertEqual(torch.count_nonzero(actual_input.grad[1]).item(), 0)
