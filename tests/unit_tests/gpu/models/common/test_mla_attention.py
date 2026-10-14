# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import pytest
import spmd_types as spmd
import torch
from torch.testing._internal.common_utils import DeterministicGuard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.attention import FlexInnerAttention, MLAAttention
from torchtitan.models.common.attention.cp_mla import (
    KVAllGatherMLAFlexAttention,
    UlyssesMLAAttention,
)
from torchtitan.models.common.decoder_sharding import (
    mla_attention_local_spmd_sharding_config,
    mla_kv_projection_sharding_config,
)
from torchtitan.models.common.linear import Linear


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 4, "requires four CUDA devices")
class TestMLAAttentionSharding(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_typechecking_preserves_forward_and_backward(self):
        for backend, cp in (
            (MLAAttention, 1),
            (KVAllGatherMLAFlexAttention, 2),
            (UlyssesMLAAttention, 2),
        ):
            with self.subTest(backend=backend.__name__), DeterministicGuard(True):
                torch.manual_seed(42)
                tp = self.world_size // cp
                parallelism_context = ParallelismContext(
                    dp_replicate=1,
                    dp_shard=1,
                    cp=cp,
                    tp=tp,
                    pp=1,
                    ep=1,
                    world_size=self.world_size,
                    enable_sequence_parallel=False,
                )
                parallelism_context.build_mesh()
                config = backend.Config(
                    wkv_b=Linear.Config(
                        in_features=32,
                        out_features=128,
                        sharding_config=mla_kv_projection_sharding_config(
                            cp=spmd.R
                            if backend is KVAllGatherMLAFlexAttention
                            else spmd.S(0)
                        ),
                    ),
                    packed_kv_head_dim=32,
                    inner_attention=FlexInnerAttention.Config(),
                    sharding_config=mla_attention_local_spmd_sharding_config(),
                )
                device = torch.device("cuda", self.rank)
                attention = config.build().to(device=device, dtype=torch.bfloat16)
                attention._parallelize(parallelism_context)
                inputs = [
                    torch.randn(*shape, device=device, dtype=torch.bfloat16)
                    for shape in ((128, 4 // tp, 32), (128, 32), (128, 16))
                ]
                positions = torch.arange(128 * cp, device=device)
                metadata = config.inner_attention.build_attention_metadata(positions)
                if backend is KVAllGatherMLAFlexAttention:
                    with parallelism_context.activate_spmd():
                        metadata = backend.prepare_cp_metadata(
                            metadata, permutation=None
                        )

                def run(typechecking):
                    attention.zero_grad(set_to_none=True)
                    local_inputs = [x.detach().clone().requires_grad_() for x in inputs]
                    with parallelism_context.activate_spmd(typechecking=typechecking):
                        if typechecking:
                            for name, value in zip(
                                ("q_THK", "kv_c_normed_TL", "k_shared_TR"),
                                local_inputs,
                                strict=True,
                            ):
                                spmd.assert_type(
                                    value, config.sharding_config.in_src_shardings[name]
                                )
                        output = attention(*local_inputs, attention_metadata=metadata)
                        # Supply a local cotangent with the output's layout.
                        with spmd.no_typecheck():
                            grad_output = torch.ones_like(output)
                        if typechecking:
                            spmd.assert_type(
                                grad_output, config.sharding_config.out_src_shardings
                            )
                        output.backward(grad_output)
                    grads = [x.grad for x in (*local_inputs, *attention.parameters())]
                    self.assertTrue(all(grad is not None for grad in grads))
                    return output.detach(), tuple(
                        grad.detach().clone() for grad in grads
                    )

                expected = run(typechecking=False)
                actual = run(typechecking=True)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
