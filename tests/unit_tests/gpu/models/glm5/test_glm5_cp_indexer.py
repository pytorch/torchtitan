# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.context_parallel import cp_all_gather_tensors
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.glm5 import build_model_config
from torchtitan.models.glm5.dsa import (
    DSAIndexerTopK,
    DsaMetadata,
    Glm5CPMLAAttention,
)
from torchtitan.models.glm5.indexer import select_local_topk
from torchtitan.models.glm5.sharding import glm5_indexer_topk_sharding_config
from torchtitan.tools.utils import set_default_dtype


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestGlm5CPIndexer(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_warmup_key_gradient_matches_summed_rank_contributions(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("requires BF16 projections")
        context = ParallelismContext(
            dp_replicate=1, dp_shard=1, cp=2, tp=1, pp=1, ep=1,
            world_size=self.world_size, enable_sequence_parallel=False,
        )
        context.build_mesh()
        device = torch.device("cuda", self.rank)
        cfg = build_model_config(
            "glm5_2_debugmodel", seq_len=128, parallelism=ParallelismConfig(),
            indexer_stage="warmup",
        ).layers[0].attention.indexer
        with set_default_dtype(torch.bfloat16):
            indexer = cfg.build(training_stage="warmup").to(device)
        indexer.init_states()
        rows = 8
        x = torch.randn(rows, cfg.wk.in_features, dtype=torch.bfloat16, device=device)
        q = torch.randn(rows, cfg.wq_b.in_features, dtype=x.dtype, device=device)
        pos = torch.arange(rows, device=device) + rows * self.rank
        # Each consumer contributes a distinct gradient for each global key.
        g = torch.arange(2 * rows, device=device)[:, None] + 1
        with context.activate_spmd():
            _, (_, keys, _) = indexer(x, q, pos, None)
            self.assertEqual(keys.dtype, torch.bfloat16)
            keys.backward((g * (1 + self.rank)).to(keys).expand_as(keys))
        actual = indexer.wk.weight.grad.detach().clone()
        indexer.zero_grad(set_to_none=True)
        _, (_, local_keys, _) = indexer(x, q, pos, None)
        summed = (g.float() + 2 * g.float())[pos]
        local_keys.backward(summed.to(local_keys).expand_as(local_keys))
        torch.testing.assert_close(actual, indexer.wk.weight.grad, rtol=0, atol=0)

    @with_comms
    def test_local_query_selection_with_real_key_gather(self):
        if torch.version.hip or torch.cuda.get_device_capability()[0] < 8:
            self.skipTest("GLM local fused selector requires NVIDIA SM80+")
        context = ParallelismContext(
            dp_replicate=1, dp_shard=1, cp=2, tp=1, pp=1, ep=1,
            world_size=self.world_size, enable_sequence_parallel=False,
        )
        context.build_mesh()
        device = torch.device("cuda", self.rank)
        positions = torch.cat((torch.arange(20), torch.arange(44))).to(device)
        padding = torch.zeros(64, device=device, dtype=torch.bool)
        padding[[19, 63]] = True
        metadata = DsaMetadata.from_positions(positions, padding)
        keys = torch.arange(64, device=device)
        # Strictly increasing key scores give an unambiguous top-k oracle.
        global_k = (keys + 1).to(torch.bfloat16)[:, None].expand(64, 16).contiguous()
        legal = (
            (metadata.documents_T[:, None] == metadata.documents_T[None, :])
            & (positions[:, None] >= positions[None, :])
            & ~padding[:, None] & ~padding[None, :]
        )
        scores = keys.float().expand(64, -1).masked_fill(~legal, -torch.inf)
        values, expected = scores.topk(8, dim=-1)
        expected = expected.to(torch.int32).masked_fill(~torch.isfinite(values), -1)
        local = slice(self.rank * 32, (self.rank + 1) * 32)
        q = torch.ones(32, 4, 16, device=device, dtype=torch.bfloat16)
        weights = torch.ones(32, 4, device=device, dtype=torch.bfloat16)
        local_k = global_k[local].contiguous()
        config = DSAIndexerTopK.Config(
            index_topk=8, sharding_config=glm5_indexer_topk_sharding_config()
        )
        selector = config.build()
        selector._parallelize(context)
        sharding = config.sharding_config
        assert sharding is not None and sharding.in_src_shardings is not None
        with context.activate_spmd(typechecking=True):
            spmd.assert_type(
                local_k, dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
            )
            (gathered_k,) = cp_all_gather_tensors((local_k,), reduce_dtype=torch.float32)
            local_metadata = Glm5CPMLAAttention.prepare_cp_metadata(
                metadata, permutation=None
            )
            local_metadata.annotate_spmd_types()
            global_metadata_type = spmd.SpmdType(
                {"dp": spmd.V, "cp": spmd.R, "tp": spmd.R},
                partition_spec=spmd.PartitionSpec("dp"),
            )
            for value in (
                local_metadata.positions_T,
                local_metadata.documents_T,
                local_metadata.valid_T,
            ):
                spmd.assert_type(value, global_metadata_type)
            self.assertEqual(local_metadata.query_indices_Q.numel(), 32)
            spmd.assert_type(
                local_metadata.query_indices_Q,
                spmd.SpmdType(
                    {"dp": spmd.R, "cp": spmd.V, "tp": spmd.R},
                    partition_spec=spmd.PartitionSpec("cp"),
                ),
            )
            for name, value in (
                ("q_QNH", q), ("k_KH", gathered_k), ("weights_QN", weights)
            ):
                spmd.assert_type(value, sharding.in_src_shardings[name])
            with patch(
                "torchtitan.models.glm5.dsa.select_local_topk", wraps=select_local_topk
            ) as kernel:
                actual = selector(q, gathered_k, weights, metadata=local_metadata)
        self.assertEqual(kernel.call_args.args[0].shape[0], 32)
        self.assertEqual(kernel.call_args.args[1].shape[0], 64)
        torch.testing.assert_close(
            actual.sort(-1).values, expected[local].sort(-1).values
        )
