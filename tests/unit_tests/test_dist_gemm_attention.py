# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The dist-GEMM attention override, at the module and config level.

Three layers, cheapest first. If you are adding a new fused module, this file is
the template:

1. ``TestDistGemmAttentionConfig`` -- CPU only. Does the override rewrite the
   config graph, and does the runtime config reach it? No GPUs, no collectives,
   so this catches wiring mistakes in milliseconds.
2. ``TestDistGemmAttentionModules`` -- 2 GPUs. Do the replacement modules produce
   the same numbers as the stock ones under TP, and do they hand back DTensors
   with the placements the parent expects? This is where sharding bugs surface.
3. An integration entry in ``tests/integration_tests/features.py`` runs a real
   training step end to end; see ``override_dist_gemm_attention`` there.
"""

import unittest
from unittest.mock import patch

import torch
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.attention import FusedQKVLinear, GQAttention
from torchtitan.models.common.decoder_sharding import rowwise_config
from torchtitan.models.common.linear import Linear
from torchtitan.overrides.dist_gemm_attention import (
    all_gather_fused_qkv,
    AllGatherFusedQKVLinear,
    attention_output_linear,
    AttentionOutputLinear,
    dist_gemm_attention,
    maybe_update_dist_gemm_config,
    walk_configs,
)

DIM = 256
N_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = 32


class TestDistGemmAttentionConfig(unittest.TestCase):
    """Config-graph rewriting. No devices involved."""

    def _qkv_cfg(self) -> FusedQKVLinear.Config:
        return FusedQKVLinear.Config(
            head_dim=HEAD_DIM,
            n_heads=N_HEADS,
            n_kv_heads=N_KV_HEADS,
            wqkv=Linear.Config(
                in_features=DIM,
                out_features=(N_KV_HEADS) * (N_HEADS // N_KV_HEADS + 2) * HEAD_DIM,
            ),
        )

    def test_qkv_override_returns_dist_gemm_config(self):
        replaced = all_gather_fused_qkv(self._qkv_cfg())
        self.assertIsInstance(replaced, AllGatherFusedQKVLinear.Config)
        # the stock parameter shape must survive, or checkpoints stop loading
        self.assertEqual(replaced.wqkv.in_features, DIM)

    def test_output_override_preserves_state_shardings(self):
        """The rewrite must not drop the parameter sharding the parent declared."""
        cfg = Linear.Config(in_features=DIM, out_features=DIM)
        replaced = attention_output_linear(cfg)
        self.assertIsInstance(replaced, AttentionOutputLinear.Config)
        self.assertIsNotNone(replaced.sharding_config)

    def test_tokens_per_rank_is_stamped_from_runtime_config(self):
        """maybe_update_dist_gemm_config carries seq_len x batch onto the modules.

        The symmetric-memory workspace is sized from this at parallelize time, so a
        None here silently degrades to lazy growth (correct, but not CUDA-graph
        safe). Worth pinning.
        """
        from torchtitan.models.llama3.config_registry import llama3_debugmodel

        trainer_cfg = llama3_debugmodel()
        trainer_cfg.training.seq_len = 512
        trainer_cfg.training.local_batch_size = 4
        model_cfg = trainer_cfg.model_spec.model
        for layer in model_cfg.layers:
            attn = getattr(layer, "attention", None)
            if isinstance(attn, GQAttention.Config):
                layer.attention = dist_gemm_attention(attn)

        ours = [
            c
            for c in walk_configs(model_cfg)
            if isinstance(
                c, (AllGatherFusedQKVLinear.Config, AttentionOutputLinear.Config)
            )
        ]
        self.assertTrue(ours, "override did not reach any attention projection")
        self.assertTrue(all(c.tokens_per_rank is None for c in ours))

        maybe_update_dist_gemm_config(model_cfg, trainer_cfg)
        self.assertTrue(all(c.tokens_per_rank == 4 * 512 for c in ours))

    def test_non_trainer_config_is_a_no_op(self):
        """Inference-only callers have no fixed token count; must not raise."""
        cfg = Linear.Config(in_features=DIM, out_features=DIM)
        replaced = attention_output_linear(cfg)
        maybe_update_dist_gemm_config(replaced, object())
        self.assertIsNone(replaced.tokens_per_rank)


@unittest.skipUnless(torch.cuda.is_available(), "symmetric memory requires CUDA")
class TestDistGemmAttentionModules(DTensorTestBase):
    """The replacement modules under a real TP mesh, against the stock modules."""

    @property
    def world_size(self) -> int:
        return 2

    TOL = 2e-2

    def _parallel_dims(self) -> ParallelDims:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=self.world_size,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        with patch(
            "torchtitan.distributed.parallel_dims.device_type", self.device_type
        ):
            parallel_dims.build_mesh()
        return parallel_dims

    def _sequence_sharded_input(self, mesh, bsz, seqlen):
        """[bsz, seqlen, DIM] sharded over the sequence, as SP hands it over."""
        torch.manual_seed(0)
        full = torch.randn(bsz, seqlen, DIM, device=self.device_type)
        local = full.chunk(self.world_size, dim=1)[self.rank].contiguous()
        return full, DTensor.from_local(local, mesh, (Shard(1),), run_check=False)

    @with_comms
    def test_parallelize_clears_wo_partial_output_contract(self):
        """The fix for the placement collision, tested where it happens.

        ``set_gqa_attention_sharding`` gives ``wo`` a ``rowwise_config()``, which
        declares out_src=Partial (a stock rowwise linear emits a partial sum and
        lets the framework reduce-scatter it). AttentionOutputLinear has already
        reduce-scattered inside the fused op, so its forward returns Shard(1) and
        the Partial declaration is wrong -- the module then fails its own out_src
        check. ``DistGemmGQAttention.parallelize`` redeclares the contract, and
        this pins that it did.

        Note the modules cannot be tested standalone: the sharding contract is
        written onto their configs by the model's sharding setup, so a bare
        ``AttentionOutputLinear.Config(...).build()`` has an unsharded weight and
        no output contract at all. Go through the attention block, as here.
        """
        from torchtitan.models.llama3.config_registry import llama3_debugmodel

        parallel_dims = self._parallel_dims()
        trainer_cfg = llama3_debugmodel()
        model_cfg = trainer_cfg.model_spec.model
        attn_cfg = next(
            dist_gemm_attention(layer.attention)
            for layer in model_cfg.layers
            if isinstance(getattr(layer, "attention", None), GQAttention.Config)
        )
        # what the model's sharding setup would write onto wo
        attn_cfg.wo.sharding_config = rowwise_config(output_sp=True)
        self.assertIsNotNone(
            attn_cfg.wo.sharding_config.out_src_shardings,
            "precondition: rowwise_config should declare a Partial output",
        )

        attn = attn_cfg.build().to(self.device_type)
        attn.parallelize(parallel_dims)

        # the Partial declaration must be gone, while the parameter sharding that
        # actually shards the weight is kept
        self.assertIsNone(attn.wo._sharding_config.out_src_shardings)
        self.assertIsNone(attn.wo._sharding_config.out_dst_shardings)
        self.assertIn("weight", attn.wo._sharding_config.state_shardings)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
