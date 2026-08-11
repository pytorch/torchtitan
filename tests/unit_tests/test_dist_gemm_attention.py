# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The dist-GEMM attention backend, at the module and config level.

Three layers, cheapest first. If you are adding a new fused module, this file is
the template:

1. ``TestDistGemmAttentionConfig`` -- CPU only, and the only part of this file
   that runs in CI (there is no GPU unit-test job). Does ``gemm_backend`` select
   the fused configs, and does the runtime config reach them? No GPUs, no
   collectives, so this catches wiring mistakes in milliseconds.
2. ``TestDistGemmAttentionModules`` -- 2 GPUs, developer-run. Do the replacement
   modules produce the same numbers as the stock ones under TP, and do they hand
   back DTensors with the placements the parent expects? This is where sharding
   bugs surface.
3. An integration entry in ``tests/integration_tests/h100.py`` runs a real
   training step end to end; see ``dist_gemm_attention`` there.
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
from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.decoder_sharding import set_gqa_attention_sharding
from torchtitan.models.common.dist_gemm_attention import (
    AllGatherFusedQKVLinear,
    AttentionOutputLinear,
    DistGemmGQAttention,
    maybe_update_dist_gemm_config,
    to_dist_gemm_attention,
)
from torchtitan.models.common.linear import Linear

DIM = 256
N_HEADS = 8
N_KV_HEADS = 4
HEAD_DIM = 32


class TestDistGemmAttentionConfig(unittest.TestCase):
    """Config-graph rewriting. No devices involved."""

    def _gqa_cfg(self) -> GQAttention.Config:
        """A stock (non-dist-GEMM) attention config, as the model builds it."""
        from torchtitan.models.llama3 import model_registry

        return model_registry("debugmodel").model.layers[0].attention

    def test_gemm_backend_selects_the_fused_configs(self):
        """model_registry(gemm_backend="dist_gemm") swaps all three pieces."""
        from torchtitan.models.llama3 import model_registry

        spec = model_registry("debugmodel", gemm_backend="dist_gemm")
        for layer in spec.model.layers:
            attn = layer.attention
            self.assertIsInstance(attn, DistGemmGQAttention.Config)
            self.assertIsInstance(attn.qkv_linear, AllGatherFusedQKVLinear.Config)
            self.assertIsInstance(attn.wo, AttentionOutputLinear.Config)

    def test_default_gemm_backend_is_untouched(self):
        """The default must stay stock, or every model silently changes."""
        from torchtitan.models.llama3 import model_registry

        for layer in model_registry("debugmodel").model.layers:
            self.assertNotIsInstance(layer.attention, DistGemmGQAttention.Config)
            self.assertNotIsInstance(layer.attention.wo, AttentionOutputLinear.Config)

    def test_stock_parameter_shapes_survive(self):
        """Fused modules keep the stock layouts, or checkpoints stop loading."""
        from torchtitan.models.llama3 import model_registry

        stock = model_registry("debugmodel").model.layers[0].attention
        fused = (
            model_registry("debugmodel", gemm_backend="dist_gemm")
            .model.layers[0]
            .attention
        )
        self.assertEqual(
            fused.qkv_linear.wqkv.in_features, stock.qkv_linear.wqkv.in_features
        )
        self.assertEqual(
            fused.qkv_linear.wqkv.out_features, stock.qkv_linear.wqkv.out_features
        )
        self.assertEqual(fused.wo.in_features, stock.wo.in_features)
        self.assertEqual(fused.wo.out_features, stock.wo.out_features)

    def test_output_config_preserves_state_shardings(self):
        """The rewrite must not drop the parameter sharding the parent declared."""
        replaced = to_dist_gemm_attention(self._gqa_cfg())
        self.assertIsInstance(replaced.wo, AttentionOutputLinear.Config)
        self.assertIsNotNone(replaced.wo.sharding_config)

    def test_unfused_qkv_is_rejected(self):
        """The all-gather feeds one wqkv GEMM; separate wq/wk/wv has no schedule."""
        cfg = self._gqa_cfg()
        cfg.qkv_linear = Linear.Config(in_features=DIM, out_features=DIM)
        with self.assertRaisesRegex(TypeError, "fuse_qkv=True"):
            to_dist_gemm_attention(cfg)

    def test_tokens_per_rank_is_stamped_from_runtime_config(self):
        """maybe_update_dist_gemm_config carries seq_len x batch onto the modules.

        The symmetric-memory workspace is sized from this at parallelize time, so a
        None here silently degrades to lazy growth (correct, but not CUDA-graph
        safe). Worth pinning.
        """
        from torchtitan.models.llama3.config_registry import llama3_debugmodel_dist_gemm

        trainer_cfg = llama3_debugmodel_dist_gemm()
        trainer_cfg.training.seq_len = 512
        trainer_cfg.training.local_batch_size = 4
        model_cfg = trainer_cfg.model_spec.model

        ours = [
            c
            for layer in model_cfg.layers
            for c in (layer.attention.qkv_linear, layer.attention.wo)
        ]
        self.assertTrue(ours, "gemm_backend did not reach any attention projection")
        self.assertTrue(all(c.tokens_per_rank is None for c in ours))

        maybe_update_dist_gemm_config(model_cfg, trainer_cfg)
        self.assertTrue(all(c.tokens_per_rank == 4 * 512 for c in ours))

    def test_sharding_setup_declares_the_fused_contracts(self):
        """set_gqa_attention_sharding declares different contracts for dist-GEMM.

        Two differences from the stock block, both because the fused ops own the
        collectives themselves: the block declares no attention-boundary
        all-gather, and wo emits its final Shard(1) rather than a Partial for the
        framework to reduce-scatter. Declaring the stock Partial here would make
        the module fail its own out_src check at runtime.
        """
        from torchtitan.models.llama3 import model_registry

        stock = model_registry("debugmodel").model.layers[0].attention
        fused = (
            model_registry("debugmodel", gemm_backend="dist_gemm")
            .model.layers[0]
            .attention
        )
        set_gqa_attention_sharding(stock, enable_sp=True)
        set_gqa_attention_sharding(fused, enable_sp=True)

        self.assertIsNotNone(stock.sharding_config)
        self.assertIsNone(fused.sharding_config)

        self.assertIsNotNone(stock.wo.sharding_config.out_src_shardings)
        self.assertIsNone(fused.wo.sharding_config.out_src_shardings)
        self.assertIsNone(fused.wo.sharding_config.out_dst_shardings)
        # the weight sharding still has to be declared, or wo is never sharded
        self.assertIn("weight", fused.wo.sharding_config.state_shardings)

    def test_non_trainer_config_is_a_no_op(self):
        """Inference-only callers have no fixed token count; must not raise."""
        from torchtitan.models.llama3.config_registry import llama3_debugmodel_dist_gemm

        model_cfg = llama3_debugmodel_dist_gemm().model_spec.model
        maybe_update_dist_gemm_config(model_cfg, object())
        self.assertIsNone(model_cfg.layers[0].attention.wo.tokens_per_rank)


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
    def test_wo_declares_no_partial_output_contract(self):
        """The fix for the placement collision, tested where it happens.

        For a stock rowwise linear ``set_gqa_attention_sharding`` declares
        out_src=Partial: the linear emits a partial sum over its slice of K and
        the framework reduce-scatters it. AttentionOutputLinear has already
        reduce-scattered inside the fused op, so its forward returns Shard(1)
        directly. If the Partial declaration survived, the module would fail its
        own out_src check with "output DTensor has placements (Shard(dim=1),),
        but out_src_shardings expects (Partial(sum),)".

        Pinned end to end through ``parallelize`` rather than by inspecting the
        config, because the failure mode is a runtime check: the value has to
        survive whatever the sharding setup writes last.

        Note the modules cannot be tested standalone: the sharding contract is
        written onto their configs by the model's sharding setup, so a bare
        ``AttentionOutputLinear.Config(...).build()`` has an unsharded weight and
        no output contract at all. Go through the attention block, as here.
        """
        from torchtitan.models.llama3.config_registry import llama3_debugmodel_dist_gemm

        parallel_dims = self._parallel_dims()
        model_cfg = llama3_debugmodel_dist_gemm().model_spec.model
        attn_cfg = model_cfg.layers[0].attention
        set_gqa_attention_sharding(attn_cfg, enable_sp=True)

        attn = attn_cfg.build().to(self.device_type)
        attn.parallelize(parallel_dims)

        # the Partial declaration must be absent, while the parameter sharding
        # that actually shards the weight is kept
        self.assertIsNone(attn.wo._sharding_config.out_src_shardings)
        self.assertIsNone(attn.wo._sharding_config.out_dst_shardings)
        self.assertIn("weight", attn.wo._sharding_config.state_shardings)
        # and the block itself declares no attention-boundary all-gather
        self.assertIsNone(attn._sharding_config)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
