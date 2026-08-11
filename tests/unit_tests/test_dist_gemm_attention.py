# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The dist-GEMM attention backend, at the module and config level.

Three layers, cheapest first. If you are adding a new fused module, this file is
the template:

1. ``TestDistGemmAttentionConfig`` -- no devices. Does ``gemm_backend`` select the
   fused configs, does the sharding setup declare the right contracts for them,
   and does the runtime config reach them? Catches wiring mistakes in
   milliseconds.
2. ``TestDistGemmAttentionSharding`` -- a 2-rank gloo mesh. Do those contracts
   survive a real ``parallelize``? Still no CUDA: nothing here runs the fused ops.
3. Numerics for the underlying primitives live in ``test_dist_linear.py`` (2
   GPUs), and an integration entry in ``tests/integration_tests/h100.py`` runs a
   real training step end to end; see ``dist_gemm_attention`` there.

Both classes here run in CI. Note there is no GPU unit-test job, so anything
CUDA-guarded is developer-run only.
"""

import unittest
from unittest.mock import patch

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.config_utils import make_gqa_config
from torchtitan.models.common.decoder_sharding import set_gqa_attention_sharding
from torchtitan.models.common.dist_gemm_attention import (
    AllGatherFusedQKVLinear,
    AttentionOutputLinear,
    DistGemmGQAttention,
    maybe_update_dist_gemm_config,
)

DIM = 256
N_HEADS = 8


class TestDistGemmAttentionConfig(unittest.TestCase):
    """Config-graph rewriting. No devices involved."""

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

    def test_unfused_qkv_is_rejected(self):
        """The all-gather feeds one wqkv GEMM; separate wq/wk/wv has no schedule."""
        from torchtitan.models.common.attention import FlexAttention
        from torchtitan.models.common.rope import ComplexRoPE

        with self.assertRaisesRegex(ValueError, "requires fuse_qkv=True"):
            make_gqa_config(
                dim=DIM,
                n_heads=N_HEADS,
                wqkv_param_init={},
                wo_param_init={},
                inner_attention=FlexAttention.Config(),
                rope=ComplexRoPE.Config(dim=DIM // N_HEADS, max_seq_len=128),
                fuse_qkv=False,
                gemm_backend="dist_gemm",
            )

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


class TestDistGemmAttentionSharding(DTensorTestBase):
    """The declared contracts, as they survive a real ``parallelize``.

    Contracts only -- nothing here runs the fused ops, so it needs no CUDA and
    does run in CI on a gloo mesh. Anything that actually calls symmetric memory
    belongs in a CUDA-guarded class.
    """

    @property
    def world_size(self) -> int:
        return 2

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

    @with_comms
    def test_parallelize_preserves_the_fused_contracts(self):
        """Nothing downstream of the declaration undoes it.

        The config-level assertions live in
        ``test_sharding_setup_declares_the_fused_contracts``; this pins that they
        survive ``parallelize``, which is where an earlier revision had to patch
        them back because the sharding setup ran last and overwrote them.
        """
        from torchtitan.models.llama3.config_registry import llama3_debugmodel_dist_gemm

        parallel_dims = self._parallel_dims()
        attn_cfg = llama3_debugmodel_dist_gemm().model_spec.model.layers[0].attention
        set_gqa_attention_sharding(attn_cfg, enable_sp=True)

        attn = attn_cfg.build().to(self.device_type)
        attn.parallelize(parallel_dims)

        self.assertIsNone(attn._sharding_config)
        self.assertIsNone(attn.wo._sharding_config.out_src_shardings)
        self.assertIsNone(attn.wo._sharding_config.out_dst_shardings)
        self.assertIn("weight", attn.wo._sharding_config.state_shardings)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
