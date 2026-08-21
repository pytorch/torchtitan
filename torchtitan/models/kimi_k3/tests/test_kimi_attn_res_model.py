# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU smoke tests for KimiK3AttnResModel.

Exercises the AttnRes weave end-to-end on a tiny config with dense
FFN only (no MoE, to dodge the ``torch.histc(Long)`` CPU limitation
in torchtitan's router). MoE path lives on GPU — see
``test_layers.py::TestKimiDeltaAttention::test_forward_shape_chunk_mode_cuda``
for the fla-core + CUDA side, and a follow-up adds a full GPU model
integration test once the adapter training run frees the box.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.attn_res_model import (
    KimiAttnResDecoderLayer,
    KimiK3AttnResModel,
)
from torchtitan.models.kimi_k3.model import KimiK3Config


def _dense_mla_only_config(num_hidden_layers: int = 4) -> KimiK3Config:
    """Small config: all MLA (no KDA), all dense FFN (no MoE). KDA
    requires CUDA/Triton and MoE CPU forward hits a torch.histc Long
    limitation, so both are skipped here. The AttnRes weave itself is
    independent of which attention / FFN variant sits below it.
    """
    return KimiK3Config(
        vocab_size=256,
        hidden_size=128,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=64,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        mla_use_nope=True,
        kda_num_heads=4,
        kda_head_dim=16,
        kda_short_conv_kernel_size=4,
        # No KDA layers; all layers will fall back to MLA via is_mla property.
        kda_layers=[],
        full_attn_layers=list(range(1, num_hidden_layers + 1)),
        # MoE disabled
        num_experts=None,
        num_experts_per_token=1,
        num_shared_experts=0,
        first_k_dense_replace=num_hidden_layers,  # all dense
        moe_layer_freq=1,
        num_expert_group=1,
        topk_group=1,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        initializer_range=0.02,
    )


class TestKimiAttnResDecoderLayer(unittest.TestCase):
    def test_attn_res_params_present(self):
        cfg = _dense_mla_only_config()
        layer = KimiAttnResDecoderLayer.make_config(cfg, layer_idx=0).build()
        names = {n for n, _ in layer.named_children()}
        for expected in (
            "attention_res_proj",
            "ffn_res_proj",
            "attention_res_norm",
            "ffn_res_norm",
            "input_layernorm",
            "post_attention_layernorm",
        ):
            self.assertIn(expected, names)
        # Attention and FFN are each a pair with one member None (upstream's
        # layout), and which member depends on the layer type -- named_children
        # reports only the one that exists. Asserting the pair rather than a
        # spelling keeps this test from depending on the fixture's layer type.
        for pair in (("attention", "delta_attention"), ("moe", "feed_forward")):
            self.assertEqual(
                len(names & set(pair)), 1, f"exactly one of {pair} must exist"
            )

    def test_forward_threads_blocks_and_partial(self):
        cfg = _dense_mla_only_config()
        layer = KimiAttnResDecoderLayer.make_config(cfg, layer_idx=0).build()

        B, T, D = 2, 16, cfg.hidden_size
        blocks = [torch.randn(B, T, D) for _ in range(2)]
        partial = torch.randn(B, T, D)

        new_blocks, new_partial, _ = layer(blocks, partial, is_block_start=True)
        # On block start, partial is committed into blocks -> +1 entry.
        self.assertEqual(len(new_blocks), 3)
        self.assertEqual(new_partial.shape, (B, T, D))

        # Non-block-start: blocks unchanged, partial accumulates.
        new_blocks2, new_partial2, _ = layer(
            new_blocks, new_partial, is_block_start=False
        )
        self.assertEqual(len(new_blocks2), 3)
        self.assertEqual(new_partial2.shape, (B, T, D))


class TestKimiK3AttnResModel(unittest.TestCase):
    def test_instantiate_full_attnres(self):
        """Full AttnRes: num_blocks == num_hidden_layers, one block per layer."""
        cfg = _dense_mla_only_config(num_hidden_layers=4)
        model = KimiK3AttnResModel(cfg, num_blocks=4)
        self.assertEqual(model.num_blocks, 4)
        self.assertEqual(model.layers_per_block, 1)
        # Pseudo-queries init to zero per paper.
        model.init_weights()
        for layer in model.layers.values():
            self.assertTrue(torch.all(layer.attention_res_proj.weight == 0))
            self.assertTrue(torch.all(layer.ffn_res_proj.weight == 0))
        self.assertTrue(torch.all(model.output_res_proj.weight == 0))

    def test_instantiate_block_attnres(self):
        """Block AttnRes N=2: 4 layers, 2 blocks, 2 layers per block."""
        cfg = _dense_mla_only_config(num_hidden_layers=4)
        model = KimiK3AttnResModel(cfg, num_blocks=2)
        self.assertEqual(model.num_blocks, 2)
        self.assertEqual(model.layers_per_block, 2)

    def test_forward_cpu_dense_only(self):
        """End-to-end forward on CPU with MLA + dense FFN only.

        Initial loss should be finite and roughly ``log(vocab_size)``
        because pseudo-queries are zero-init (initial AttnRes softmax
        is uniform -> equivalent to standard residuals) and the model
        is freshly initialized.
        """
        cfg = _dense_mla_only_config(num_hidden_layers=4)
        torch.manual_seed(0)
        model = KimiK3AttnResModel(cfg, num_blocks=2)
        model.init_weights()

        B, T = 2, 8
        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        logits = model(tokens)
        self.assertEqual(logits.shape, (B, T, cfg.vocab_size))
        self.assertTrue(torch.isfinite(logits).all())

        # Loss at init: log(vocab_size) for uniform output distribution.
        # With random init it won't be exactly uniform, but finite.
        import math

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, cfg.vocab_size),
            tokens.view(-1),
        )
        self.assertTrue(torch.isfinite(loss).all())
        # Sanity: CE loss should be at most a few times log(vocab_size).
        self.assertLess(loss.item(), 10 * math.log(cfg.vocab_size))

    def test_partial_final_block_is_allowed(self):
        # K3 requires this: attn_res_block_size=12 over 93 layers gives 7 full
        # blocks plus a 9-layer partial tail (report sec 2.2, "giving a partial
        # final block"). A non-divisible split must therefore be ACCEPTED, with
        # layers_per_block ceil-derived.
        cfg = _dense_mla_only_config(num_hidden_layers=5)  # prime
        model = KimiK3AttnResModel(cfg, num_blocks=2)
        self.assertEqual(model.layers_per_block, 3)  # ceil(5/2)
        # commits fire at layer 0 and 3; the 2-layer tail never commits
        self.assertEqual(model.num_committed_blocks, 2)

    def test_official_k3_partition(self):
        # the exact official shape: 93 layers, block size 12 -> 8 blocks
        n_layers, block_size = 93, 12
        num_blocks = -(-n_layers // block_size)
        self.assertEqual(num_blocks, 8)
        cfg = _dense_mla_only_config(num_hidden_layers=n_layers)
        model = KimiK3AttnResModel(cfg, num_blocks=num_blocks)
        self.assertEqual(model.layers_per_block, block_size)
        self.assertEqual(model.num_committed_blocks, 8)
        commits = [i for i in range(n_layers) if i % block_size == 0]
        self.assertEqual(commits, [0, 12, 24, 36, 48, 60, 72, 84])
        # the tail is the 9 layers after the last commit -- 84..92
        self.assertEqual(n_layers - commits[-1], 9)


def _partitioned(n_layers: int, num_blocks: int, **kwargs) -> KimiK3AttnResModel:
    """Build on meta -- these assertions read the partition, never a weight."""
    cfg = _dense_mla_only_config(num_hidden_layers=n_layers)
    with torch.device("meta"):
        return KimiK3AttnResModel(cfg, num_blocks=num_blocks, **kwargs)


class TestBlockSizePartition(unittest.TestCase):
    """K3 partitions by block SIZE: full blocks plus a short tail.

    num_blocks alone cannot express that -- see
    KimiK3AttnResModel.__init__ for why ceil(n_layers / num_blocks) is not
    invertible back to the block size.
    """

    def test_official_pair_is_unchanged_by_the_explicit_path(self):
        # 93 layers / 8 blocks: the ceil derivation already lands on the
        # released block size 12, so passing it explicitly must agree.
        for kwargs in ({}, {"layers_per_block": 12}):
            with self.subTest(**kwargs):
                model = _partitioned(n_layers=93, num_blocks=8, **kwargs)
                self.assertEqual(model.layers_per_block, 12)
                self.assertEqual(model.num_committed_blocks, 8)

    def test_size_derived_count_honors_the_block_size(self):
        # k3mini's shape: block size 12 over 21 layers is 2 blocks of 12+9.
        # Deriving from num_blocks=2 instead gives an 11+10 equal split, and
        # no num_blocks satisfies ceil(21 / n) == 12, which is why the size
        # has to be passed rather than recovered.
        model = _partitioned(n_layers=21, num_blocks=2, layers_per_block=12)
        self.assertEqual(model.layers_per_block, 12)
        self.assertEqual(model.num_committed_blocks, 2)
        commits = [i for i in range(21) if i % model.layers_per_block == 0]
        self.assertEqual(commits, [0, 12])

    def test_k3mini_flavor_wires_the_block_size_through(self):
        from torchtitan.models.kimi_k3 import model_configs as mc

        size = "k3mini"
        num_blocks = mc.resolve_num_blocks(size, "block_attn_res")
        block_size = mc.attn_res_block_size(size)
        config = mc.build_kimi_linear_config(size)
        with torch.device("meta"):
            model = KimiK3AttnResModel(
                config, num_blocks=num_blocks, layers_per_block=block_size
            )
        self.assertEqual(model.layers_per_block, block_size)
        # Round trip closes now: size 12 -> ceil(21/12) = 2 blocks -> size 12.
        self.assertEqual(
            -(-config.num_hidden_layers // model.layers_per_block), num_blocks
        )

    def test_out_of_range_block_size_is_rejected(self):
        with self.assertRaises(ValueError):
            _partitioned(n_layers=8, num_blocks=2, layers_per_block=9)


if __name__ == "__main__":
    unittest.main()
