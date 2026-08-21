# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The 2p8t flavor must equal Kimi K3's official config.json, field by field.

The artifact is stored at
``phase13_k3like_48b_posttrain/official_k3/config.json``; this test reads it and
compares rather than hardcoding a second copy of the numbers, so a stale flavor
cannot pass by agreeing with a stale expectation.
"""

import json
import pathlib
import unittest

from torchtitan.models.kimi_k3.model_configs import (
    attn_res_block_size,
    build_kimi_linear_config,
    resolve_num_blocks,
)

# tests/ -> kimi_k3 -> experiments -> torchtitan -> <submodule> -> <logbook>
_ARTIFACT = (
    pathlib.Path(__file__).resolve().parents[5]
    / "phase13_k3like_48b_posttrain"
    / "official_k3"
    / "config.json"
)


def _official():
    if not _ARTIFACT.exists():
        raise unittest.SkipTest(f"official artifact not present: {_ARTIFACT}")
    return json.loads(_ARTIFACT.read_text())["text_config"]


class TestK3OfficialConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.off = _official()
        cls.ours = build_kimi_linear_config("2p8t")

    def test_scalar_fields_match(self):
        direct = [
            "num_hidden_layers", "hidden_size", "num_attention_heads",
            "num_key_value_heads", "intermediate_size", "q_lora_rank",
            "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "num_experts", "num_experts_per_token",
            "num_shared_experts", "moe_intermediate_size",
            "routed_expert_hidden_size", "vocab_size",
            "max_position_embeddings", "hidden_act", "rms_norm_eps",
            "first_k_dense_replace", "tie_word_embeddings",
            "latent_moe_use_norm", "moe_renormalize", "routed_scaling_factor",
            "moe_layer_freq", "num_expert_group", "topk_group",
            "activation_situ_beta", "activation_situ_linear_beta",
        ]
        mismatch = {
            f: (getattr(self.ours, f), self.off[f])
            for f in direct
            if f in self.off and getattr(self.ours, f) != self.off[f]
        }
        self.assertEqual(mismatch, {}, f"fields differ from official: {mismatch}")

    def test_router_activation(self):
        self.assertEqual(
            self.ours.moe_router_activation_func,
            self.off["moe_router_activation_func"],
        )

    def test_layer_pattern_matches_including_the_double_global_tail(self):
        lac = self.off["linear_attn_config"]
        self.assertEqual(self.ours.full_attn_layers, lac["full_attn_layers"])
        self.assertEqual(self.ours.kda_layers, lac["kda_layers"])
        # the property that makes the tail special: 92 AND 93 are both global
        self.assertIn(92, self.ours.full_attn_layers)
        self.assertIn(93, self.ours.full_attn_layers)
        self.assertEqual(len(self.ours.full_attn_layers), 24)
        self.assertEqual(len(self.ours.kda_layers), 69)

    def test_kda_config_matches(self):
        lac = self.off["linear_attn_config"]
        self.assertEqual(self.ours.kda_head_dim, lac["head_dim"])
        self.assertEqual(self.ours.kda_num_heads, lac["num_heads"])
        self.assertEqual(
            self.ours.kda_short_conv_kernel_size, lac["short_conv_kernel_size"]
        )
        self.assertEqual(self.ours.kda_gate_lower_bound, lac["gate_lower_bound"])
        self.assertEqual(
            self.ours.kda_use_full_rank_gate, lac["use_full_rank_gate"]
        )

    def test_mla_flags_match(self):
        self.assertEqual(self.ours.mla_use_nope, self.off["mla_use_nope"])
        # our mla_gated is the config knob for the official mla_use_output_gate
        self.assertEqual(self.ours.mla_gated, self.off["mla_use_output_gate"])
        self.assertEqual(self.ours.attn_gate_param, "full_rank")

    def test_attn_res_partition_matches(self):
        self.assertEqual(attn_res_block_size("2p8t"), self.off["attn_res_block_size"])
        n = self.off["num_hidden_layers"]
        bs = self.off["attn_res_block_size"]
        self.assertEqual(resolve_num_blocks("2p8t", "block_attn_res"), -(-n // bs))
        self.assertEqual(-(-n // bs), 8)  # report sec 2.2: 8 blocks

    def test_quantization_scope_is_routed_experts_only(self):
        # not a config field of ours, but the fact the QLoRA scope must honor
        q = self.off["quantization_config"]
        self.assertEqual(q["format"], "mxfp4-pack-quantized")
        self.assertEqual(q["config_groups"]["group_0"]["weights"]["group_size"], 32)
        self.assertIsNone(q["config_groups"]["group_0"]["input_activations"])
        for pat in ("self_attn", "shared_experts", "lm_head", "vision_tower"):
            self.assertTrue(
                any(pat in ig for ig in q["ignore"]),
                f"{pat} must be in the official ignore list",
            )


if __name__ == "__main__":
    unittest.main()
