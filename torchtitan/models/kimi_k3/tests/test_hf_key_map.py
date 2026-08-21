# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Can we place every tensor in the released Kimi K3 checkpoint?

Driven by the real ``model.safetensors.index.json`` (497,220 keys), because a
hand-written expectation would only test the mapping against itself. Coverage is
the judge of whether official weights can be loaded at all: an unmapped key is a
tensor that would be silently dropped, and a dropped tensor is a layer running
on init noise -- the same failure mode as the uninitialized routed experts.
"""

from __future__ import annotations

import json
import pathlib
import re
import unittest

from torchtitan.models.kimi_k3.hf_key_map import (
    official_to_titan,
    titan_to_official,
    UnmappedKey,
)

_INDEX = (
    pathlib.Path(__file__).resolve().parents[5]
    / "phase13_k3like_48b_posttrain"
    / "official_k3"
    / "reference"
    / "model.safetensors.index.json"
)

# The release: 93 layers, 24 full-attention (MLA) and 69 KDA. Checkpoint keys
# are 0-based; linear_attn_config is 1-based, so shift.
_OFFICIAL_FULL_ATTN_1BASED = [
    4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76,
    80, 84, 88, 92, 93,
]
_KDA_0BASED = {
    i for i in range(93) if (i + 1) not in _OFFICIAL_FULL_ATTN_1BASED
}


def _keys():
    if not _INDEX.exists():
        raise unittest.SkipTest("checkpoint index not present")
    return list(json.loads(_INDEX.read_text())["weight_map"].keys())


class TestOfficialKeyCoverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.keys = _keys()
        cls.patterns = sorted({re.sub(r"\.\d+\.", ".N.", k) for k in cls.keys})

    def test_every_key_maps(self):
        failures = []
        for k in self.keys:
            try:
                official_to_titan(k, kda_layers=_KDA_0BASED)
            except UnmappedKey:
                failures.append(re.sub(r"\.\d+\.", ".N.", k))
        self.assertEqual(
            sorted(set(failures)), [], "unmapped checkpoint key patterns"
        )

    def test_key_count_is_what_we_think(self):
        # a sanity anchor: if the release is re-uploaded with a different
        # layout, this fails loudly rather than the mapping quietly drifting
        self.assertEqual(len(self.keys), 497220)
        self.assertEqual(len(self.patterns), 59)

    def test_expert_weights_are_recognized_as_packed_mxfp4(self):
        kinds = set()
        for k in self.keys:
            if ".experts." in k:
                kinds.add(official_to_titan(k, kda_layers=_KDA_0BASED)[1])
        self.assertEqual(kinds, {"expert_packed", "expert_scale"})

    def test_nothing_outside_the_routed_experts_is_quantized(self):
        """The quantization scope quant_scope.py encodes, read off the actual
        checkpoint rather than off the config's ignore list."""
        packed = {
            k for k in self.keys if k.endswith((".weight_packed", ".weight_scale"))
        }
        self.assertTrue(packed)
        for k in packed:
            self.assertIn(".block_sparse_moe.experts.", k)

    def test_attention_res_keys_are_present_and_complete(self):
        """Block Attention Residuals in the shipped weights: a per-layer pair
        for all 93 layers plus one final aggregation."""
        per_layer = [k for k in self.keys if "self_attention_res_proj" in k]
        mlp_res = [k for k in self.keys if "mlp_res_proj" in k]
        self.assertEqual(len(per_layer), 93)
        self.assertEqual(len(mlp_res), 93)
        self.assertEqual(
            len([k for k in self.keys if "output_attn_res_proj" in k]), 1
        )
        # and they land on our names
        ours = official_to_titan(
            "language_model.model.layers.7.self_attention_res_proj.weight",
            kda_layers=_KDA_0BASED,
        )[0]
        self.assertEqual(ours, "layers.7.attention_res_proj.weight")

    def test_g_proj_resolves_by_layer_type(self):
        """The release calls both output gates g_proj. Ours are named
        differently per attention type, so the mapping must use the layer type;
        getting it wrong silently swaps two same-shaped tensors."""
        kda_layer = min(_KDA_0BASED)
        mla_layer = 3  # 0-based for the 1-based layer 4
        self.assertNotIn(mla_layer, _KDA_0BASED)
        kda_key = f"language_model.model.layers.{kda_layer}.self_attn.g_proj.weight"
        mla_key = f"language_model.model.layers.{mla_layer}.self_attn.g_proj.weight"
        self.assertEqual(
            official_to_titan(kda_key, kda_layers=_KDA_0BASED)[0],
            f"layers.{kda_layer}.delta_attention.g_proj.weight",
        )
        self.assertEqual(
            official_to_titan(mla_key, kda_layers=_KDA_0BASED)[0],
            f"layers.{mla_layer}.attention.attn_gate_proj.weight",
        )

    def test_router_bias_is_mapped_as_a_buffer(self):
        k = "language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias"
        ours, kind = official_to_titan(k, kda_layers=_KDA_0BASED)
        self.assertEqual(ours, "layers.1.moe._moe.expert_bias_E")
        self.assertEqual(kind, "buffer")

    def test_routed_and_shared_experts_use_different_conventions(self):
        """Same block, two naming conventions: routed experts are w1/w2/w3 and
        shared experts are gate/up/down_proj. One global rename breaks one."""
        routed, _ = official_to_titan(
            "language_model.model.layers.1.block_sparse_moe.experts.5.w1.weight_packed",
            kda_layers=_KDA_0BASED,
        )
        self.assertTrue(routed.endswith("inner_experts.w1_EFD[5]"))
        shared, _ = official_to_titan(
            "language_model.model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight",
            kda_layers=_KDA_0BASED,
        )
        self.assertEqual(shared, "layers.1.moe.shared_experts.gate_proj.weight")

    def test_dense_layer_and_moe_layers_both_land_on_ffn(self):
        dense, _ = official_to_titan(
            "language_model.model.layers.0.mlp.gate_proj.weight",
            kda_layers=_KDA_0BASED,
        )
        self.assertEqual(dense, "layers.0.feed_forward.gate_proj.weight")

    def test_vision_keys_map_onto_the_tower(self):
        for k in (
            "vision_tower.patch_embed.proj.weight",
            "vision_tower.encoder.blocks.3.wqkv.weight",
            "vision_tower.encoder.final_layernorm.weight",
            "mm_projector.post_norm.weight",
            "mm_projector.proj.0.weight",
        ):
            ours, kind = official_to_titan(k, kda_layers=_KDA_0BASED)
            self.assertEqual(kind, "vision")
            self.assertTrue(ours.startswith("vision_tower."), ours)

    def test_round_trip_for_every_non_expert_pattern(self):
        """Every mapping must be invertible, or exporting back to HF silently
        renames layers."""
        failures = []
        for k in self.keys:
            if ".experts." in k:
                continue  # stacked on our side; covered separately
            ours, kind = official_to_titan(k, kda_layers=_KDA_0BASED)
            try:
                back = titan_to_official(ours, kda_layers=_KDA_0BASED)
            except UnmappedKey as e:
                failures.append((re.sub(r"\.\d+\.", ".N.", k), f"raised {e}"))
                continue
            if back != k:
                failures.append(
                    (re.sub(r"\.\d+\.", ".N.", k), re.sub(r"\.\d+\.", ".N.", back))
                )
        self.assertEqual(sorted(set(failures)), [], "round-trip mismatches")

    def test_expert_round_trip_needs_the_expert_index(self):
        ours = (
            "layers.1.moe._moe.routed_experts.inner_experts.w2_EDF"
        )
        with self.assertRaisesRegex(UnmappedKey, "expert_idx"):
            titan_to_official(ours, kda_layers=_KDA_0BASED)
        self.assertEqual(
            titan_to_official(ours, kda_layers=_KDA_0BASED, expert_idx=17),
            "language_model.model.layers.1.block_sparse_moe.experts.17.w2.weight",
        )

    def test_unknown_key_raises_rather_than_returning_none(self):
        with self.assertRaises(UnmappedKey):
            official_to_titan("some.unexpected.key", kda_layers=_KDA_0BASED)


if __name__ == "__main__":
    unittest.main()
