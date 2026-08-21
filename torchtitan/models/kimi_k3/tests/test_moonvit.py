# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT-V2 against the released reference and the shipped checkpoint keys.

The checkpoint's key list is the strongest available ground truth: it settles
questions the config cannot (whether the time embedding is learned, which
projector variant shipped) and questions the report gets wrong (whether
attention is factorized into two passes).
"""

from __future__ import annotations

import json
import pathlib
import re
import unittest

import torch
import torch.nn as nn

from torchtitan.models.kimi_k3.moonvit import (
    MoonViT,
    MoonViTConfig,
    PatchMergerMLPV2,
    sincos_1d,
    tpool_patch_merger,
)

_OFFICIAL = (
    pathlib.Path(__file__).resolve().parents[5]
    / "phase13_k3like_48b_posttrain"
    / "official_k3"
)
_CONFIG = _OFFICIAL / "config.json"
_INDEX = _OFFICIAL / "reference" / "model.safetensors.index.json"


def _tiny() -> MoonViTConfig:
    """Same structure, small extents. head_dim 24 stays divisible by 4 for
    2-D RoPE and qkv_hidden_size stays wider than hidden_size, as K3's is."""
    return MoonViTConfig(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=2,
        qkv_hidden_size=48,
        intermediate_size=64,
        patch_size=4,
        init_pos_emb_time=4,
        init_pos_emb_height=8,
        init_pos_emb_width=8,
        text_hidden_size=64,
        rope_max_grid=32,
    )


class TestAgainstOfficialConfig(unittest.TestCase):
    def test_defaults_match_the_released_vision_config(self):
        if not _CONFIG.exists():
            self.skipTest("official config not present")
        v = json.loads(_CONFIG.read_text())["vision_config"]
        c = MoonViTConfig()
        for ours, theirs in (
            ("num_hidden_layers", "vt_num_hidden_layers"),
            ("hidden_size", "vt_hidden_size"),
            ("num_attention_heads", "vt_num_attention_heads"),
            ("intermediate_size", "vt_intermediate_size"),
            ("qkv_hidden_size", "qkv_hidden_size"),
            ("patch_size", "patch_size"),
            ("text_hidden_size", "text_hidden_size"),
            ("init_pos_emb_time", "init_pos_emb_time"),
            ("init_pos_emb_height", "init_pos_emb_height"),
            ("init_pos_emb_width", "init_pos_emb_width"),
            ("projector_ln_eps", "projector_ln_eps"),
        ):
            self.assertEqual(getattr(c, ours), v[theirs], ours)
        self.assertEqual(list(c.merge_kernel_size), v["merge_kernel_size"])
        self.assertEqual(c.head_dim, 128)

    def test_encoder_size_matches_the_model_card(self):
        # the model card states MoonViT-V2 is 401M parameters
        n = MoonViT(MoonViTConfig()).encoder_num_parameters()
        self.assertAlmostEqual(n / 1e6, 401.0, delta=1.5)


class TestAgainstCheckpointKeys(unittest.TestCase):
    """Our submodule names must line up with the shipped checkpoint's, so the
    state-dict adapter is a prefix rename rather than a structural remap."""

    @classmethod
    def setUpClass(cls):
        if not _INDEX.exists():
            raise unittest.SkipTest("checkpoint index not present")
        keys = json.loads(_INDEX.read_text())["weight_map"].keys()
        cls.vision = {
            re.sub(r"\.\d+\.", ".N.", k)
            for k in keys
            if k.startswith("vision_tower.") or k.startswith("mm_projector.")
        }

    def _ours(self):
        model = MoonViT(MoonViTConfig())
        out = set()
        for name, _ in model.named_parameters():
            if name.startswith("mm_projector."):
                out.add(re.sub(r"\.\d+\.", ".N.", name))
            else:
                out.add("vision_tower." + re.sub(r"\.\d+\.", ".N.", name))
        return out

    def test_every_checkpoint_vision_key_has_a_home(self):
        self.assertEqual(
            self.vision - self._ours(),
            set(),
            "checkpoint keys we cannot load",
        )

    def test_we_invent_no_vision_parameters(self):
        self.assertEqual(
            self._ours() - self.vision,
            set(),
            "parameters with no counterpart in the checkpoint",
        )

    def test_one_attention_projection_set_per_block(self):
        """The report claims factorized spatial/temporal passes. The checkpoint
        has exactly one wqkv and one wo per block, so it does not. This is the
        assertion that would have caught the earlier factorized version -- the
        parameter COUNT would not, since one set used twice counts the same."""
        keys = json.loads(_INDEX.read_text())["weight_map"].keys()
        wqkv = [k for k in keys if k.endswith(".wqkv.weight")]
        wo = [k for k in keys if k.endswith(".wo.weight")]
        self.assertEqual(len(wqkv), 27)
        self.assertEqual(len(wo), 27)

    def test_time_embedding_is_not_a_checkpoint_parameter(self):
        """divided_FIXED: only the 2-D spatial table is learned. A learned time
        table would appear here."""
        pos = {k for k in self.vision if "pos_emb" in k}
        self.assertEqual(pos, {"vision_tower.patch_embed.pos_emb.weight"})
        model = MoonViT(_tiny())
        self.assertNotIn(
            "patch_embed.time_weight",
            dict(model.named_parameters()),
            "time embedding must be a buffer, not a parameter",
        )
        self.assertIn("patch_embed.time_weight", dict(model.named_buffers()))

    def test_projector_is_v2_post_norm(self):
        """PatchMergerMLPV2 has post_norm and no pre_norm; the v1 variant is the
        other way round."""
        proj = {k for k in self.vision if k.startswith("mm_projector.")}
        self.assertIn("mm_projector.post_norm.weight", proj)
        self.assertFalse(any("pre_norm" in k for k in proj))


class TestStructure(unittest.TestCase):
    def test_no_biases_anywhere(self):
        model = MoonViT(_tiny())
        offenders = [
            name
            for name, m in model.named_modules()
            if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None
        ]
        self.assertEqual(offenders, [])

    def test_all_norms_are_rmsnorm(self):
        # report sec 2.4: RMSNorm throughout. The projector's is RMSNorm too in
        # v2, unlike v1's LayerNorm.
        model = MoonViT(_tiny())
        for name, m in model.named_modules():
            if isinstance(m, nn.LayerNorm) and not isinstance(m, nn.RMSNorm):
                self.fail(f"{name} is a plain LayerNorm")

    def test_sincos_table_is_deterministic_and_bounded(self):
        a = sincos_1d(32, 4)
        b = sincos_1d(32, 4)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(a.shape, (4, 32))
        self.assertLessEqual(a.abs().max().item(), 1.0)


class TestForward(unittest.TestCase):
    def _model(self):
        torch.manual_seed(0)
        m = MoonViT(_tiny())
        m.init_weights()
        return m

    def test_image_forward_and_token_reduction(self):
        m = self._model()
        patches, grid = MoonViT.patchify(torch.randn(2, 3, 32, 32), 4)
        out = m(patches, grid)
        self.assertEqual(len(out), 2)
        # 8x8 patch grid -> 64 tokens -> 16 after the 2x2 merge
        for item in out:
            self.assertEqual(item.shape, (16, 64))
            self.assertTrue(torch.isfinite(item).all())

    def test_video_collapses_the_time_axis_entirely(self):
        m = self._model()
        patches, grid = MoonViT.patchify(torch.randn(1, 4, 3, 32, 32), 4)
        out = m(patches, grid)
        # mean over ALL frames, so 4 frames still yield one frame's tokens
        self.assertEqual(out[0].shape, (16, 64))

    def test_mixed_resolution_batch(self):
        """Native-resolution packing: one batch, different grids per sample."""
        m = self._model()
        a, ga = MoonViT.patchify(torch.randn(1, 3, 32, 32), 4)
        b, gb = MoonViT.patchify(torch.randn(1, 3, 16, 24), 4)
        patches = torch.cat([a, b], dim=0)
        grid = torch.cat([ga, gb], dim=0)
        out = m(patches, grid)
        self.assertEqual(out[0].shape, (16, 64))  # 8x8 -> 4x4
        self.assertEqual(out[1].shape, (6, 64))  # 4x6 -> 2x3

    def test_samples_do_not_attend_across_each_other(self):
        """Block-diagonal attention: a sample's output must not change when a
        different sample is packed alongside it."""
        m = self._model()
        a, ga = MoonViT.patchify(torch.randn(1, 3, 32, 32), 4)
        b, gb = MoonViT.patchify(torch.randn(1, 3, 16, 16), 4)
        alone = m(a, ga)[0]
        together = m(torch.cat([a, b]), torch.cat([ga, gb]))[0]
        self.assertLess(
            ((together - alone).norm() / alone.norm()).item(),
            1e-5,
            "packing leaked attention across samples",
        )

    def test_frames_interact(self):
        """One joint 3-D attention means frame 2 affects frame 1's tokens. If
        it did not, a video would just be a batch of images."""
        m = self._model()
        frames = torch.randn(1, 2, 3, 32, 32)
        p2, g2 = MoonViT.patchify(frames, 4)
        joint = m(p2, g2)[0]
        p1, g1 = MoonViT.patchify(frames[:, :1], 4)
        p1b, g1b = MoonViT.patchify(frames[:, 1:], 4)
        separate = (m(p1, g1)[0] + m(p1b, g1b)[0]) / 2
        rel = ((joint - separate).norm() / separate.norm()).item()
        self.assertGreater(rel, 1e-3, "frames did not interact")

    def test_rope_makes_position_matter_beyond_the_absolute_embedding(self):
        """2-D RoPE is applied on top of the absolute table. Zeroing the
        absolute table must still leave the tower position-sensitive."""
        m = self._model()
        with torch.no_grad():
            m.patch_embed.pos_emb.weight.zero_()
        patches, grid = MoonViT.patchify(torch.randn(1, 3, 16, 16), 4)
        base = m(patches, grid)[0]
        # swap two patch positions; with no positional signal at all the
        # merged output would be a permutation of the same values
        swapped = patches.clone()
        swapped[[0, 5]] = swapped[[5, 0]]
        other = m(swapped, grid)[0]
        self.assertGreater(
            ((other - base).norm() / base.norm()).item(), 1e-4
        )

    def test_grid_indivisible_by_merge_kernel_is_rejected(self):
        x = torch.randn(7 * 8, 32)
        grid = torch.tensor([[1, 7, 8]])
        with self.assertRaisesRegex(ValueError, "merge kernel"):
            tpool_patch_merger(x, grid)

    def test_too_many_frames_is_rejected(self):
        m = self._model()
        patches, grid = MoonViT.patchify(torch.randn(1, 4, 3, 16, 16), 4)
        grid[0, 0] = 5  # beyond init_pos_emb_time
        with self.assertRaisesRegex(ValueError, "init_pos_emb_time"):
            m(patches, grid)

    def test_spatial_merge_is_space_to_depth_not_pooling(self):
        cfg = _tiny()
        merger = PatchMergerMLPV2(cfg)
        with torch.no_grad():
            nn.init.eye_(merger.proj[0].weight)
            nn.init.normal_(merger.proj[2].weight, std=0.05)
            merger.post_norm.weight.fill_(1.0)
        a = torch.zeros(1, 4, cfg.hidden_size)
        a[0, 0] = 1.0
        b = torch.zeros(1, 4, cfg.hidden_size)
        b[0, 3] = 1.0  # same mean, different position within the 2x2
        self.assertFalse(torch.allclose(merger(a), merger(b)))

    def test_gradients_reach_the_learned_position_table(self):
        m = self._model()
        patches, grid = MoonViT.patchify(torch.randn(1, 2, 3, 32, 32), 4)
        torch.cat(m(patches, grid)).sum().backward()
        g = m.patch_embed.pos_emb.weight.grad
        self.assertIsNotNone(g)
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(g.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
