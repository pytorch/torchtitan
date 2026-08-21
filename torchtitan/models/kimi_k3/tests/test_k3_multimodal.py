# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3's native vision path: MoonViT-V2 spliced into the Kimi Linear backbone."""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config
from torchtitan.models.kimi_k3.moonvit import MoonViT, MoonViTConfig
from torchtitan.models.kimi_k3.multimodal_model import (
    KimiK3MultimodalConfig,
    KimiK3MultimodalModel,
)

SENTINEL = -200


def _cfg(num_blocks=2):
    kc = build_kimi_linear_config("k3mini", vocab_size=256)
    vc = MoonViTConfig(
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=2,
        qkv_hidden_size=48,
        intermediate_size=64,
        patch_size=4,
        init_pos_emb_height=8,
        init_pos_emb_width=8,
        text_hidden_size=kc.hidden_size,
        rope_max_grid=32,
    )
    return KimiK3MultimodalConfig(
        kimi_config=kc,
        vision_config=vc,
        num_blocks=num_blocks,
        vision_token_id=SENTINEL,
    )


class TestK3MultimodalStructure(unittest.TestCase):
    def test_submodule_names_match_the_checkpoint(self):
        with torch.device("meta"):
            m = KimiK3MultimodalModel(_cfg())
        self.assertEqual(
            {n for n, _ in m.named_children()}, {"vision_tower", "language_model"}
        )
        # the projector is a tower child, as in the checkpoint, not a sibling
        self.assertIn("mm_projector", {n for n, _ in m.vision_tower.named_children()})

    def test_vision_tower_is_trainable(self):
        """Report sec 2.4 trains MoonViT-V2 from scratch jointly; the LLaVA
        stage-1 habit of freezing the tower reproduces the opposite recipe."""
        with torch.device("meta"):
            m = KimiK3MultimodalModel(_cfg())
        self.assertTrue(all(p.requires_grad for p in m.vision_tower.parameters()))

    def test_projector_width_mismatch_is_rejected(self):
        cfg = _cfg()
        cfg.vision_config.text_hidden_size += 8
        with self.assertRaisesRegex(ValueError, "hidden size"):
            with torch.device("meta"):
                KimiK3MultimodalModel(cfg)


@unittest.skipUnless(torch.cuda.is_available(), "KDA and MoE need CUDA")
class TestK3MultimodalForward(unittest.TestCase):
    def _model(self):
        torch.manual_seed(0)
        m = KimiK3MultimodalModel(_cfg()).cuda().bfloat16()
        m.init_weights(buffer_device="cuda")
        return m

    def test_text_only_path(self):
        m = self._model()
        ids = torch.randint(0, 256, (1, 128), device="cuda")
        out = m(ids)
        self.assertEqual(out.shape[:2], (1, 128))
        self.assertTrue(torch.isfinite(out).all())

    def test_image_is_spliced_and_grows_the_sequence(self):
        m = self._model()
        ids = torch.randint(0, 256, (1, 128), device="cuda")
        ids[0, 10] = SENTINEL
        patches, grid = MoonViT.patchify(
            torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.bfloat16), 4
        )
        out = m(ids, patches, grid)
        # 8x8 patch grid -> 64 tokens -> 16 after the 2x2 merge; one sentinel
        # is consumed, so the sequence grows by 15
        self.assertEqual(out.shape[1], 128 + 16 - 1)
        self.assertTrue(torch.isfinite(out).all())

    def test_vision_features_actually_reach_the_logits(self):
        """A splice that silently dropped the features would still produce the
        right shape."""
        m = self._model()
        ids = torch.randint(0, 256, (1, 128), device="cuda")
        ids[0, 10] = SENTINEL
        pixels = torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.bfloat16)
        patches, grid = MoonViT.patchify(pixels, 4)
        a = m(ids, patches, grid)
        other, _ = MoonViT.patchify(pixels * 3.0 + 1.0, 4)
        b = m(ids, other, grid)
        rel = ((a.float() - b.float()).norm() / a.float().norm()).item()
        self.assertGreater(rel, 1e-4, "changing the image did not change logits")

    def test_gradients_flow_into_the_tower(self):
        m = self._model()
        ids = torch.randint(0, 256, (1, 128), device="cuda")
        ids[0, 10] = SENTINEL
        patches, grid = MoonViT.patchify(
            torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.bfloat16), 4
        )
        m(ids, patches, grid).float().sum().backward()
        g = m.vision_tower.patch_embed.proj.weight.grad
        self.assertIsNotNone(g, "no gradient reached the patch embed")
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(g.abs().sum().item(), 0.0)

    def test_patches_without_a_sentinel_is_rejected(self):
        m = self._model()
        ids = torch.randint(0, 256, (1, 128), device="cuda")
        patches, grid = MoonViT.patchify(
            torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.bfloat16), 4
        )
        with self.assertRaisesRegex(ValueError, "no vision_token_id"):
            m(ids, patches, grid)

    def test_image_count_mismatch_is_rejected(self):
        m = self._model()
        ids = torch.randint(0, 256, (1, 128), device="cuda")
        ids[0, 10] = SENTINEL
        ids[0, 20] = SENTINEL  # two sentinels, one image
        patches, grid = MoonViT.patchify(
            torch.randn(1, 3, 32, 32, device="cuda", dtype=torch.bfloat16), 4
        )
        with self.assertRaisesRegex(ValueError, "match neither the image count"):
            m(ids, patches, grid)


class TestVisionSideStreamGating(unittest.TestCase):
    """The vision side stream must not carry an autograd graph.

    A graph recorded on it has its backward run on it, and with prefetch several
    micro-batches then accumulate into the same tower parameters from two streams
    with no ordering between them. That cost mm_full/tp2_pp2_cp2 its
    reproducibility -- seven runs, seven distinct 10-step traces -- while changing
    nothing about the result: with the stream forced off, the numbers are
    bit-identical to the DEP-without-prefetch ones. See
    NONDETERMINISM_tp2_pp2_cp2_2026-08-20.md in the logbook.
    """

    def _stream(self, grad_enabled):
        from torchtitan.models.kimi_k3.multimodal_model import KimiK3MultimodalModel

        obj = KimiK3MultimodalModel.__new__(KimiK3MultimodalModel)
        with torch.set_grad_enabled(grad_enabled):
            return KimiK3MultimodalModel._vision_stream(obj)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA for a real stream")
    def test_no_side_stream_while_recording_a_graph(self):
        self.assertIsNone(self._stream(True))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA for a real stream")
    def test_side_stream_available_under_no_grad(self):
        self.assertIsNotNone(self._stream(False))

    def test_no_side_stream_without_cuda(self):
        # Guarded first, so the grad check never has to reason about a CPU-only box.
        if torch.cuda.is_available():
            self.skipTest("CUDA present; this asserts the CPU-only path")
        self.assertIsNone(self._stream(False))


if __name__ == "__main__":
    unittest.main()
