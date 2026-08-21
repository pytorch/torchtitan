# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""head -> body -> tail chained must equal the single vision stage.

Report 5.2.3's "balances vision forward and backward passes across PP stages" needs the
tower to span stages. ``test_moonvit_stage_split`` pins the tower arithmetic; this pins
the STAGE layer around it -- the text embedding, the sentinel mask, the fixed-capacity
patch payload and the splice -- driven in one process so a mismatch is attributable to
this code and not to PP plumbing.

The roles are chained by hand here, exactly as PP will chain them: the head's output
tuple becomes the next stage's positional arguments. ``_dep_current_mb`` is set by hand
too, because the real value comes from a schedule patch that does not exist in a
single-process test.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config
from torchtitan.models.kimi_k3.moonvit import MoonViTConfig
from torchtitan.models.kimi_k3.multimodal_model import (
    KimiK3MultimodalConfig,
    KimiK3ViTStage,
)

SENTINEL = -200
MERGE = 2


def _cfg(num_vit_layers: int = 4):
    kc = build_kimi_linear_config("k3mini", vocab_size=256)
    vc = MoonViTConfig(
        num_hidden_layers=num_vit_layers,
        hidden_size=32,
        num_attention_heads=2,
        qkv_hidden_size=32,
        intermediate_size=64,
        patch_size=4,
        init_pos_emb_height=8,
        init_pos_emb_width=8,
        text_hidden_size=kc.hidden_size,
        rope_max_grid=32,
        merge_kernel_size=(MERGE, MERGE),
    )
    return KimiK3MultimodalConfig(
        kimi_config=kc,
        vision_config=vc,
        num_blocks=None,
        vision_token_id=SENTINEL,
        dep_max_images=2,
        dep_max_grid_h=8,
        dep_max_grid_w=8,
    )


class _Inputs:
    """A tiny batch whose sentinel count matches the projector's token count."""

    def __init__(self, cfg):
        vc = cfg.vision_config
        self.grid = torch.tensor([[1, 4, 4]], dtype=torch.int32)
        n_patches = int(self.grid.prod(dim=-1).sum())
        # One post-merge token per (MERGE, MERGE) block.
        self.n_tokens = (4 // MERGE) * (4 // MERGE)
        torch.manual_seed(3)
        self.pixel_values = torch.randn(
            1, n_patches, vc.in_channels * vc.patch_size * vc.patch_size
        )
        ids = torch.arange(1, 1 + 16, dtype=torch.long).unsqueeze(0)
        ids[0, 2 : 2 + self.n_tokens] = SENTINEL
        self.input_ids = ids


def _stage(cfg):
    """A vision stage with FULLY initialised weights.

    ``init_weights`` is not optional here even though every test builds the model the
    same way. Constructing alone leaves some parameters as raw ``torch.empty`` memory --
    ``patch_embed.pos_emb.weight`` and the MoE expert weights among them -- and two
    "same seed" instances then differ by values like 7e+37 that still look like numbers.
    An equivalence test built on that compares noise and fails for a reason that has
    nothing to do with the code under test. (It did.)
    """
    from torchtitan.models.kimi_k3.moonvit import MoonViT
    from torchtitan.models.kimi_k3.multimodal_model import KimiK3Model

    torch.manual_seed(0)
    tower = MoonViT(cfg.vision_config)
    lm = KimiK3Model.make_config(cfg.kimi_config).build()
    stage = KimiK3ViTStage.from_parts(cfg, tower, lm).to(torch.float32)
    torch.manual_seed(0)
    stage.init_weights()
    return stage.eval()


class _StepInputs:
    """Stand-in for ``VisionStepInputs`` holding one micro-batch."""

    def __init__(self, grid):
        self._grid = grid

    def grid_for(self, mb):
        return self._grid if mb == 0 else None


class TestViTStageRoles(unittest.TestCase):
    def _chain(self, cfg, inputs, num_shares: int):
        """Build ``num_shares`` stages sharing one tower and run them in order."""
        stage = _stage(cfg)
        bounds = stage.vision_tower.block_bounds(num_shares)
        si = _StepInputs(inputs.grid)

        roles = ["head"] + ["body"] * (num_shares - 2) + ["tail"]
        payload = None
        for i, role in enumerate(roles):
            stage.set_dep_role(
                role, bounds=bounds[i], num_shares=num_shares, step_inputs=si
            )
            stage._dep_current_mb = 0
            if role == "head":
                payload = stage(inputs.input_ids, inputs.pixel_values, inputs.grid)
            else:
                payload = stage(*payload)
        return payload

    def test_head_tail_equals_single_stage(self):
        cfg = _cfg()
        inputs = _Inputs(cfg)

        single = _stage(cfg)
        with torch.no_grad():
            want = single(inputs.input_ids, inputs.pixel_values, inputs.grid)
            got = self._chain(cfg, inputs, 2)

        self.assertEqual(got.shape, want.shape)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    def test_head_body_tail_equals_single_stage(self):
        cfg = _cfg()
        inputs = _Inputs(cfg)

        single = _stage(cfg)
        with torch.no_grad():
            want = single(inputs.input_ids, inputs.pixel_values, inputs.grid)
            got = self._chain(cfg, inputs, 3)

        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)

    def test_head_payload_is_fixed_capacity(self):
        """The mid-tower payload must not depend on the batch's image count, or PP's
        one-time buffer sizing is wrong on a later step."""
        cfg = _cfg()
        inputs = _Inputs(cfg)
        stage = _stage(cfg)
        bounds = stage.vision_tower.block_bounds(2)
        stage.set_dep_role("head", bounds=bounds[0], num_shares=2)
        stage._dep_current_mb = 0

        with torch.no_grad():
            patches, text_embeds, mask = stage(
                inputs.input_ids, inputs.pixel_values, inputs.grid
            )
            # Same stage, a batch with NO images: the payload shape must be identical.
            empty_patches, _, _ = stage(inputs.input_ids, None, None)

        expected = cfg.dep_max_images * cfg.dep_max_grid_h * cfg.dep_max_grid_w
        self.assertEqual(patches.shape[0], expected)
        self.assertEqual(empty_patches.shape, patches.shape)
        self.assertEqual(text_embeds.shape[:2], inputs.input_ids.shape)
        self.assertEqual(mask.shape, inputs.input_ids.shape)

    def test_sentinel_mask_marks_exactly_the_sentinels(self):
        cfg = _cfg()
        inputs = _Inputs(cfg)
        stage = _stage(cfg)
        stage.set_dep_role(
            "head", bounds=stage.vision_tower.block_bounds(2)[0], num_shares=2
        )
        stage._dep_current_mb = 0

        with torch.no_grad():
            _, _, mask = stage(inputs.input_ids, inputs.pixel_values, inputs.grid)

        self.assertEqual(int(mask.sum()), inputs.n_tokens)
        self.assertTrue(torch.equal((mask > 0.5), inputs.input_ids == SENTINEL))

    def test_tail_rejects_per_image_convention(self):
        """One sentinel per image changes the sequence length per sample, which PP
        cannot size a buffer for -- it must raise, not produce a working-once shape."""
        cfg = _cfg()
        inputs = _Inputs(cfg)
        stage = _stage(cfg)
        bounds = stage.vision_tower.block_bounds(2)
        si = _StepInputs(inputs.grid)

        stage.set_dep_role("head", bounds=bounds[0], num_shares=2, step_inputs=si)
        stage._dep_current_mb = 0
        with torch.no_grad():
            patches, text_embeds, mask = stage(
                inputs.input_ids, inputs.pixel_values, inputs.grid
            )

        # One sentinel for the whole image instead of one per visual token.
        mask = torch.zeros_like(mask)
        mask[0, 2] = 1.0
        stage.set_dep_role("tail", bounds=bounds[1], num_shares=2, step_inputs=si)
        with self.assertRaises(ValueError) as ctx, torch.no_grad():
            stage(patches, text_embeds, mask)
        self.assertIn("per-token collator convention", str(ctx.exception))

    def test_metadata_inference_passes_shapes_through(self):
        """With no micro-batch in flight, a later share must return the right SHAPES
        without needing grid_thw -- that is all PP is measuring at that point."""
        cfg = _cfg()
        inputs = _Inputs(cfg)
        stage = _stage(cfg)
        bounds = stage.vision_tower.block_bounds(3)
        si = _StepInputs(inputs.grid)

        stage.set_dep_role("head", bounds=bounds[0], num_shares=3, step_inputs=si)
        stage._dep_current_mb = 0
        with torch.no_grad():
            payload = stage(inputs.input_ids, inputs.pixel_values, inputs.grid)

        stage.set_dep_role("body", bounds=bounds[1], num_shares=3, step_inputs=si)
        stage._dep_current_mb = None
        with torch.no_grad():
            body_out = stage(*payload)
        self.assertEqual(len(body_out), 3)
        for a, b in zip(body_out, payload):
            self.assertEqual(a.shape, b.shape)

        stage.set_dep_role("tail", bounds=bounds[2], num_shares=3, step_inputs=si)
        with torch.no_grad():
            tail_out = stage(*payload)
        self.assertEqual(tail_out.shape, payload[1].shape)

    def test_roles_are_validated(self):
        cfg = _cfg()
        stage = _stage(cfg)
        with self.assertRaises(ValueError):
            stage.set_dep_role("middle", bounds=(0, 1))
        with self.assertRaises(ValueError):
            stage.set_dep_role("head")  # no bounds
        # step_inputs is NOT required: PP forwards the batch kwargs to every stage, so
        # a later share normally reads grid_thw straight from them and the cache is
        # only a fallback.
        stage.set_dep_role("tail", bounds=(0, 1))

    def test_gradient_reaches_both_the_head_and_tail_blocks(self):
        """The report balances vision BACKWARD passes too, so both shares must train."""
        cfg = _cfg()
        inputs = _Inputs(cfg)
        stage = _stage(cfg)
        bounds = stage.vision_tower.block_bounds(2)
        si = _StepInputs(inputs.grid)
        stage.set_dep_role("head", bounds=bounds[0], num_shares=2, step_inputs=si)
        stage._dep_current_mb = 0
        payload = stage(inputs.input_ids, inputs.pixel_values, inputs.grid)
        stage.set_dep_role("tail", bounds=bounds[1], num_shares=2, step_inputs=si)
        out = stage(*payload)
        out.sum().backward()

        first = stage.vision_tower.encoder.blocks[bounds[0][0]].wqkv.weight
        last = stage.vision_tower.encoder.blocks[bounds[1][0]].wqkv.weight
        for name, w in (("first share", first), ("last share", last)):
            self.assertIsNotNone(w.grad, f"{name} got no gradient")
            self.assertGreater(float(w.grad.abs().sum()), 0.0, f"{name} grad is zero")


if __name__ == "__main__":
    unittest.main()
