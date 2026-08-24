# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_mask

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.multimodal import scatter_vision_embeds
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.common.vision_encoder import create_block_diagonal_mask
from torchtitan.models.kimi_k2_7.vision_encoder import (
    _compute_2d_rope_cache as _compute_kimi_2d_rope_cache,
    _compute_learned_pos_embeds as _compute_kimi_learned_pos_embeds,
    _tpool_patch_merger,
)
from torchtitan.models.qwen3_5.vision_encoder import (
    _compute_2d_rope_cache,
    _compute_learned_pos_embeds,
    PatchMerger,
)


class TestPackedVision(unittest.TestCase):
    def test_collator_flattens_text_segments(self) -> None:
        tokenizer = type("Tokenizer", (), {"pad_id": 99})()
        context = SimpleNamespace(
            tokenizer=tokenizer,
            num_tokens_per_batch=8,
            max_context_length=4,
        )
        collator = MultiModalCollator.Config(
            patch_size=1,
            temporal_patch_size=1,
            spatial_merge_size=1,
        ).build(context=context)
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 4, 5]),
                "labels": torch.tensor([1, 2, 3, 4, 5]),
                "positions": torch.tensor([0, 1, 2, 3, 4]),
            },
            {
                "input_ids": torch.tensor([6, 7, 8]),
                "labels": torch.tensor([6, 7, 8]),
                "positions": torch.tensor([0, 1, 2]),
            },
        ]

        inputs_T, labels_T, positions_T = collator.collate_text(batch)

        torch.testing.assert_close(inputs_T, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))
        torch.testing.assert_close(labels_T, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))
        torch.testing.assert_close(positions_T, torch.tensor([0, 1, 2, 3, 4, 0, 1, 2]))

    def test_collator_resets_long_padding_positions(self) -> None:
        tokenizer = type("Tokenizer", (), {"pad_id": 99})()
        context = SimpleNamespace(
            tokenizer=tokenizer,
            num_tokens_per_batch=10,
            max_context_length=4,
        )
        collator = MultiModalCollator.Config(
            patch_size=1,
            temporal_patch_size=1,
            spatial_merge_size=1,
        ).build(context=context)
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "labels": torch.tensor([1, 2, 3]),
                "positions": torch.tensor([0, 1, 2]),
            }
        ]

        _, labels, positions = collator.collate_text(batch)

        torch.testing.assert_close(labels[3:], torch.full((7,), IGNORE_INDEX))
        torch.testing.assert_close(
            positions, torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 1, 2])
        )

    def test_collator_concatenates_patches(self) -> None:
        patches_0 = torch.arange(12).view(3, 4)
        patches_1 = torch.arange(8).view(2, 4) + 20
        grid_0 = torch.tensor([1, 1, 3])
        grid_1 = torch.tensor([1, 1, 2])
        tokenizer = type("Tokenizer", (), {"pad_id": 99})()
        context = SimpleNamespace(
            tokenizer=tokenizer,
            num_tokens_per_batch=8,
            max_context_length=4,
        )
        collator = MultiModalCollator.Config(
            patch_size=1,
            temporal_patch_size=1,
            spatial_merge_size=1,
        ).build(context=context)

        with patch(
            "torchtitan.hf_datasets.multimodal.mm_collator.vision_to_patches",
            side_effect=[(patches_0, grid_0), (patches_1, grid_1)],
        ):
            packed, grids = collator.collate_images([torch.empty(0), torch.empty(0)])

        torch.testing.assert_close(packed, torch.cat([patches_0, patches_1]))
        torch.testing.assert_close(grids, torch.stack([grid_0, grid_1]))

    def test_collator_counts_partial_temporal_patch(self) -> None:
        collator = MultiModalCollator.Config(
            max_images_per_batch=1,
            temporal_patch_size=2,
        ).build(
            context=SimpleNamespace(
                tokenizer=None,
                num_tokens_per_batch=0,
                max_context_length=0,
            )
        )
        batch = [{"pixel_values_videos": [torch.empty(3, 1, 1, 1)]}]

        with self.assertRaisesRegex(ValueError, "2 vision entries"):
            collator(batch)

    def test_block_diagonal_mask_respects_segments(self) -> None:
        mask = create_block_diagonal_mask(
            torch.tensor([3, 2]), total_tokens=5, device=torch.device("cpu")
        )
        dense_mask = create_mask(
            mask.mask_mod,
            B=1,
            H=1,
            Q_LEN=5,
            KV_LEN=5,
            device="cpu",
        )[0, 0]
        expected = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 1, 1],
                [0, 0, 0, 1, 1],
            ],
            dtype=torch.bool,
        )
        torch.testing.assert_close(dense_mask, expected)

    def test_qwen_position_embeddings_are_packed(self) -> None:
        grids = [[1, 2, 2], [2, 2, 2]]
        learned_pos = _compute_learned_pos_embeds(
            torch.randn(16, 4),
            grids,
            num_grid_per_side=4,
            spatial_merge_size=1,
            dim=4,
        )
        rope_cache = _compute_2d_rope_cache(
            torch.randn(2, 2),
            grids,
            spatial_merge_size=1,
            head_dim=8,
        )

        self.assertEqual(learned_pos.shape, (12, 4))
        self.assertEqual(rope_cache.shape, (12, 1, 16))

    def test_qwen_patch_merger_keeps_packed_order(self) -> None:
        merger = PatchMerger(
            PatchMerger.Config(
                spatial_merge_size=2,
                merged_hidden_size=8,
                norm=LayerNorm.Config(normalized_shape=2),
                fc1=Linear.Config(in_features=8, out_features=8),
                fc2=Linear.Config(in_features=8, out_features=8),
            )
        )
        merger.norm = nn.Identity()
        merger.linear_fc1 = nn.Identity()
        merger.act_fn = nn.Identity()
        merger.linear_fc2 = nn.Identity()
        x_TD = torch.arange(24).view(12, 2)

        merged_MK = merger(x_TD)

        torch.testing.assert_close(merged_MK, x_TD.view(3, 8))

    def test_kimi_position_embeddings_are_packed(self) -> None:
        grids = [[1, 2, 2], [2, 2, 2]]
        learned_pos = _compute_kimi_learned_pos_embeds(
            torch.randn(2, 2, 4), grids, interpolation_mode="bicubic"
        )
        rope_cache = _compute_kimi_2d_rope_cache(torch.randn(2, 2), grids, head_dim=8)

        self.assertEqual(learned_pos.shape, (12, 4))
        self.assertEqual(rope_cache.shape, (12, 1, 4))

    def test_kimi_temporal_pool_and_spatial_merge_are_packed(self) -> None:
        hidden_TD = torch.arange(12, dtype=torch.float32).view(12, 1)

        merged_MK = _tpool_patch_merger(
            hidden_TD,
            grids=[[1, 2, 2], [2, 2, 2]],
            merge_kernel_size=(2, 2),
        )

        expected_MK = torch.tensor([[0.0, 1.0, 2.0, 3.0], [6.0, 7.0, 8.0, 9.0]])
        torch.testing.assert_close(merged_MK, expected_MK)

    def test_scatter_vision_embeds_uses_packed_layout(self) -> None:
        inputs_TD = torch.zeros(5, 2)
        vision_TD = torch.arange(8, dtype=torch.float32).view(4, 2)

        result_TD = scatter_vision_embeds(
            inputs_TD,
            vision_embeds=vision_TD,
            vision_positions=[(0, 0, 2), (1, 3, 2)],
        )

        torch.testing.assert_close(result_TD[:2], vision_TD[:2])
        torch.testing.assert_close(result_TD[3:], vision_TD[2:])
        torch.testing.assert_close(result_TD[2], torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
