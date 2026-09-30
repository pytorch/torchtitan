# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for DeepSeek-V4 CSA top-k selection and packed-document handling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.models.deepseek_v4 import config_registry, model_registry
from torchtitan.models.deepseek_v4.compressor import Indexer
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model


class TestIndexerSelect(unittest.TestCase):
    def test_indexer_select_matches_old_topk(self):
        torch.manual_seed(0)
        device = torch.device("cpu")
        seqlen, ratio, topk = 512, 4, 16
        n_cmp = seqlen // ratio
        g = torch.Generator(device).manual_seed(11)
        idx_q = torch.randn(seqlen, 8, 32, generator=g, device=device)
        idx_k = torch.randn(n_cmp, 32, generator=g, device=device)
        idx_w = torch.randn(seqlen, 8, generator=g, device=device)

        selected = Indexer.select(
            idx_q, idx_k, idx_w, seqlen=seqlen, ratio=ratio, topk=topk
        )

        # Old formulation: causal-masked scores -> topk -> map invalid to -1.
        scores = torch.einsum("shd,td->sht", idx_q, idx_k)
        scores = scores.relu_() * idx_w.unsqueeze(-1)
        scores = scores.sum(dim=1)
        causal_limit = torch.arange(1, seqlen + 1, device=device).unsqueeze(1) // ratio
        mask = torch.arange(n_cmp, device=device).repeat(seqlen, 1) >= causal_limit
        scores = scores + torch.where(mask, torch.finfo(idx_q.dtype).min, 0)
        _, topk_idxs = scores.topk(min(topk, n_cmp), dim=-1)
        old = torch.where(topk_idxs >= causal_limit, -1, topk_idxs)

        # lightning_indexer marks causally unselectable slots -1 itself and
        # leaves order unspecified, so compare each row as a sorted set.
        self.assertEqual(selected.shape, (seqlen, topk))
        self.assertEqual(selected.dtype, torch.int32)
        self.assertTrue(
            torch.equal(selected.long().sort(dim=-1).values, old.sort(dim=-1).values)
        )


class TestDSVPackedDocuments(unittest.TestCase):
    def test_recipes_preserve_single_document_batches_after_resume(self):
        tokenizer = HuggingFaceTokenizer(tokenizer_path="tests/assets/tokenizer")
        for recipe in (
            config_registry.deepseek_v4_debugmodel,
            config_registry.deepseek_v4_mtp_debugmodel,
            config_registry.deepseek_v4_flash,
            config_registry.deepseek_v4_pro,
        ):
            with self.subTest(recipe=recipe.__name__):
                config = recipe(seq_len=512)
                config.dataloader.shuffle = False
                config.dataloader.num_prefetch_microbatches = 0
                dataloader = config.dataloader.build(
                    dp_world_size=1,
                    dp_rank=0,
                    tokenizer=tokenizer,
                    max_context_length=config.training.max_context_length,
                    num_tokens_per_microbatch=(
                        config.training.num_tokens_per_microbatch_per_dp_rank
                    ),
                )
                try:
                    iterator = iter(dataloader)
                    for _ in range(8):
                        batch = next(iterator)
                        self.assertIsNone(
                            DeepSeekV4Model.get_attention_masks(
                                None,
                                batch.positions,
                                padding_mask=batch.padding_mask,
                            )
                        )
                        self.assertEqual(batch.input.numel(), 512)
                        valid_positions = batch.positions[~batch.padding_mask]
                        self.assertTrue(
                            torch.equal(
                                valid_positions, torch.arange(len(valid_positions))
                            )
                        )
                        self.assertTrue(
                            torch.all(batch.labels[batch.padding_mask] == IGNORE_INDEX)
                        )
                    state = dataloader.state_dict()
                    expected = next(iterator)
                    dataloader.load_state_dict(state)
                    actual = next(iter(dataloader))
                    for field in ("input", "labels", "positions", "padding_mask"):
                        self.assertTrue(
                            torch.equal(
                                getattr(expected, field), getattr(actual, field)
                            )
                        )
                finally:
                    dataloader.close()

    def test_get_attention_masks_rejects_position_resets(self):
        positions = torch.arange(64).repeat(2)
        with self.assertRaisesRegex(
            NotImplementedError, "packed documents.*position resets"
        ):
            DeepSeekV4Model.get_attention_masks(None, positions)

    def test_preprocess_inputs_rejects_position_resets(self):
        # The shared decoder hook skips get_attention_masks for non-Flex cores.
        with torch.device("meta"):
            model = model_registry("debugmodel", enable_sp=False).build()
        input_dict = {
            "input": torch.zeros(128, dtype=torch.long),
            "labels": torch.zeros(128, dtype=torch.long),
            "positions": torch.arange(64).repeat(2),
        }
        with patch(
            "torchtitan.models.common.decoder.annotate_input_spmd_types",
            side_effect=lambda _parallelism_context, batch, _input_sharding: batch,
        ), self.assertRaisesRegex(
            NotImplementedError, "packed documents.*position resets"
        ):
            model.preprocess_inputs(
                input_dict,
                parallelism_context=SimpleNamespace(cp_enabled=False),
                parallelism=SimpleNamespace(),
            )

    def test_get_attention_masks_accepts_single_document(self):
        positions = torch.arange(128)
        self.assertIsNone(DeepSeekV4Model.get_attention_masks(None, positions))

    def test_get_attention_masks_ignores_padding_position_resets(self):
        positions = torch.tensor([0, 1, 2, 3, 0, 1])
        padding_mask = torch.tensor([False, False, False, False, True, True])
        self.assertIsNone(
            DeepSeekV4Model.get_attention_masks(
                None, positions, padding_mask=padding_mask
            )
        )


if __name__ == "__main__":
    unittest.main()
