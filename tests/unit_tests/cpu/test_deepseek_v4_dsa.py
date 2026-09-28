# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for DeepSeek-V4 CSA top-k selection and packed-document handling."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchtitan.models.common.attention import VarlenAttentionMetadata
from torchtitan.models.deepseek_v4.compressor import Indexer
from torchtitan_recipes.tests.models import deepseek_v4 as config_registry


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
            idx_q, idx_k, idx_w, max_seqlen=seqlen, ratio=ratio, topk=topk
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
    def test_get_attention_metadata_builds_document_offsets(self):
        # Documents start where positions reset; the padding tail is its own segment.
        positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 0, 1])
        padding_mask = torch.tensor([False] * 8 + [True] * 2)
        with torch.device("meta"):
            config = config_registry.deepseek_v4_debugmodel(seq_len=128)
            config.model.set_sharding_(config.parallelism)
            model = config.model.build()
        metadata = model.get_attention_metadata(positions, padding_mask=padding_mask)
        self.assertTrue(metadata)
        for value in metadata.values():
            self.assertIsInstance(value, VarlenAttentionMetadata)
            self.assertEqual(value.cu_seq_q.tolist(), [0, 3, 8, 10])

    def test_preprocess_inputs_passes_document_offsets(self):
        with torch.device("meta"):
            config = config_registry.deepseek_v4_debugmodel(seq_len=128)
            config.model.set_sharding_(config.parallelism)
            model = config.model.build()
        input_dict = {
            "input": torch.zeros(128, dtype=torch.long),
            "labels": torch.zeros(128, dtype=torch.long),
            "positions": torch.arange(64).repeat(2),
        }
        # SPMD annotation needs a device mesh; this checks only the plumbing.
        with patch(
            "torchtitan.models.common.decoder.annotate_input_spmd_types",
            side_effect=lambda _parallelism_context, batch, _input_sharding: batch,
        ):
            _, _, kwargs = model.preprocess_inputs(
                input_dict,
                parallelism_context=SimpleNamespace(
                    cp_enabled=False, activate_spmd=contextlib.nullcontext
                ),
                parallelism=SimpleNamespace(),
            )
        metadata = kwargs["attention_metadata"]
        self.assertTrue(metadata)
        for value in metadata.values():
            self.assertEqual(value.cu_seq_q.tolist(), [0, 64, 128])

    def test_packed_attention_matches_each_document_alone(self):
        """Packed documents must not see each other through the sliding window,
        the compressed KV pool, or the indexer."""
        torch.manual_seed(0)
        config = config_registry.deepseek_v4_debugmodel()
        config.model.set_sharding_(config.parallelism)
        model_config = config.model
        # One layer per attention type (the debug model has two SWA layers).
        attention_configs = {
            layer.attention.compress_ratio: layer.attention
            for layer in model_config.layers
        }
        # Exactly-zero indexer scores (every head's relu at 0) tie, and top-k
        # breaks ties by layout; let CSA select every causal candidate instead
        # (the longest document below has 384 // 4 compressed entries).
        attention_configs[4].inner_attention.index_topk = 96
        # Group-aligned lengths also compare against the unpacked path; the
        # others cover partial groups and documents shorter than a group.
        for lengths, reference in (
            ([256, 128, 384], "unpacked"),
            ([300, 37, 150, 5], "packed alone"),
        ):
            positions = torch.cat([torch.arange(n) for n in lengths])
            for attention_config in attention_configs.values():
                attention = attention_config.build().double()
                backend = type(attention.inner_attention)
                metadata = backend.build_attention_metadata(
                    positions,
                    config=attention_config.inner_attention,
                )
                for param in attention.parameters():
                    torch.nn.init.normal_(param, std=0.1)
                x = torch.randn(len(positions), model_config.dim, dtype=torch.float64)
                name = type(attention.inner_attention).__name__
                with self.subTest(lengths=lengths, attention=name):
                    packed = attention(
                        x, attention_metadata=metadata, positions=positions
                    )
                    docs = []
                    for x_doc in x.split(lengths):
                        doc_positions = torch.arange(len(x_doc))
                        doc_metadata = (
                            None
                            if reference == "unpacked"
                            else backend.build_attention_metadata(
                                doc_positions,
                                config=attention_config.inner_attention,
                            )
                        )
                        docs.append(
                            attention(
                                x_doc,
                                attention_metadata=doc_metadata,
                                positions=doc_positions,
                            )
                        )
                    torch.testing.assert_close(packed, torch.cat(docs))


if __name__ == "__main__":
    unittest.main()
