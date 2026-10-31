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

from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    VarlenAttentionMetadata,
)
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4.attention import DeepSeekV4CompressedSparseAttention
from torchtitan.models.deepseek_v4.compressor import InnerIndexer
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

        metadata = DeepSeekV4CompressedSparseAttention.Config(
            window_size=1, compress_ratio=ratio, softmax_scale=1.0
        ).build_attention_metadata(torch.arange(seqlen))
        inner_indexer = InnerIndexer(
            InnerIndexer.Config(topk=topk, compress_ratio=ratio)
        )
        selected, selected_scores = inner_indexer(
            idx_q, idx_k, idx_w, attention_metadata=metadata, return_scores=True
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
        expected_scores = scores.gather(1, selected.long().clamp_min(0))
        expected_scores = expected_scores.masked_fill(selected < 0, -torch.inf)
        torch.testing.assert_close(selected_scores, expected_scores)


class TestDSVPackedDocuments(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(AuxLoss._group_counts, clear=True))
        self.enterContext(patch.dict(AuxLoss.group_acc, clear=True))

    def test_get_attention_metadata_builds_document_offsets(self):
        # Documents start where positions reset and are compression-aligned.
        positions = torch.arange(128).repeat(3)
        padding_mask = torch.zeros(384, dtype=torch.bool)
        with torch.device("meta"):
            config = config_registry.deepseek_v4_debugmodel(seq_len=128)
            model = config.model.build()
        metadata = model._get_attention_metadata(positions, padding_mask=padding_mask)
        # One metadata object per inner-attention variant, not per layer.
        self.assertEqual(len(metadata), 3)
        for value in metadata.values():
            self.assertEqual(value.cu_seq_q.tolist(), [0, 128, 256, 384])

    def test_preprocess_inputs_passes_document_offsets(self):
        with torch.device("meta"):
            config = config_registry.deepseek_v4_debugmodel(seq_len=128)
            model = config.model.build()
        input_dict = {
            "input": torch.zeros(256, dtype=torch.long),
            "labels": torch.zeros(256, dtype=torch.long),
            "positions": torch.arange(128).repeat(2),
        }
        # SPMD annotation needs a device mesh; this checks only the plumbing.
        with patch(
            "torchtitan.models.common.decoder.annotate_input_spmd_types",
            side_effect=lambda _parallelism_context, batch, _input_sharding: batch,
        ):
            _, _, kwargs, _ = model.preprocess_inputs(
                input_dict,
                parallelism_context=SimpleNamespace(
                    cp_enabled=False, activate_spmd=contextlib.nullcontext
                ),
                parallelism=SimpleNamespace(),
            )
        metadata = kwargs["attention_metadata"]
        self.assertEqual(len(metadata), 3)
        for value in metadata.values():
            self.assertEqual(value.cu_seq_q.tolist(), [0, 128, 256])

    def test_packed_attention_matches_each_document_alone(self):
        """Packed documents must not see each other through the sliding window,
        the compressed KV pool, or the indexer."""
        torch.manual_seed(0)
        config = config_registry.deepseek_v4_debugmodel()
        model_config = config.model
        # One layer per attention type (the debug model has two SWA layers).
        attention_configs = {
            layer.attention.compress_ratio: layer.attention
            for layer in model_config.layers
        }
        # Exactly-zero indexer scores (every head's relu at 0) tie, and top-k
        # breaks ties by layout; let CSA select every causal candidate instead
        # (the longest document below has 384 // 4 compressed entries).
        attention_configs[4].indexer.inner_indexer.topk = 96
        # Group-aligned lengths also compare against the unpacked path; the
        # others cover partial groups and documents shorter than a group.
        for lengths, reference in (
            ([256, 128, 384], "unpacked"),
            ([128, 128, 128], "packed alone"),
        ):
            positions = torch.cat([torch.arange(n) for n in lengths])
            for attention_config in attention_configs.values():
                attention = attention_config.build().double()
                metadata = attention_config.inner_attention.build_attention_metadata(
                    positions
                )
                for param in attention.parameters():
                    torch.nn.init.normal_(param, std=0.1)
                x = torch.randn(len(positions), model_config.dim, dtype=torch.float64)
                name = type(attention.inner_attention).__name__
                with self.subTest(lengths=lengths, attention=name):
                    packed = attention(
                        x,
                        attention_metadata=metadata,
                        positions=positions,
                        aux_loss_denominator=torch.tensor(len(positions)),
                    )
                    docs = []
                    for x_doc in x.split(lengths):
                        doc_positions = torch.arange(len(x_doc))
                        doc_metadata = (
                            attention_config.inner_attention.build_attention_metadata(
                                doc_positions
                            )
                        )
                        docs.append(
                            attention(
                                x_doc,
                                attention_metadata=doc_metadata,
                                positions=doc_positions,
                                aux_loss_denominator=torch.tensor(len(x_doc)),
                            )
                        )
                    torch.testing.assert_close(packed, torch.cat(docs))


class TestDSV4MetadataGeometry(unittest.TestCase):
    def test_each_variant_metadata_matches_its_ratio(self):
        """A shared metadata object must carry its variant's own geometry."""
        model_config = config_registry.deepseek_v4_debugmodel(seq_len=1024).model
        lengths = (256, 128, 384)
        positions = torch.cat([torch.arange(n) for n in lengths])
        cu = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), 0).tolist()])
        doc_T = torch.repeat_interleave(
            torch.arange(len(lengths)), torch.tensor(lengths)
        )
        # ``positions`` restart at every document, as packing produces them.
        pos_in_doc = positions

        seen_ratios = set()
        for layer in model_config.layers:
            inner = layer.attention.inner_attention
            ratio = inner.compress_ratio
            seen_ratios.add(ratio)
            metadata = inner.build_attention_metadata(positions)
            if ratio <= 1:
                self.assertIsInstance(metadata, VarlenAttentionMetadata)
                self.assertEqual(metadata.cu_seq_q.tolist(), cu.tolist())
                continue

            expected_cmp = torch.cat(
                [torch.tensor([0]), torch.cumsum(torch.tensor(lengths) // ratio, 0)]
            ).int()
            self.assertEqual(metadata.cu_seq_cmp.tolist(), expected_cmp.tolist())
            if ratio == 4:
                self.assertEqual(
                    metadata.cmp_base_T.tolist(), expected_cmp[doc_T].tolist()
                )
            else:
                n_cmp = min(len(positions) // ratio, int(lengths[-1]) // ratio)
                limit = (pos_in_doc + 1) // ratio
                expected = torch.where(
                    torch.arange(n_cmp)[None, :] < limit[:, None],
                    torch.arange(n_cmp)[None, :],
                    -1,
                )
                self.assertEqual(metadata.causal_cmp_TK.tolist(), expected.tolist())
        self.assertEqual(seen_ratios, {1, 4, 128})


if __name__ == "__main__":
    unittest.main()
