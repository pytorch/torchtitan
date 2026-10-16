# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.attention.flex_attention import and_masks, create_block_mask

from torchtitan.models.common.attention import (
    FlexInnerAttention,
    get_causal_mask_mod,
    get_document_mask_mod,
    SlidingWindowFlexInnerAttention,
)


class TestFlexInnerAttentionLayouts(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = FlexInnerAttention(FlexInnerAttention.Config())

    @staticmethod
    def _mask(seq_len: int, batch_size: int):
        return create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=batch_size,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device="cpu",
        )

    def test_thk_thv_layout(self) -> None:
        num_tokens, num_heads, head_dim = 8, 4, 16
        q_THK = torch.randn(num_tokens, num_heads, head_dim)
        k_THK = torch.randn_like(q_THK)
        v_THV = torch.randn_like(q_THK)

        def kernel(q_1HTK, k_1HTK, v_1HTV, **kwargs):
            self.assertEqual(q_1HTK.shape, (1, num_heads, num_tokens, head_dim))
            self.assertEqual(k_1HTK.shape, q_1HTK.shape)
            self.assertEqual(v_1HTV.shape, q_1HTK.shape)
            lse_1HT = torch.randn(1, num_heads, num_tokens)
            return q_1HTK, SimpleNamespace(lse=lse_1HT)

        with patch.object(FlexInnerAttention, "compiled_flex_attn", side_effect=kernel):
            out_THV = self.attention(
                q_THK,
                k_THK,
                v_THV,
                attention_metadata=self._mask(num_tokens, 1),
            )

        torch.testing.assert_close(out_THV, q_THK)

    def test_thk_thv_out_transform_layout(self) -> None:
        num_tokens, num_heads, head_dim = 8, 4, 16
        q_THK = torch.randn(num_tokens, num_heads, head_dim)
        expected_lse_TH = torch.randn(num_tokens, num_heads)

        def kernel(q_1HTK, k_1HTK, v_1HTV, **kwargs):
            return q_1HTK, SimpleNamespace(
                lse=expected_lse_TH.transpose(0, 1).unsqueeze(0)
            )

        def out_transform(out_THV, lse_TH):
            torch.testing.assert_close(lse_TH, expected_lse_TH)
            return out_THV

        with patch.object(FlexInnerAttention, "compiled_flex_attn", side_effect=kernel):
            out_THV = self.attention(
                q_THK,
                q_THK,
                q_THK,
                attention_metadata=self._mask(num_tokens, 1),
                out_transform=out_transform,
            )

        torch.testing.assert_close(out_THV, q_THK)


# Six packed documents in one 16-token sequence. Positions reset to 0 at each
# document start.
PACKED_POSITIONS = torch.tensor(
    [0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1], dtype=torch.int32
)
# One document per sequence: block_causal and causal must agree here.
SINGLE_DOC_POSITIONS = torch.arange(16, dtype=torch.int32)
SEQ_LEN = 16
MASK_BLOCK_SIZE = 4


def _dense(mask_mods, seq_len: int) -> torch.Tensor:
    """Materialize composed mask_mods into a [seq_len, seq_len] bool matrix."""
    composed = and_masks(*mask_mods)
    b = h = torch.tensor(0)
    return torch.tensor(
        [
            [
                bool(composed(b, h, torch.tensor(q), torch.tensor(kv)))
                for kv in range(seq_len)
            ]
            for q in range(seq_len)
        ]
    )


def _mods(positions: torch.Tensor, mask_type: str) -> list:
    return FlexInnerAttention.Config(mask_type=mask_type)._causal_mask_mods(positions)


class TestFlexMaskType(unittest.TestCase):
    def test_mask_mods_match_mask_type(self) -> None:
        for positions in (PACKED_POSITIONS, SINGLE_DOC_POSITIONS):
            causal = _dense([get_causal_mask_mod()], SEQ_LEN)
            same_doc = _dense([get_document_mask_mod(positions)], SEQ_LEN)
            for mask_type, expected in (
                ("block_causal", causal & same_doc),
                ("causal", causal),
            ):
                with self.subTest(
                    single_doc=positions is SINGLE_DOC_POSITIONS, mask_type=mask_type
                ):
                    torch.testing.assert_close(
                        _dense(_mods(positions, mask_type), SEQ_LEN), expected
                    )

    def test_block_causal_blocks_more_than_causal(self) -> None:
        causal = _dense(_mods(PACKED_POSITIONS, "causal"), SEQ_LEN)
        block_causal = _dense(_mods(PACKED_POSITIONS, "block_causal"), SEQ_LEN)
        # block_causal never allows what causal blocks, and allows strictly less.
        self.assertTrue(bool((block_causal <= causal).all()))
        self.assertLess(int(block_causal.sum()), int(causal.sum()))

    def test_unknown_mask_type_refused(self) -> None:
        with self.assertRaises(ValueError):
            _mods(PACKED_POSITIONS, "casual")

    def test_configs_default_to_block_causal_and_honor_mask_type(self) -> None:
        for name, make in (
            ("flex", lambda **kw: FlexInnerAttention.Config(**kw)),
            (
                "sliding_window",
                lambda **kw: SlidingWindowFlexInnerAttention.Config(
                    window_size=8, **kw
                ),
            ),
        ):
            with self.subTest(config=name):
                self.assertEqual(make().mask_type, "block_causal")
                block_causal = make(
                    block_size=MASK_BLOCK_SIZE
                ).build_attention_metadata(PACKED_POSITIONS)
                causal = make(
                    block_size=MASK_BLOCK_SIZE, mask_type="causal"
                ).build_attention_metadata(PACKED_POSITIONS)
                self.assertFalse(
                    bool(torch.equal(block_causal.to_dense(), causal.to_dense()))
                )


if __name__ == "__main__":
    unittest.main()
