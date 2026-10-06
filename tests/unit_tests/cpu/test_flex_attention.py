# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.models.common.attention import FlexInnerAttention


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
                attention_masks=self._mask(num_tokens, 1),
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
                attention_masks=self._mask(num_tokens, 1),
                out_transform=out_transform,
            )

        torch.testing.assert_close(out_THV, q_THK)


class TestFlexInnerAttentionCompilation(unittest.TestCase):
    @staticmethod
    def _mask(num_tokens: int):
        return create_block_mask(
            lambda b, h, q_idx, kv_idx: q_idx >= kv_idx,
            B=1,
            H=None,
            Q_LEN=num_tokens,
            KV_LEN=num_tokens,
            device="cpu",
        )

    def test_subclass_selects_compile_backend(self) -> None:
        compiled_graphs = []

        def backend(graph, example_inputs):
            compiled_graphs.append(graph)
            return graph.forward

        def kernel(q_1HTK, k_1HTK, v_1HTV, **kwargs):
            return q_1HTK + k_1HTK + v_1HTV, SimpleNamespace(lse=None)

        class CustomFlexInnerAttention(FlexInnerAttention):
            @dataclass(kw_only=True, slots=True)
            class Config(FlexInnerAttention.Config):
                pass

            _compiled_flex_attn = torch.compile(kernel, backend=backend, fullgraph=True)

        attention = CustomFlexInnerAttention.Config().build()
        other_attention = CustomFlexInnerAttention.Config().build()
        self.assertIsInstance(attention, CustomFlexInnerAttention)
        self.assertIs(
            type(attention)._compiled_flex_attn,
            type(other_attention)._compiled_flex_attn,
        )
        q_THK = torch.randn(4, 2, 8, requires_grad=True)
        mask = self._mask(4)

        with patch.object(
            FlexInnerAttention,
            "_compiled_flex_attn",
            side_effect=AssertionError("The base callable must not run"),
        ):
            for instance in (attention, other_attention):
                out_THV = instance(q_THK, q_THK, q_THK, attention_masks=mask)
                torch.testing.assert_close(out_THV, q_THK * 3, rtol=0, atol=0)
            out_THV.sum().backward()

        self.assertEqual(len(compiled_graphs), 1)
        torch.testing.assert_close(q_THK.grad, torch.full_like(q_THK, 3))

    def test_inherited_callable_stays_shared_and_replaceable(self) -> None:
        class InheritedFlexInnerAttention(FlexInnerAttention):
            @dataclass(kw_only=True, slots=True)
            class Config(FlexInnerAttention.Config):
                pass

        original_callable = FlexInnerAttention._compiled_flex_attn
        with patch("torch.compile", side_effect=AssertionError("Unexpected compile")):
            attention = FlexInnerAttention.Config().build()
            inherited_attention = InheritedFlexInnerAttention.Config().build()

        for instance in (attention, inherited_attention):
            self.assertIs(type(instance)._compiled_flex_attn, original_callable)

        q_THK = torch.randn(4, 2, 8)
        mask = self._mask(4)

        def replacement(q_1HTK, k_1HTK, v_1HTV, **kwargs):
            return q_1HTK + 1, SimpleNamespace(lse=None)

        # Keep replacements of the base callable visible to existing instances.
        with patch.object(FlexInnerAttention, "_compiled_flex_attn", replacement):
            for instance in (attention, inherited_attention):
                out_THV = instance(q_THK, q_THK, q_THK, attention_masks=mask)
                torch.testing.assert_close(out_THV, q_THK + 1)


if __name__ == "__main__":
    unittest.main()
