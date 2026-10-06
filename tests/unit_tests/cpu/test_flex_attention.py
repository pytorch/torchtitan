# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
from torch.nn.attention.flex_attention import create_block_mask

from torchtitan.config.transform.base import convert_config_type
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.cp_attention import (
    KVAllGatherCPFlexInnerAttention,
    UlyssesCPFlexInnerAttention,
)
from torchtitan.models.kimi_k2_7.qk_clip import QKClipFlexInnerAttention


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
    def setUp(self) -> None:
        cache_patch = patch.object(FlexInnerAttention, "_compiled_flex_attn_cache", [])
        cache_patch.start()
        self.addCleanup(cache_patch.stop)

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

    def test_config_selects_backend_and_shares_compilation(self) -> None:
        compiled_options = []

        def backend(graph, example_inputs, *, options):
            compiled_options.append(options)
            return graph.forward

        def kernel(q_1HTK, k_1HTK, v_1HTV, **kwargs):
            return q_1HTK + k_1HTK + v_1HTV, SimpleNamespace(lse=None)

        options: dict[str, Any] = {"custom_flag": True, "nested": {"levels": [1, 2]}}
        q_THK = torch.randn(4, 2, 8, requires_grad=True)
        mask = self._mask(4)
        with (
            patch("torchtitan.models.common.attention.flex_attention", kernel),
            patch("torch.compile", wraps=torch.compile) as compile_fn,
            patch.object(
                FlexInnerAttention,
                "_compiled_flex_attn",
                side_effect=AssertionError("The default backend must not run"),
            ),
        ):
            attention = FlexInnerAttention.Config(
                compile_backend=backend, compile_options=options
            ).build()
            other_attention = FlexInnerAttention.Config(
                compile_backend=backend,
                compile_options={"nested": {"levels": [1, 2]}, "custom_flag": True},
            ).build()
            self.assertIs(
                attention._compiled_flex_attn_override,
                other_attention._compiled_flex_attn_override,
            )
            compile_fn.assert_called_once_with(kernel, backend=backend, options=options)
            # Mutating the recipe after construction must not change cached options.
            options["nested"]["levels"].append(3)
            for instance in (attention, other_attention):
                out_THV = instance(q_THK, q_THK, q_THK, attention_masks=mask)
                torch.testing.assert_close(out_THV, q_THK * 3, rtol=0, atol=0)
            out_THV.sum().backward()

        self.assertEqual(
            compiled_options,
            [{"custom_flag": True, "nested": {"levels": [1, 2]}}],
        )
        torch.testing.assert_close(q_THK.grad, torch.full_like(q_THK, 3))

    def test_backend_options_are_forwarded_without_inductor_defaults(self) -> None:
        for backend, options in (
            ("eager", None),
            ("eager", {}),
            ("eager", {"custom_flag": True}),
            ("inductor", {}),
            ("inductor", {"max_autotune": False}),
        ):
            with self.subTest(backend=backend, options=options):
                with patch("torch.compile") as compile_fn:
                    FlexInnerAttention.Config(
                        compile_backend=backend, compile_options=options
                    ).build()
                compile_fn.assert_called_once()
                self.assertEqual(
                    compile_fn.call_args.kwargs,
                    {"backend": backend, "options": options},
                )

    def test_cache_separates_backends_and_snapshots_options(self) -> None:
        options = {"nested": {"levels": [1]}}
        with patch(
            "torch.compile", side_effect=[lambda: None for _ in range(3)]
        ) as compile_fn:
            first = FlexInnerAttention.Config(
                compile_backend="backend_a", compile_options=options
            ).build()
            different_backend = FlexInnerAttention.Config(
                compile_backend="backend_b", compile_options=options
            ).build()
            options["nested"]["levels"].append(2)
            different_options = FlexInnerAttention.Config(
                compile_backend="backend_a", compile_options=options
            ).build()
            original_options = FlexInnerAttention.Config(
                compile_backend="backend_a", compile_options={"nested": {"levels": [1]}}
            ).build()

        self.assertEqual(compile_fn.call_count, 3)
        self.assertIs(
            first._compiled_flex_attn_override,
            original_options._compiled_flex_attn_override,
        )
        self.assertIsNot(
            first._compiled_flex_attn_override,
            different_backend._compiled_flex_attn_override,
        )
        self.assertIsNot(
            first._compiled_flex_attn_override,
            different_options._compiled_flex_attn_override,
        )

    def test_compiler_config_survives_attention_conversion(self) -> None:
        config = FlexInnerAttention.Config(
            compile_backend="eager", compile_options={"custom_flag": True}
        )
        with patch("torch.compile") as compile_fn:
            for attention_type in (
                KVAllGatherCPFlexInnerAttention,
                UlyssesCPFlexInnerAttention,
                QKClipFlexInnerAttention,
            ):
                with self.subTest(attention_type=attention_type.__name__):
                    converted = convert_config_type(config, attention_type)
                    attention = converted.build()
                    self.assertIs(
                        attention._compiled_flex_attn_override, compile_fn.return_value
                    )
        compile_fn.assert_called_once()
        self.assertEqual(
            compile_fn.call_args.kwargs,
            {"backend": "eager", "options": {"custom_flag": True}},
        )

    def test_default_callable_stays_shared_and_replaceable(self) -> None:
        class InheritedFlexInnerAttention(FlexInnerAttention):
            @dataclass(kw_only=True, slots=True)
            class Config(FlexInnerAttention.Config):
                pass

        original_callable = FlexInnerAttention._compiled_flex_attn
        with patch("torch.compile", side_effect=AssertionError("Unexpected compile")):
            attention = FlexInnerAttention.Config().build()
            inherited_attention = InheritedFlexInnerAttention.Config().build()

        for instance in (attention, inherited_attention):
            self.assertIsNone(instance._compiled_flex_attn_override)
        self.assertIs(FlexInnerAttention._compiled_flex_attn, original_callable)
        self.assertEqual(FlexInnerAttention._compiled_flex_attn_cache, [])

        q_THK = torch.randn(4, 2, 8)
        mask = self._mask(4)

        def replacement(q_1HTK, k_1HTK, v_1HTV, **kwargs):
            return q_1HTK + 1, SimpleNamespace(lse=None)

        # Keep replacements of the default callable visible to existing instances.
        with patch.object(FlexInnerAttention, "_compiled_flex_attn", replacement):
            for instance in (attention, inherited_attention):
                out_THV = instance(q_THK, q_THK, q_THK, attention_masks=mask)
                torch.testing.assert_close(out_THV, q_THK + 1)


if __name__ == "__main__":
    unittest.main()
