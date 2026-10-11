# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass

import torch

from torchtitan.experiments.graph_trainer.common_utils import (
    build_decoder_config_for_backend,
)
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.linear import Linear

from tests.unit_tests.cpu.models.common.attention_reference import (
    CausalAttentionReference,
)


class _AttentionModel(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        inner_attention: Module.Config
        linear: Linear.Config


def _build_attention_config(*, attn_backend):
    return _AttentionModel.Config(
        inner_attention=get_attention_config(attn_backend),
        linear=Linear.Config(in_features=8, out_features=8),
    )


class TestGraphTrainerAttention(unittest.TestCase):
    def test_varlen_config_keeps_traceable_linears(self):
        config = build_decoder_config_for_backend(_build_attention_config, "varlen")
        self.assertIsInstance(config.inner_attention, VarlenInnerAttention.Config)
        self.assertFalse(config.linear.inplace_wgrad_accum)

    def test_sdpa_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sdpa is no longer supported"):
            build_decoder_config_for_backend(_build_attention_config, "sdpa")

    def test_varlen_recipe_selects_varlen(self):
        from torchtitan_recipes.tests.graph_trainer.llama3 import (
            graph_trainer_llama3_debugmodel_varlen,
        )

        config = graph_trainer_llama3_debugmodel_varlen()
        assert isinstance(config.model, Decoder.Config)
        for layer in config.model.layers:
            self.assertIsInstance(
                layer.attention.inner_attention, VarlenInnerAttention.Config
            )

    def test_cpu_reference_is_causal_with_grouped_heads(self):
        torch.manual_seed(42)
        attention = CausalAttentionReference.Config().build()
        q = torch.randn(8, 4, 16, dtype=torch.float64, requires_grad=True)
        k = torch.randn(8, 2, 16, dtype=torch.float64, requires_grad=True)
        v = torch.randn(8, 2, 12, dtype=torch.float64, requires_grad=True)
        out = attention(q, k, v)
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                is_causal=True,
                enable_gqa=True,
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        torch.testing.assert_close(out, expected)
        self.assertEqual(out.shape, (8, 4, 12))
        torch.testing.assert_close(out[0], v[0].repeat_interleave(2, dim=0))
        out[:3].sum().backward()
        for x in (q, k, v):
            assert x.grad is not None
            self.assertEqual(torch.count_nonzero(x.grad[3:]).item(), 0)
