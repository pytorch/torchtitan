# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffix legend:
#   T = packed tokens, H = attention heads, K = query/key head dimension,
#   V = value head dimension, D = model dimension

import unittest
from unittest.mock import patch

import spmd_types as spmd
import torch

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    GQAttention,
    QKVLinear,
    VarlenAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import ComplexRoPE


class TestPackedVarlenMetadata(unittest.TestCase):
    def test_document_boundaries(self):
        positions_T = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
        metadata = create_varlen_metadata_for_document(positions_T)

        expected_cu_seq = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
        torch.testing.assert_close(metadata.cu_seq_q, expected_cu_seq)
        torch.testing.assert_close(metadata.cu_seq_k, expected_cu_seq)
        self.assertEqual(metadata.max_q, 4)
        self.assertEqual(metadata.max_k, 4)


class TestPackedVarlenAttention(unittest.TestCase):
    def test_gqa_preserves_td_shape(self):
        torch.manual_seed(42)
        num_tokens, dim, num_heads, head_dim = 6, 8, 2, 4
        attention = GQAttention.Config(
            n_heads=num_heads,
            n_kv_heads=num_heads,
            head_dim=head_dim,
            dim=dim,
            qkv_linear=QKVLinear.Config(
                head_dim=head_dim,
                wq=Linear.Config(in_features=dim, out_features=dim),
                wkv=Linear.Config(in_features=dim, out_features=dim),
            ),
            wo=Linear.Config(in_features=dim, out_features=dim),
            inner_attention=VarlenAttention.Config(),
            rope=ComplexRoPE.Config(dim=head_dim, max_context_length=num_tokens),
        ).build()
        x_TD = torch.randn(num_tokens, dim)
        positions_T = torch.tensor([0, 1, 0, 1, 2, 3])
        metadata = create_varlen_metadata_for_document(positions_T)

        def _identity_varlen(q_THK, k_THK, v_THV, *args, **kwargs):
            self.assertEqual(q_THK.ndim, 3)
            self.assertEqual(k_THK.ndim, 3)
            self.assertEqual(v_THV.ndim, 3)
            return q_THK

        with patch(
            "torchtitan.models.common.attention._varlen_attn",
            side_effect=_identity_varlen,
        ):
            out_TD = attention(x_TD, metadata, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_thk_thv_sharding_uses_varlen_argument_names(self):
        from torchtitan.models.llama3 import llama3_configs
        from torchtitan.models.llama3.sharding import set_llama3_sharding_config

        build_config, max_context_length = llama3_configs["debugmodel"]
        model_config = build_config("varlen", seq_len=max_context_length)
        set_llama3_sharding_config(model_config, enable_sp=False)

        sharding = model_config.layers[0].attention.inner_attention.sharding_config
        assert sharding is not None
        self.assertEqual(
            set(sharding.in_src_shardings or {}),
            {"q_THK", "k_THK", "v_THV"},
        )
        q_layout = (sharding.in_src_shardings or {})["q_THK"]
        k_dst_layout = (sharding.in_dst_shardings or {})["k_THK"]
        axis_types = _per_axis_types(q_layout)
        self.assertEqual(axis_types[MeshAxisName.DP], spmd.S(0))
        self.assertEqual(axis_types[MeshAxisName.CP], spmd.S(0))
        self.assertEqual(axis_types[MeshAxisName.TP], spmd.S(1))
        self.assertEqual(_per_axis_types(k_dst_layout)[MeshAxisName.CP], spmd.R)

    def test_out_transform_receives_th_lse(self):
        num_tokens, num_heads, head_dim = 5, 2, 4
        q_THK = torch.randn(num_tokens, num_heads, head_dim)
        positions_T = torch.tensor([0, 1, 0, 1, 2])
        metadata = create_varlen_metadata_for_document(positions_T)
        inner_attention = VarlenAttention.Config().build()

        def _varlen_with_lse(q, k, v, *args, **kwargs):
            lse_HT = torch.randn(num_heads, num_tokens)
            return q, lse_HT

        def _check_shapes(out_THV, lse_TH):
            self.assertEqual(out_THV.shape, q_THK.shape)
            self.assertEqual(lse_TH.shape, (num_tokens, num_heads))
            return out_THV

        with patch(
            "torchtitan.models.common.attention._varlen_attn",
            side_effect=_varlen_with_lse,
        ):
            out_THV = inner_attention(
                q_THK,
                q_THK,
                q_THK,
                attention_masks=metadata,
                out_transform=_check_shapes,
            )

        self.assertEqual(out_THV.shape, q_THK.shape)

    def test_llama_decoder_preserves_td_shape(self):
        from torchtitan.models.llama3 import llama3_configs

        build_config, max_context_length = llama3_configs["debugmodel"]
        model = build_config("varlen", seq_len=max_context_length).build()
        model.init_states()
        num_tokens = 6
        tokens_T = torch.randint(0, 2048, (num_tokens,))
        positions_T = torch.tensor([0, 1, 0, 1, 2, 3])
        metadata = model.get_attention_masks(positions_T)

        def _identity_varlen(q_THK, k_THK, v_THV, *args, **kwargs):
            return q_THK

        with patch(
            "torchtitan.models.common.attention._varlen_attn",
            side_effect=_identity_varlen,
        ):
            logits_TV = model(tokens_T, positions_T, metadata)

        self.assertEqual(logits_TV.shape, (num_tokens, 2048))


if __name__ == "__main__":
    unittest.main()
