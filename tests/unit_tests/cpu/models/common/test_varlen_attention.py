# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffix legend:
#   T = packed tokens, H = attention heads, K = query/key head dimension,
#   V = value head dimension, D = model dimension

import importlib
import unittest
from unittest.mock import patch

import pytest

import spmd_types as spmd
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    GQAttention,
    QKVLinear,
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import ComplexRoPE


@pytest.mark.parametrize("model_name", ["deepseek_v3", "kimi_k3"])
@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
@pytest.mark.parametrize("backend_active", [False, True])
@pytest.mark.parametrize("attn_backend", ["varlen", "flex"])
def test_mla_construction_checks_backend(
    monkeypatch, model_name, capability, backend_active, attn_backend
):
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_config = model_module.build_model_config(
        "debugmodel", attn_backend=attn_backend, parallelism=ParallelismConfig()
    )
    attention_config = model_config.first_full_attention
    assert attention_config is not None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    impl = {9: "FA3", 10: "FA4"}[capability[0]] if backend_active else None
    monkeypatch.setattr("torch.nn.attention.current_flash_attention_impl", lambda: impl)

    with torch.device("meta"):
        if backend_active or attn_backend == "flex":
            attention_config.build()
        else:
            with pytest.raises(ValueError, match="FA2 requires equal"):
                attention_config.build()


@pytest.mark.parametrize("model_name", ["llama3", "qwen3", "qwen3_5", "gpt_oss"])
def test_equal_head_model_constructs_without_fa3(monkeypatch, model_name):
    model_module = importlib.import_module(f"torchtitan.models.{model_name}")
    model_config = model_module.build_model_config(
        "debugmodel", attn_backend="varlen", parallelism=ParallelismConfig()
    )
    attention_config = model_config.first_full_attention
    assert attention_config is not None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("torch.nn.attention.current_flash_attention_impl", lambda: None)
    with torch.device("meta"):
        attention = attention_config.build()
    assert isinstance(attention.inner_attention, VarlenInnerAttention)


def test_ulysses_varlen_validates_head_dims(monkeypatch):
    from torchtitan.models.common.attention.cp_attention import (
        UlyssesVarlenInnerAttention,
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr("torch.nn.attention.current_flash_attention_impl", lambda: None)
    attention = UlyssesVarlenInnerAttention.Config().build()
    with pytest.raises(ValueError, match="FA2 requires equal"):
        attention.validate_head_dims(192, v_head_dim=128)


class TestPackedVarlenAttentionMetadata(unittest.TestCase):
    def test_spmd_annotation_includes_partition_spec(self):
        metadata = VarlenAttentionMetadata(
            cu_seq_q=torch.tensor([0, 2], dtype=torch.int32),
            cu_seq_k=torch.tensor([0, 3], dtype=torch.int32),
            max_q=2,
            max_k=3,
        )
        expected_type = spmd.SpmdType(
            {
                MeshAxisName.DP: spmd.V,
                MeshAxisName.TP: spmd.R,
            },
            partition_spec=spmd.PartitionSpec(MeshAxisName.DP),
        )

        with patch(
            "torchtitan.models.common.attention.attention.spmd.assert_type"
        ) as assert_type:
            metadata.annotate_spmd_types()

        self.assertEqual(assert_type.call_count, 2)
        self.assertIs(assert_type.call_args_list[0].args[0], metadata.cu_seq_q)
        self.assertIs(assert_type.call_args_list[1].args[0], metadata.cu_seq_k)
        self.assertEqual(assert_type.call_args_list[0].args[1], expected_type)
        self.assertEqual(assert_type.call_args_list[1].args[1], expected_type)

    def test_document_boundaries(self):
        positions_T = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
        metadata = create_varlen_metadata_for_document(positions_T)

        expected_cu_seq = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
        torch.testing.assert_close(metadata.cu_seq_q, expected_cu_seq)
        torch.testing.assert_close(metadata.cu_seq_k, expected_cu_seq)
        self.assertEqual(metadata.max_q, 4)
        self.assertEqual(metadata.max_k, 4)

    def test_document_cap_produces_fixed_shape_metadata(self):
        three_documents = create_varlen_metadata_for_document(
            torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3]),
            max_num_documents=5,
            max_context_length=4,
        )
        two_documents = create_varlen_metadata_for_document(
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4]),
            max_num_documents=5,
            max_context_length=5,
        )

        torch.testing.assert_close(
            three_documents.cu_seq_q,
            torch.tensor([0, 3, 5, 9, 9, 9], dtype=torch.int32),
        )
        torch.testing.assert_close(
            two_documents.cu_seq_q,
            torch.tensor([0, 4, 9, 9, 9, 9], dtype=torch.int32),
        )
        self.assertEqual(three_documents.cu_seq_q.shape, two_documents.cu_seq_q.shape)
        self.assertEqual(three_documents.max_q, 4)
        self.assertEqual(two_documents.max_q, 5)

    def test_document_cap_reserves_padding_segments_separately(self):
        metadata = create_varlen_metadata_for_document(
            torch.tensor([0, 1, 0, 1, 0, 1, 2, 3]),
            padding_mask=torch.tensor(
                [False, False, False, False, True, True, True, True]
            ),
            max_num_documents=2,
            max_context_length=4,
        )

        torch.testing.assert_close(
            metadata.cu_seq_q,
            torch.tensor([0, 2, 4, 8, 8], dtype=torch.int32),
        )
        self.assertEqual(metadata.max_q, 4)


class TestPackedVarlenInnerAttention(unittest.TestCase):
    def test_preserves_fp16_and_bfloat16_inputs(self):
        attention = VarlenInnerAttention.Config().build()
        metadata = create_varlen_metadata_for_document(torch.arange(4))
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                inputs = [torch.randn(4, 2, 8, dtype=dtype) for _ in range(3)]
                with patch(
                    "torchtitan.models.common.attention.attention._varlen_attn",
                    return_value=inputs[2],
                ) as kernel:
                    output = attention(*inputs, attention_metadata=metadata)

                for actual, expected in zip(kernel.call_args.args[:3], inputs):
                    self.assertIs(actual, expected)
                self.assertIs(output, inputs[2])

    def test_upcast_queries_and_keys_use_value_dtype(self):
        attention = VarlenInnerAttention.Config().build()
        metadata = create_varlen_metadata_for_document(torch.arange(4))
        q_THK, k_THK = [torch.randn(4, 2, 8) for _ in range(2)]
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                v_THV = torch.randn(4, 2, 8, dtype=dtype)
                with patch(
                    "torchtitan.models.common.attention.attention._varlen_attn",
                    return_value=v_THV,
                ) as kernel:
                    output = attention(q_THK, k_THK, v_THV, attention_metadata=metadata)

                actual_q, actual_k, actual_v = kernel.call_args.args[:3]
                torch.testing.assert_close(actual_q, q_THK.to(dtype))
                torch.testing.assert_close(actual_k, k_THK.to(dtype))
                self.assertIs(actual_v, v_THV)
                self.assertEqual(output.dtype, q_THK.dtype)

    def test_fp32_inputs_preserve_bfloat16_fallback(self):
        attention = VarlenInnerAttention.Config().build()
        metadata = create_varlen_metadata_for_document(torch.arange(4))
        inputs = [torch.randn(4, 2, 8) for _ in range(3)]
        with patch(
            "torchtitan.models.common.attention.attention._varlen_attn",
            return_value=inputs[2].bfloat16(),
        ) as kernel:
            output = attention(*inputs, attention_metadata=metadata)

        for actual, expected in zip(kernel.call_args.args[:3], inputs):
            torch.testing.assert_close(actual, expected.bfloat16())
        torch.testing.assert_close(output, inputs[2].bfloat16().float())

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
                n_heads=num_heads,
                n_kv_heads=num_heads,
                wqkv=Linear.Config(in_features=dim, out_features=3 * dim),
            ),
            wo=Linear.Config(in_features=dim, out_features=dim),
            inner_attention=VarlenInnerAttention.Config(),
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
            "torchtitan.models.common.attention.attention._varlen_attn",
            side_effect=_identity_varlen,
        ):
            out_TD = attention(x_TD, metadata, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_thk_thv_sharding_uses_varlen_argument_names(self):
        from torchtitan.models.llama3 import build_model_config

        model_config = build_model_config(
            "debugmodel",
            attn_backend="varlen",
            parallelism=ParallelismConfig(enable_sequence_parallel=False),
        )

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
        self.assertEqual(_per_axis_types(k_dst_layout)[MeshAxisName.CP], spmd.S(0))
        self.assertEqual(
            _per_axis_types(k_dst_layout),
            _per_axis_types((sharding.in_src_shardings or {})["k_THK"]),
        )

    def test_out_transform_receives_th_lse(self):
        num_tokens, num_heads, head_dim = 5, 2, 4
        q_THK = torch.randn(num_tokens, num_heads, head_dim)
        positions_T = torch.tensor([0, 1, 0, 1, 2])
        metadata = create_varlen_metadata_for_document(positions_T)
        inner_attention = VarlenInnerAttention.Config().build()

        def _varlen_with_lse(q, k, v, *args, **kwargs):
            lse_HT = torch.randn(num_heads, num_tokens)
            return q, lse_HT

        def _check_shapes(out_THV, lse_TH):
            self.assertEqual(out_THV.shape, q_THK.shape)
            self.assertEqual(lse_TH.shape, (num_tokens, num_heads))
            return out_THV

        with patch(
            "torchtitan.models.common.attention.attention._varlen_attn",
            side_effect=_varlen_with_lse,
        ):
            out_THV = inner_attention(
                q_THK,
                q_THK,
                q_THK,
                attention_metadata=metadata,
                out_transform=_check_shapes,
            )

        self.assertEqual(out_THV.shape, q_THK.shape)

    def test_llama_decoder_preserves_td_shape(self):
        from torchtitan.models.llama3 import MODEL_FLAVORS

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
        model = build_config(
            "varlen", seq_len=max_context_length, parallelism=ParallelismConfig()
        ).build()
        model.bfloat16()
        model.init_states()
        num_tokens = 6
        tokens_T = torch.randint(0, 2048, (num_tokens,))
        positions_T = torch.tensor([0, 1, 0, 1, 2, 3])
        metadata = model._get_attention_metadata(positions_T)

        def _identity_varlen(q_THK, k_THK, v_THV, *args, **kwargs):
            return q_THK

        with patch(
            "torchtitan.models.common.attention.attention._varlen_attn",
            side_effect=_identity_varlen,
        ):
            logits_TV = model(tokens_T, positions_T, metadata)

        self.assertEqual(logits_TV.shape, (num_tokens, 2048))


if __name__ == "__main__":
    unittest.main()
