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
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import _per_axis_types
from torchtitan.models.common.attention import (
    create_varlen_metadata_for_document,
    GQAttention,
    QKVLinear,
    VarlenInnerAttention,
    VarlenMetadata,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rope import ComplexRoPE


def _build_fixed_rows_attention() -> VarlenInnerAttention:
    with patch(
        "torchtitan.tools.utils.get_cuda_flash_attention_impl",
        return_value=None,
    ):
        return VarlenInnerAttention.Config(fixed_length_rows=True).build()


class TestPackedVarlenMetadata(unittest.TestCase):
    def test_spmd_annotation_includes_partition_spec(self):
        metadata = VarlenMetadata(
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
            "torchtitan.models.common.attention.spmd.assert_type"
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

    def test_fixed_length_rows_have_intrinsically_fixed_metadata(self):
        metadata = create_varlen_metadata_for_document(
            torch.arange(4).repeat(2),
            max_num_documents=8,
            max_context_length=4,
            fixed_length_rows=True,
        )

        torch.testing.assert_close(
            metadata.cu_seq_q,
            torch.tensor([0, 4, 8], dtype=torch.int32),
        )
        self.assertEqual(metadata.max_q, 4)


class TestPackedVarlenInnerAttention(unittest.TestCase):
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
            "torchtitan.models.common.attention._varlen_attn",
            side_effect=_identity_varlen,
        ):
            out_TD = attention(x_TD, metadata, positions_T)

        self.assertEqual(out_TD.shape, x_TD.shape)

    def test_thk_thv_sharding_uses_varlen_argument_names(self):
        from torchtitan.models.llama3 import MODEL_FLAVORS
        from torchtitan.models.llama3.sharding import set_llama3_sharding_config

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
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

    def test_fixed_rows_use_dense_sdpa_with_distinct_value_dimension(self):
        torch.manual_seed(42)
        num_sequences, sequence_length = 2, 4
        q_THK = torch.randn(8, 4, 5, requires_grad=True)
        k_THK = torch.randn(8, 2, 5, requires_grad=True)
        v_THV = torch.randn(8, 2, 3, requires_grad=True)
        positions_T = torch.arange(sequence_length).repeat(num_sequences)
        metadata = create_varlen_metadata_for_document(
            positions_T,
            max_num_documents=num_sequences,
            max_context_length=sequence_length,
            fixed_length_rows=True,
        )
        inner_attention = _build_fixed_rows_attention()

        out_THV = inner_attention(
            q_THK,
            k_THK,
            v_THV,
            attention_masks=metadata,
            scale=0.25,
            enable_gqa=True,
        )

        q_ref_THK = q_THK.detach().clone().requires_grad_()
        k_ref_THK = k_THK.detach().clone().requires_grad_()
        v_ref_THV = v_THV.detach().clone().requires_grad_()
        with sdpa_kernel(SDPBackend.MATH):
            expected_BHLV = F.scaled_dot_product_attention(
                q_ref_THK.reshape(2, 4, 4, 5).transpose(1, 2).to(torch.bfloat16),
                k_ref_THK.reshape(2, 4, 2, 5).transpose(1, 2).to(torch.bfloat16),
                v_ref_THV.reshape(2, 4, 2, 3).transpose(1, 2).to(torch.bfloat16),
                is_causal=True,
                scale=0.25,
                enable_gqa=True,
            )
        expected_THV = expected_BHLV.transpose(1, 2).reshape(8, 4, 3).float()
        torch.testing.assert_close(out_THV, expected_THV)

        out_THV.square().sum().backward()
        expected_THV.square().sum().backward()
        for actual, expected in (
            (q_THK.grad, q_ref_THK.grad),
            (k_THK.grad, k_ref_THK.grad),
            (v_THV.grad, v_ref_THV.grad),
        ):
            self.assertIsNotNone(actual)
            torch.testing.assert_close(actual, expected)

    def test_fixed_rows_reject_nonuniform_metadata(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "fixed_length_rows requires positions to reset only at row boundaries",
        ):
            create_varlen_metadata_for_document(
                torch.tensor([0, 1, 0, 1, 2, 3]),
                max_num_documents=2,
                max_context_length=3,
                fixed_length_rows=True,
            )

    def test_fixed_attention_rejects_unvalidated_metadata(self):
        metadata = create_varlen_metadata_for_document(
            torch.arange(3).repeat(2),
            max_num_documents=2,
            max_context_length=3,
        )
        q_THK = torch.randn(6, 2, 4)
        inner_attention = _build_fixed_rows_attention()

        with self.assertRaisesRegex(
            ValueError,
            "requires metadata constructed for fixed rows",
        ):
            inner_attention(
                q_THK,
                q_THK,
                q_THK,
                attention_masks=metadata,
            )

    def test_fixed_rows_trace_as_one_full_graph(self):
        torch.manual_seed(42)
        q_THK = torch.randn(8, 2, 4, requires_grad=True)
        k_THK = torch.randn(8, 2, 4, requires_grad=True)
        v_THV = torch.randn(8, 2, 3, requires_grad=True)
        metadata = create_varlen_metadata_for_document(
            torch.arange(4).repeat(2),
            max_num_documents=2,
            max_context_length=4,
            fixed_length_rows=True,
        )
        inner_attention = _build_fixed_rows_attention()

        def run_dense(q, k, v):
            return inner_attention(q, k, v, attention_masks=metadata)

        q_ref_THK = q_THK.detach().clone().requires_grad_()
        k_ref_THK = k_THK.detach().clone().requires_grad_()
        v_ref_THV = v_THV.detach().clone().requires_grad_()
        expected = run_dense(q_ref_THK, k_ref_THK, v_ref_THV)
        expected.sum().backward()

        compiled = torch.compile(run_dense, backend="aot_eager", fullgraph=True)
        actual = compiled(q_THK, k_THK, v_THV)
        actual.sum().backward()

        torch.testing.assert_close(actual, expected)
        for actual_grad, expected_grad in (
            (q_THK.grad, q_ref_THK.grad),
            (k_THK.grad, k_ref_THK.grad),
            (v_THV.grad, v_ref_THV.grad),
        ):
            self.assertIsNotNone(actual_grad)
            torch.testing.assert_close(actual_grad, expected_grad)

    def test_fixed_rows_with_output_transform_retain_varlen_path(self):
        num_tokens, num_heads, head_dim = 6, 2, 4
        q_THK = torch.randn(num_tokens, num_heads, head_dim)
        metadata = create_varlen_metadata_for_document(
            torch.arange(3).repeat(2),
            max_num_documents=2,
            max_context_length=3,
            fixed_length_rows=True,
        )
        inner_attention = _build_fixed_rows_attention()

        def _varlen_with_lse(q, k, v, *args, **kwargs):
            return q, torch.randn(num_heads, num_tokens)

        with patch(
            "torchtitan.models.common.attention._varlen_attn",
            side_effect=_varlen_with_lse,
        ) as varlen_mock:
            out_THV = inner_attention(
                q_THK,
                q_THK,
                q_THK,
                attention_masks=metadata,
                out_transform=lambda out, lse: out,
            )

        self.assertEqual(out_THV.shape, q_THK.shape)
        varlen_mock.assert_called_once()

    def test_fixed_rows_require_causal_attention(self):
        with self.assertRaisesRegex(ValueError, "only supports causal"):
            with patch(
                "torchtitan.tools.utils.get_cuda_flash_attention_impl",
                return_value=None,
            ):
                VarlenInnerAttention.Config(
                    fixed_length_rows=True,
                    window_size=(-1, -1),
                ).build()

    def test_decoder_infers_fixed_row_length_for_precompile_inputs(self):
        from torchtitan.models.llama3 import llama3_configs

        build_config, _ = llama3_configs["debugmodel"]
        model_config = build_config("varlen", seq_len=4)
        inner_attention = model_config.first_full_attention_backend
        self.assertIsInstance(inner_attention, VarlenInnerAttention.Config)
        inner_attention.fixed_length_rows = True
        with patch(
            "torchtitan.tools.utils.get_cuda_flash_attention_impl",
            return_value=None,
        ):
            model = model_config.build()

        metadata = model.get_attention_masks(torch.arange(4).repeat(2))

        self.assertIsInstance(metadata, VarlenMetadata)
        self.assertTrue(metadata.fixed_length_rows)
        torch.testing.assert_close(
            metadata.cu_seq_q,
            torch.tensor([0, 4, 8], dtype=torch.int32),
        )

    def test_llama_decoder_preserves_td_shape(self):
        from torchtitan.models.llama3 import MODEL_FLAVORS

        build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
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
