# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffix legend:
#   T = packed tokens, H = attention heads, K = query/key head dimension,
#   V = value head dimension, D = model dimension

import unittest
from copy import deepcopy
from unittest.mock import patch

import spmd_types as spmd
import torch
from torchtitan.components.validator import Validator
from torchtitan.config.override import apply_overrides, OverrideConfig

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
from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan.models.deepseek_v3.model import Attention
from torchtitan_recipes.overrides.dense_causal_sdpa import (
    dense_causal_bf16_sdpa,
    DenseCausalBf16SDPA,
)
from torchtitan_recipes.overrides.fused_mla import FusedMLAAttention
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_debugmodel_mtp


class TestDenseCausalSDPAOverride(unittest.TestCase):
    def test_continuous_row_metadata(self):
        metadata = DenseCausalBf16SDPA.Config().build_attention_metadata(
            torch.arange(7, 11), padding_mask=torch.zeros(4, dtype=torch.bool)
        )
        torch.testing.assert_close(
            metadata.cu_seq_q, torch.tensor([0, 4], dtype=torch.int32)
        )
        self.assertIs(metadata.cu_seq_q, metadata.cu_seq_k)
        self.assertEqual(metadata.max_q, 4)

    def test_packed_positions_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "enable unmasked_text_rows"):
            DenseCausalBf16SDPA.Config().build_attention_metadata(
                torch.tensor([0, 1, 0, 1])
            )

    def test_padding_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "does not support padding"):
            DenseCausalBf16SDPA.Config().build_attention_metadata(
                torch.arange(4), padding_mask=torch.tensor([False, False, False, True])
            )

    def test_noncausal_windows_are_rejected(self):
        for window in ((-1, -1), (2, 0)):
            with self.subTest(window=window), self.assertRaisesRegex(
                ValueError, "plain causal"
            ):
                dense_causal_bf16_sdpa(VarlenInnerAttention.Config(window_size=window))

    def test_override_preserves_shared_config_fields(self):
        sharding = object()
        config = dense_causal_bf16_sdpa(
            VarlenInnerAttention.Config(sharding_config=sharding)
        )
        self.assertIs(config.sharding_config, sharding)
        self.assertEqual(config.window_size, (-1, 0))

    def test_data_and_attention_overrides_cover_training_validation_and_mtp(self):
        imports = [
            "torchtitan_recipes.overrides.dense_causal_sdpa.dense_causal_fused_mla",
            "torchtitan_recipes.overrides.dense_causal_sdpa.unmasked_text_rows",
        ]
        for targets in (imports, list(reversed(imports))):
            with self.subTest(targets=targets):
                config = deepseek_v3_debugmodel_mtp(seq_len=8)
                config.model = build_model_config(
                    "debugmodel",
                    seq_len=8,
                    num_mtp_layers=1,
                    attn_backend="varlen",
                    parallelism=config.parallelism,
                )
                config.validator = Validator.Config(
                    dataloader=deepcopy(config.dataloader), steps=1
                )
                apply_overrides(OverrideConfig(imports=targets), config)
                self.assertFalse(config.dataloader.collator.mask_document_boundaries)
                self.assertFalse(
                    config.validator.dataloader.collator.mask_document_boundaries
                )
                attention_configs = [
                    attention
                    for _, attention, _, _ in config.model.traverse(Attention.Config)
                ]
                self.assertGreater(len(attention_configs), len(config.model.layers))
                for attention in attention_configs:
                    self.assertIsInstance(attention, FusedMLAAttention.Config)
                    self.assertIsInstance(
                        attention.inner_attention, DenseCausalBf16SDPA.Config
                    )

    def test_separate_dense_and_fused_mla_overrides_remain_rejected(self):
        config = deepseek_v3_debugmodel_mtp(seq_len=8)
        config.model = build_model_config(
            "debugmodel",
            seq_len=8,
            num_mtp_layers=1,
            attn_backend="varlen",
            parallelism=config.parallelism,
        )
        with self.assertRaisesRegex(ValueError, "Nested overrides"):
            apply_overrides(
                OverrideConfig(
                    imports=[
                        "torchtitan_recipes.overrides.dense_causal_sdpa.dense_causal_bf16_sdpa",
                        "torchtitan_recipes.overrides.fused_mla.fused_mla",
                    ]
                ),
                config,
            )

    def test_dense_override_does_not_silently_skip_flex_with_data_override(self):
        config = deepseek_v3_debugmodel_mtp(seq_len=8)
        config.model = build_model_config(
            "debugmodel",
            seq_len=8,
            num_mtp_layers=1,
            attn_backend="flex",
            parallelism=config.parallelism,
        )
        with self.assertRaisesRegex(ValueError, "plain causal Varlen"):
            apply_overrides(
                OverrideConfig(
                    imports=[
                        "torchtitan_recipes.overrides.dense_causal_sdpa.dense_causal_bf16_sdpa",
                        "torchtitan_recipes.overrides.dense_causal_sdpa.unmasked_text_rows",
                    ]
                ),
                config,
            )

    def test_dense_graph_metadata_does_not_require_document_cap(self):
        config = deepseek_v3_debugmodel_mtp(seq_len=8)
        config.model = build_model_config(
            "debugmodel",
            seq_len=8,
            num_mtp_layers=1,
            attn_backend="varlen",
            parallelism=config.parallelism,
        )
        config.training.disable_cuda_graphs = False
        config.dataloader.max_num_documents = None
        apply_overrides(
            OverrideConfig(
                imports=[
                    "torchtitan_recipes.overrides.dense_causal_sdpa.dense_causal_bf16_sdpa",
                ]
            ),
            config,
        )
        with patch(
            "torchtitan.distributed.cuda_graph.cuda_graphs_supported", return_value=True
        ):
            config.__post_init__()
        self.assertFalse(config.training.disable_cuda_graphs)
        self.assertIsNone(config.dataloader.max_num_documents)
        metadata = DenseCausalBf16SDPA.Config().build_attention_metadata(
            torch.arange(8)
        )
        self.assertEqual(metadata.cu_seq_q.tolist(), [0, 8])


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
