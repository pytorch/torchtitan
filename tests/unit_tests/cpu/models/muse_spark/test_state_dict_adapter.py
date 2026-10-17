# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import unittest
from unittest.mock import patch

import torch

from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.muse_spark import build_model_config
from torchtitan.models.muse_spark.state_dict_adapter import MuseSparkStateDictAdapter


class MuseSparkStateDictAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_model_config("debugmodel_mm", seq_len=8)
        self.adapter = MuseSparkStateDictAdapter(self.config, hf_assets_path=None)
        self.model = self.config.build()
        self.model.init_states()
        self.state = self.model.state_dict()

    def test_hf_names_and_tensor_layouts(self) -> None:
        hf = self.adapter.to_hf(self.state)
        for checkpoint_key, native_key in (
            (
                "model.language_model.embed_tokens.weight",
                "tok_embeddings.embedding.weight",
            ),
            ("model.language_model.norm.weight", "norm.weight"),
            ("lm_head.weight", "lm_head.weight"),
            (
                "model.language_model.layers.0.input_layernorm.weight",
                "layers.0.attention_norm.weight",
            ),
            (
                "model.language_model.layers.0.mlp.gate.weight",
                "layers.0.feed_forward.router.gate.weight",
            ),
            (
                "model.vision_tower.patch_embedder.position_embedding_table.weight",
                "vision_encoder.positional_embedding_vlm",
            ),
            ("model.vision_adapter.fc1.weight", "vision_adapter.c_fc.weight"),
            ("model.vision_projection.weight", "vision_projection.weight"),
            (
                "model.audio_tower.layers.0.patch_embedder.proj.weight",
                "audio_encoder.layers.0.patchifier.linear.weight",
            ),
            (
                "model.audio_tower.layers.0.conv_blocks.0.conv.weight",
                "audio_encoder.layers.0.block1.conv.conv.weight",
            ),
            ("model.audio_tower.proj.weight", "audio_encoder.o_proj.weight"),
            (
                "model.audio_tower.post_proj_norm.weight",
                "audio_encoder.dual_norm.weight",
            ),
        ):
            with self.subTest(checkpoint_key=checkpoint_key):
                torch.testing.assert_close(
                    hf[checkpoint_key], self.state[native_key], rtol=0, atol=0
                )

        attention = self.config.layers[0].attention
        native_state = self.adapter._native_fused_linears_to_hf(self.state)
        for checkpoint_key, native_key, num_heads in (
            (
                "model.language_model.layers.0.self_attn.q_proj.weight",
                "layers.0.attention.qkv_linear.wq.weight",
                attention.n_heads,
            ),
            (
                "model.language_model.layers.0.self_attn.k_proj.weight",
                "layers.0.attention.qkv_linear.wk.weight",
                attention.n_kv_heads,
            ),
            (
                "model.vision_tower.layers.0.attn.q_proj.bias",
                "vision_encoder.layers.0.attn.wq.bias",
                self.config.vision_encoder.num_heads,
            ),
        ):
            native = native_state[native_key]
            expected = native.reshape(num_heads, -1, 2, *native.shape[1:])
            expected = expected.transpose(1, 2).reshape(native.shape)
            torch.testing.assert_close(hf[checkpoint_key], expected, rtol=0, atol=0)

        w13 = self.state["layers.0.feed_forward.routed_experts.w13.weight"]
        torch.testing.assert_close(
            hf["model.language_model.layers.0.mlp.experts.gate_up_proj"],
            torch.cat((w13[:, 0], w13[:, 1]), dim=1),
            rtol=0,
            atol=0,
        )

    def test_checkpoint_roundtrip(self) -> None:
        for dense_text_only in (False, True):
            with self.subTest(dense_text_only=dense_text_only):
                config = self.config
                if dense_text_only:
                    config = build_model_config("debugmodel", seq_len=8)
                    config.vision_encoder = config.vision_adapter = None
                    config.vision_projection = config.perception_emb_norm = None
                    config.audio_encoder = None
                    config.layers[0].feed_forward = DenseFeedForward.Config(
                        w13=Linear.Config(
                            in_features=config.dim, out_features=64, num_linears=2
                        ),
                        w2=Linear.Config(in_features=64, out_features=config.dim),
                    )
                model = config.build()
                model.init_states()
                state = model.state_dict()
                adapter = MuseSparkStateDictAdapter(config, hf_assets_path=None)
                restored = adapter.from_hf(adapter.to_hf(state))
                torch.testing.assert_close(restored, state, rtol=0, atol=0)
                model.load_state_dict(restored, strict=True)

    def test_packed_weights_require_matching_metadata_dtypes(self) -> None:
        metadata = self.adapter._native_metadata.copy()
        key = "layers.0.feed_forward.routed_experts.w3_EFD"
        metadata[key] = dataclasses.replace(metadata[key], dtype=torch.bfloat16)
        with patch.object(
            MuseSparkStateDictAdapter, "_build_native_metadata", return_value=metadata
        ):
            with self.assertRaisesRegex(ValueError, "compatible rank-3 weights"):
                MuseSparkStateDictAdapter(self.config, hf_assets_path=None)

    def test_packed_weights_validate_input_dtypes(self) -> None:
        state = self.adapter._native_fused_linears_to_hf(
            self.state, split_routed_experts=True
        )
        key = "layers.0.feed_forward.routed_experts.w3_EFD"
        state[key] = state[key].to(torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "must use the same dtype"):
            self.adapter.to_hf(state)
        key = "layers.0.feed_forward.routed_experts.w1_EFD"
        state[key] = state[key].to(torch.bfloat16)
        hf = self.adapter.to_hf(state)
        self.assertEqual(
            hf["model.language_model.layers.0.mlp.experts.gate_up_proj"].dtype,
            torch.bfloat16,
        )
