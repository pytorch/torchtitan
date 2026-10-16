# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torchtitan.config import ParallelismConfig
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.muse_glimmer import build_model_config


def cpu_flex_attention(q, k, v, **kwargs):
    """Evaluate the FlexAttention block mask with CPU SDPA."""
    indices = torch.arange(q.shape[-2], device=q.device)
    mask = kwargs["block_mask"].mask_mod(
        torch.tensor(0, device=q.device),
        torch.tensor(0, device=q.device),
        indices[:, None],
        indices[None, :],
    )
    return (
        F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            scale=kwargs["scale"],
            enable_gqa=kwargs["enable_gqa"],
        ),
        SimpleNamespace(),
    )


class TestMuseGlimmerConditionalVision(unittest.TestCase):
    def test_image_grid_rejects_mismatched_patch_count(self):
        model = build_model_config(
            "debugmodel_mm", seq_len=8, parallelism=ParallelismConfig()
        ).build()
        model.init_states()
        assert model.vision_encoder is not None
        for num_patches in (3, 5):
            with self.subTest(num_patches=num_patches):
                pixels = torch.randn(
                    num_patches, model.vision_encoder.conv1_linear.in_features
                )
                with self.assertRaisesRegex(ValueError, "describes 4 patches"):
                    model._get_vision_features(pixels, torch.tensor([[1, 2, 2]]))

    def test_direct_forward_rejects_temporal_image_grid(self):
        model = build_model_config(
            "debugmodel_mm", seq_len=8, parallelism=ParallelismConfig()
        ).build()
        model.init_states()
        assert model.vision_encoder is not None
        pixels = torch.randn(8, model.vision_encoder.conv1_linear.in_features)
        with self.assertRaisesRegex(ValueError, "grid_thw"):
            model(
                torch.zeros(4, dtype=torch.long),
                pixel_values=pixels,
                grid_thw=torch.tensor([[2, 2, 2]]),
                vision_bank_indices_T=torch.zeros(4, dtype=torch.long),
            )

    def test_image_features_forward_and_backward(self):
        model = build_model_config(
            "debugmodel_mm", seq_len=8, parallelism=ParallelismConfig()
        ).build()
        model.init_states()
        assert model.vision_encoder is not None
        pixels = torch.randn(
            4, model.vision_encoder.conv1_linear.in_features, requires_grad=True
        )
        grid = torch.tensor([[1, 2, 2]])
        with patch.object(FlexInnerAttention, "compiled_flex_attn", cpu_flex_attention):
            features = model._get_vision_features(pixels, grid)
            features.sum().backward()
        self.assertEqual(features.shape, (1, model.config.dim))
        self.assertTrue(torch.isfinite(features).all())
        self.assertIsNotNone(pixels.grad)
        self.assertTrue(torch.isfinite(pixels.grad).all())

    def test_built_config_shards_owned_vision_modules(self):
        config = build_model_config(
            "debugmodel_mm",
            seq_len=8,
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                context_parallel_degree=2,
                enable_sequence_parallel=True,
            ),
        )
        assert config.vision_encoder is not None
        assert config.vision_adapter is not None

        self.assertIsNotNone(config.vision_encoder.sharding_config)
        self.assertIsNotNone(config.vision_encoder.conv1.sharding_config)
        self.assertIsNotNone(config.vision_encoder.block.attn.wq.sharding_config)
        self.assertIsNotNone(config.vision_adapter.c_fc.sharding_config)

    def test_image_free_path_gives_all_vision_parameters_zero_gradients(self):
        model = build_model_config(
            "debugmodel_mm", seq_len=8, parallelism=ParallelismConfig()
        ).build()
        model.init_states()
        hidden_TD = torch.randn(4, model.config.dim, requires_grad=True)
        encoder_outputs = []
        assert model.vision_encoder is not None
        assert model.vision_adapter is not None
        with patch.object(
            FlexInnerAttention,
            "compiled_flex_attn",
            cpu_flex_attention,
        ):
            handle = model.vision_encoder.register_forward_hook(
                lambda _module, _args, output: encoder_outputs.append(output)
            )
            try:
                result_TD = model._prepare_multimodal_embeds(
                    hidden_TD,
                    pixel_values=None,
                    grid_thw=None,
                    vision_bank_indices_T=None,
                )
                result_TD.sum().backward()
            finally:
                handle.remove()

        torch.testing.assert_close(result_TD, hidden_TD, rtol=0, atol=0)
        self.assertEqual(len(encoder_outputs), 1)
        self.assertEqual(
            encoder_outputs[0].shape, (1, model.vision_adapter.c_fc.in_features)
        )
        self.assertTrue(encoder_outputs[0].requires_grad)

        assert model.vision_projection is not None
        assert model.perception_emb_norm is not None
        for module in (
            model.vision_encoder,
            model.vision_adapter,
            model.vision_projection,
            model.perception_emb_norm,
        ):
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))


if __name__ == "__main__":
    unittest.main()
