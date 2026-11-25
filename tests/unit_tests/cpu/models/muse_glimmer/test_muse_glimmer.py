# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.muse_glimmer import build_model_config
from torchtitan.models.muse_glimmer.state_dict_adapter import (
    MuseGlimmerStateDictAdapter,
)


def cpu_varlen_attention(
    q_THK,
    k_THK,
    v_THV,
    cu_seq_q,
    cu_seq_k,
    max_q,
    max_k,
    scale,
    window_size,
    num_splits,
):
    """Full attention within each packed segment with CPU SDPA."""
    cu_seqlens = cu_seq_q.tolist()
    outputs = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        q, k, v = (t[start:end].transpose(0, 1) for t in (q_THK, k_THK, v_THV))
        outputs.append(
            F.scaled_dot_product_attention(q, k, v, scale=scale).transpose(0, 1)
        )
    return (torch.cat(outputs),)


# The NaViT vision attention kernel; it has no CPU implementation.
_VARLEN_ATTN = "torchtitan.models.common.attention.attention.deterministic_varlen_attn"


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
        with patch(_VARLEN_ATTN, cpu_varlen_attention):
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
        self.assertIsNotNone(config.vision_encoder.conv1_linear.sharding_config)
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
        with patch(_VARLEN_ATTN, cpu_varlen_attention):
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


class TestMuseGlimmerStateDictAdapterRowPermute(unittest.TestCase):
    def test_a_rank_wide_factor_permutes_like_its_projection(self):
        """The q/k RoPE permute reorders rows only, so a LoRA factor with the projection's rows
        and a rank-wide second dim must follow the weight's row order; sizing the k permute from
        the model config instead of the tensor used to fail on such a factor."""
        config = build_model_config(
            "debugmodel", parallelism=ParallelismConfig(), seq_len=64
        )
        adapter = MuseGlimmerStateDictAdapter(config, None)
        attention = config.layers[0].attention
        rows = (attention.n_heads + 2 * attention.n_kv_heads) * attention.head_dim
        rank = 4
        factor = (
            torch.arange(rows, dtype=torch.float32)[:, None]
            .expand(rows, rank)
            .contiguous()
        )
        weight = (
            torch.arange(rows, dtype=torch.float32)[:, None]
            .expand(rows, config.dim)
            .contiguous()
        )

        exported = adapter.to_hf({"layers.0.attention.qkv_linear.wqkv.weight": factor})
        reference = adapter.to_hf({"layers.0.attention.qkv_linear.wqkv.weight": weight})

        self.assertEqual(
            {name: tuple(tensor.shape) for name, tensor in exported.items()},
            {
                f"model.language_model.layers.0.self_attn.{proj}.weight": (out, rank)
                for proj, out in (
                    ("q_proj", attention.n_heads * attention.head_dim),
                    ("k_proj", attention.n_kv_heads * attention.head_dim),
                    ("v_proj", attention.n_kv_heads * attention.head_dim),
                )
            },
        )
        for name, tensor in exported.items():
            torch.testing.assert_close(tensor[:, 0], reference[name][:, 0])


if __name__ == "__main__":
    unittest.main()
