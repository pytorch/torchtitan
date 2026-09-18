# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import torchtitan.models.muse_glimmer.parallelize as parallelize_module
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.models.muse_glimmer import (
    model_registry,
    muse_glimmer_vision_encoder_config,
    parallelize_muse_glimmer,
)


class TestMuseGlimmerConditionalVision(unittest.TestCase):
    def _build_vision_encoder(self):
        return muse_glimmer_vision_encoder_config(
            latent_dim=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2.0,
            patch_size=2,
            patch_temporal=1,
            downsample_factor=2,
            sparse_attention_factor=1,
            pos_emb_grid_h=2,
            pos_emb_grid_w=2,
        ).build()

    def test_empty_encoder_output_carries_autograd(self):
        encoder = self._build_vision_encoder()

        output = encoder(None, grid_thw=None)

        self.assertEqual(output.shape, (0, encoder.output_dim))
        self.assertEqual(output.dtype, encoder.conv1_linear.weight.dtype)
        self.assertEqual(output.device, encoder.conv1_linear.weight.device)
        self.assertTrue(output.requires_grad)

        with torch.no_grad():
            inference_output = encoder(None, grid_thw=None)
        self.assertFalse(inference_output.requires_grad)

    def test_image_free_path_runs_vision_modules_without_changing_embeddings(self):
        model = model_registry("debugmodel_mm", seq_len=8).model.build()
        hidden_TD = torch.randn(4, model.config.dim, requires_grad=True)
        encoder_outputs = []
        assert model.vision_encoder is not None
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
        finally:
            handle.remove()

        torch.testing.assert_close(result_TD, hidden_TD, rtol=0, atol=0)
        self.assertEqual(len(encoder_outputs), 1)
        self.assertTrue(encoder_outputs[0].requires_grad)

        result_TD.sum().backward()
        assert model.vision_adapter is not None
        assert model.vision_projection is not None
        assert model.perception_emb_norm is not None
        for module in (
            model.vision_adapter,
            model.vision_projection,
            model.perception_emb_norm,
        ):
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))

    def test_multimodal_parallelize_enables_unused_parameter_reduction(self):
        class FakeFSDPModule:
            def __init__(self):
                self.calls = []

            def set_reduce_scatter_unused_params(self, enabled, *, recurse):
                self.calls.append((enabled, recurse))

        class FakeModel(FakeFSDPModule):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(vision_encoder=object())
                self.vision_encoder = FakeFSDPModule()
                self.vision_adapter = FakeFSDPModule()

            def parallelize(self, _parallel_dims):
                pass

        model = FakeModel()
        parallel_dims = SimpleNamespace(tp_enabled=False, pp_enabled=False)
        with (
            patch.object(parallelize_module, "FSDPModule", FakeFSDPModule, create=True),
            patch.object(
                parallelize_module,
                "resolve_fsdp_mesh",
                return_value=(object(), object()),
            ),
            patch.object(parallelize_module, "apply_fsdp_to_vision_encoder"),
            patch.object(parallelize_module, "apply_fsdp_to_decoder"),
        ):
            parallelize_muse_glimmer(
                model,
                parallel_dims=parallel_dims,
                training=TrainingConfig(disable_cuda_graphs=False),
                parallelism=ParallelismConfig(),
                compile_config=CompileConfig(),
                ac_config=None,
                dump_folder="",
            )

        self.assertEqual(model.calls, [(True, False)])
        self.assertEqual(model.vision_encoder.calls, [(True, False)])
        self.assertEqual(model.vision_adapter.calls, [(True, False)])


if __name__ == "__main__":
    unittest.main()
