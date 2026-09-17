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
from torchtitan.components.optimizer import default_adamw
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

    def test_get_vision_features_returns_model_dimension_bank(self):
        model_config = model_registry("debugmodel_mm", seq_len=8).model
        assert model_config.vision_encoder is not None
        model_config.vision_encoder.num_layers = 0
        model = model_config.build()
        with torch.no_grad():
            model.init_weights()
        assert model.vision_encoder is not None
        patch_dim = model.vision_encoder.conv1_linear.in_features
        pixel_values = torch.randn(4, patch_dim)
        grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.int64)

        vision_bank_VD = model._get_vision_features(pixel_values, grid_thw)

        self.assertEqual(vision_bank_VD.shape, (1, model.config.dim))
        vision_bank_VD.sum().backward()
        vision_modules = (
            model.vision_encoder,
            model.vision_adapter,
            model.vision_projection,
            model.perception_emb_norm,
        )
        for module in vision_modules:
            assert module is not None
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)

    def test_only_training_preprocessing_records_image_activity(self):
        model = model_registry("debugmodel_mm", seq_len=8).model.build()
        parallel_dims = SimpleNamespace(cp_enabled=False, tp_enabled=False)
        input_dict = {
            "input": torch.tensor([7, 1]),
            "labels": torch.tensor([7, 1]),
            "pixel_values": torch.empty(0, 12),
            "grid_thw": torch.empty(0, 3, dtype=torch.int64),
            "special_tokens": {"image_id": 7},
        }

        with patch(
            "torchtitan.models.muse_glimmer.model.annotate_input_spmd_types",
            side_effect=lambda _parallel_dims, batch, _sharding: batch,
        ):
            model.train()
            model.preprocess_inputs(
                input_dict,
                parallel_dims=parallel_dims,
                parallelism=ParallelismConfig(),
            )
            model.preprocess_inputs(
                {
                    "input": torch.tensor([1, 1]),
                    "labels": torch.tensor([1, 1]),
                },
                parallel_dims=parallel_dims,
                parallelism=ParallelismConfig(),
            )
            self.assertTrue(model._consume_vision_activity())
            self.assertFalse(model._consume_vision_activity())

            model.eval()
            model.preprocess_inputs(
                input_dict,
                parallel_dims=parallel_dims,
                parallelism=ParallelismConfig(),
            )
            self.assertFalse(model._consume_vision_activity())

    def test_globally_inactive_step_does_not_decay_vision_parameters(self):
        model_spec = model_registry("debugmodel_mm", seq_len=8)
        model = model_spec.model.build()
        with torch.no_grad():
            model.init_weights()
        optimizer_config = default_adamw(lr=0.1)
        optimizer_config.implementation = "for-loop"
        optimizers = optimizer_config.build(model_parts=[model])
        assert model_spec.post_optimizer_build_fn is not None
        parallel_dims = SimpleNamespace(get_optional_mesh=lambda _name: None)
        model_spec.post_optimizer_build_fn(optimizers, [model], parallel_dims)

        vision_modules = (
            model.vision_encoder,
            model.vision_adapter,
            model.vision_projection,
            model.perception_emb_norm,
        )
        vision_parameters = [
            parameter
            for module in vision_modules
            if module is not None
            for parameter in module.parameters()
        ]
        for parameter in vision_parameters:
            parameter.grad = torch.zeros_like(parameter)
        weight_before_step = vision_parameters[0].detach().clone()

        optimizers.step()

        torch.testing.assert_close(vision_parameters[0], weight_before_step)
        self.assertTrue(all(parameter.grad is None for parameter in vision_parameters))

    def test_text_only_model_does_not_register_a_conditional_optimizer_hook(self):
        self.assertIsNone(
            model_registry("debugmodel", seq_len=8).post_optimizer_build_fn
        )

    def test_partial_vision_ownership_is_rejected(self):
        model_spec = model_registry("debugmodel_mm", seq_len=8)
        model = model_spec.model.build()
        optimizers = default_adamw(lr=0.1).build(model_parts=[model])
        model.vision_adapter = None
        assert model_spec.post_optimizer_build_fn is not None

        with self.assertRaisesRegex(ValueError, "either all vision modules or none"):
            model_spec.post_optimizer_build_fn(
                optimizers,
                [model],
                SimpleNamespace(get_optional_mesh=lambda _name: None),
            )

    def test_vision_modules_without_token_embeddings_are_rejected(self):
        model_spec = model_registry("debugmodel_mm", seq_len=8)
        model = model_spec.model.build()
        optimizers = default_adamw(lr=0.1).build(model_parts=[model])
        model.tok_embeddings = None
        assert model_spec.post_optimizer_build_fn is not None

        with self.assertRaisesRegex(ValueError, "colocated with token embeddings"):
            model_spec.post_optimizer_build_fn(
                optimizers,
                [model],
                SimpleNamespace(get_optional_mesh=lambda _name: None),
            )

    def test_embedding_stage_without_vision_modules_is_rejected(self):
        model_spec = model_registry("debugmodel_mm", seq_len=8)
        model = model_spec.model.build()
        optimizers = default_adamw(lr=0.1).build(model_parts=[model])
        model.vision_encoder = None
        model.vision_adapter = None
        model.vision_projection = None
        model.perception_emb_norm = None
        assert model_spec.post_optimizer_build_fn is not None

        with self.assertRaisesRegex(ValueError, "embedding stage must own"):
            model_spec.post_optimizer_build_fn(
                optimizers,
                [model],
                SimpleNamespace(get_optional_mesh=lambda _name: None),
            )

    def test_repeated_local_vision_ownership_is_rejected(self):
        model_spec = model_registry("debugmodel_mm", seq_len=8)
        model = model_spec.model.build()
        optimizers = default_adamw(lr=0.1).build(model_parts=[model])
        assert model_spec.post_optimizer_build_fn is not None

        with self.assertRaisesRegex(ValueError, "exactly one local pipeline stage"):
            model_spec.post_optimizer_build_fn(
                optimizers,
                [model, model],
                SimpleNamespace(get_optional_mesh=lambda _name: None),
            )

    def test_multimodal_parallelize_rejects_cuda_graphs_before_mutation(self):
        model = model_registry("debugmodel_mm", seq_len=8).model.build()
        parallel_dims = SimpleNamespace(tp_enabled=False, pp_enabled=False)
        self.assertFalse(model._parallelized)

        with self.assertRaisesRegex(ValueError, "CUDA graphs"):
            parallelize_muse_glimmer(
                model,
                parallel_dims=parallel_dims,
                training=TrainingConfig(disable_cuda_graphs=False),
                parallelism=ParallelismConfig(),
                compile_config=CompileConfig(),
                ac_config=None,
                dump_folder="",
            )

        self.assertFalse(model._parallelized)

    def test_multimodal_inference_without_dp_does_not_require_cuda_graphs_disabled(
        self,
    ):
        class FakeModel:
            def __init__(self):
                self.config = SimpleNamespace(vision_encoder=object())
                self.vision_encoder = object()
                self.vision_adapter = object()
                self.parallelized = False

            def parallelize(self, _parallel_dims):
                self.parallelized = True

        model = FakeModel()
        result = parallelize_muse_glimmer(
            model,
            parallel_dims=SimpleNamespace(tp_enabled=False, pp_enabled=False),
            training=TrainingConfig(disable_cuda_graphs=False),
            parallelism=ParallelismConfig(),
            compile_config=CompileConfig(),
            ac_config=None,
            dump_folder="",
            skip_dp=True,
        )

        self.assertIs(result, model)
        self.assertTrue(model.parallelized)

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
                training=TrainingConfig(disable_cuda_graphs=True),
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
