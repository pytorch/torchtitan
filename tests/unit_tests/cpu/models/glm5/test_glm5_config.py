# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import MLAAttention
from torchtitan.models.common.attention.cp_mla import AllGatherKVMLAFlexAttention
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe.token_dispatcher import DeepEPTokenDispatcher
from torchtitan.models.glm5 import build_model_config, MODEL_FLAVORS
from torchtitan.models.glm5.dsa import Glm5CPMLAAttention
from torchtitan.models.glm5.model import Glm5Attention, Glm5Model
from torchtitan.models.glm5.mtp import Glm5MTPTransformerBlock


class TestGlm5Config(unittest.TestCase):
    def test_sparse_attention_flops_saturate_for_shared_consumers(self) -> None:
        config = build_model_config(
            "glm5_2_debugmodel", parallelism=ParallelismConfig(), seq_len=128
        )
        producer = config.layers[2].attention
        consumer = config.layers[3].attention
        assert isinstance(producer, Glm5Attention.Config)
        assert isinstance(consumer, Glm5Attention.Config)
        self.assertIsNone(consumer.indexer)
        self.assertEqual(producer.index_topk, consumer.index_topk)
        for attention in (producer, consumer):
            topk = attention.index_topk
            self.assertEqual(
                attention.flops_per_token(topk), attention.flops_per_token(topk * 8)
            )
            self.assertLess(
                attention.flops_per_token(topk // 2), attention.flops_per_token(topk)
            )

    def test_optimizer_step_rolls_and_clears_aux_metrics(self) -> None:
        from torchtitan.models.common.aux_loss import AuxLoss

        with patch.dict(AuxLoss.group_acc, clear=True), patch.dict(
            AuxLoss._group_counts, clear=True
        ):
            model = torch.nn.Module()
            model.layers = torch.nn.ModuleList()
            model.aux = AuxLoss.Config(coeff=1.0).build()
            parameter = torch.nn.Parameter(torch.ones(1))
            optimizer = torch.optim.SGD([parameter], lr=0.1)
            Glm5Model._register_optimizer_hooks(optimizer, [model], None)
            key = (model.aux.reduce_mesh, model.aux.metric_name)
            for value in (3.0, 7.0):
                model.aux.instance_acc.fill_(value)
                parameter.grad = torch.ones_like(parameter)
                optimizer.step()
                self.assertEqual(model.aux.instance_acc.item(), 0)
                self.assertEqual(AuxLoss.group_acc[key].item(), value)

    def test_all_flavors_build(self) -> None:
        parallelism = ParallelismConfig()
        for flavor in MODEL_FLAVORS:
            with self.subTest(flavor=flavor):
                config = build_model_config(
                    flavor,
                    parallelism=parallelism,
                    seq_len=128,
                )
                self.assertEqual(config.max_context_length, 128)
                self.assertGreater(len(config.layers), 0)
                self.assertIsInstance(config.layers[0].attention, Glm5Attention.Config)

    def test_debug_flavor_preserves_dense_to_moe_transition(self) -> None:
        config = build_model_config(
            "glm5_2_debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        self.assertTrue(
            all(layer.feed_forward is not None for layer in config.layers[:3])
        )
        self.assertTrue(
            all(
                isinstance(layer.feed_forward, MoE.Config)
                for layer in config.layers[3:]
            )
        )

    def test_rejects_context_beyond_flavor_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds max context length"):
            build_model_config(
                "debugmodel",
                parallelism=ParallelismConfig(),
                seq_len=DEFAULT_DEBUG_MODEL_SEQ_LEN + 1,
            )

    def test_debug_model_builds_and_initializes(self) -> None:
        config = build_model_config(
            "glm5_2_debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=8,
        )
        model = config.build()
        model.init_states()
        self.assertGreater(
            sum(parameter.numel() for parameter in model.parameters()), 0
        )

    def test_mtp_flavor_builds_full_nextn_block(self) -> None:
        config = build_model_config(
            "glm5_2_debugmodel_mtp",
            parallelism=ParallelismConfig(),
            seq_len=8,
        )
        self.assertEqual(len(config.mtp_layers), 1)
        self.assertIsInstance(config.mtp_layers[0], Glm5MTPTransformerBlock.Config)
        self.assertTrue(all(layer.preserve_output_for_mtp for layer in config.layers))
        model = config.build()
        model.init_states()
        self.assertEqual(len(model.mtp_layers), 1)

    def test_released_flavor_matches_official_mtp_depth(self) -> None:
        config = build_model_config(
            "744B-A40B",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        self.assertEqual(len(config.layers), 78)
        self.assertEqual(len(config.mtp_layers), 1)
        self.assertIsInstance(config.mtp_layers[0], Glm5MTPTransformerBlock.Config)
        # GLM-5.2 stores the MTP block as HF layer 78 and shares the top-level
        # token embedding and lm_head with the 78-layer target decoder.
        self.assertEqual(
            config.mtp_layers[0].eh_proj.in_features,
            2 * config.dim,
        )
        self.assertEqual(config.mtp_layers[0].eh_proj.out_features, config.dim)

    def test_mtp_recipe_propagates_depth_to_dataloader(self) -> None:
        from torchtitan_recipes.tests.models.glm5 import glm5_2_debugmodel_mtp

        config = glm5_2_debugmodel_mtp()
        self.assertEqual(config.dataloader.num_mtp_layers, 1)

    def test_mtp_recipe_exposes_loss_scale(self) -> None:
        from torchtitan.models.deepseek_v3.mtp import MTPCrossEntropyLoss as MTPLoss
        from torchtitan_recipes.tests.models.glm5 import (
            glm5_2_debugmodel_mtp,
            glm5_debugmodel,
        )

        config = glm5_2_debugmodel_mtp(mtp_scale=0.125)
        loss_config = config.loss.loss_fn
        self.assertIsInstance(loss_config, MTPLoss.Config)
        self.assertEqual(loss_config.mtp_scale, 0.125)

        with self.assertRaisesRegex(ValueError, "only be set for a GLM MTP"):
            glm5_debugmodel(mtp_scale=0.125)
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            glm5_2_debugmodel_mtp(mtp_scale=-0.1)

    def test_deepep_recipe_replaces_only_the_dispatcher(self) -> None:
        from torchtitan_recipes.tests.models.glm5 import glm5_2_debugmodel_deepep

        config = glm5_2_debugmodel_deepep(seq_len=128)
        self.assertEqual(config.parallelism.expert_parallel_degree, 4)
        routed_experts = [
            layer.feed_forward.routed_experts
            for layer in config.model.layers
            if isinstance(layer.feed_forward, MoE.Config)
        ]
        self.assertTrue(routed_experts)
        self.assertTrue(
            all(
                isinstance(
                    experts.token_dispatcher,
                    DeepEPTokenDispatcher.Config,
                )
                for experts in routed_experts
            )
        )
        self.assertTrue(
            all(
                experts.token_dispatcher.num_max_tokens_per_rank == 8 * 128
                for experts in routed_experts
            )
        )

        with self.assertRaisesRegex(ValueError, "expert_parallel_degree > 1"):
            glm5_2_debugmodel_deepep(
                parallelism=ParallelismConfig(expert_parallel_degree=1)
            )

    def test_preprocess_inputs_returns_loss_kwargs(self) -> None:
        config = build_model_config(
            "glm5_2_debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=8,
        )
        model = config.build()
        token_loss_weight = torch.tensor(0.5)
        input_dict = {
            "input": torch.arange(8),
            "labels": torch.arange(8),
            "loss_kwargs": {"token_loss_weight": token_loss_weight},
        }

        with patch(
            "torchtitan.models.glm5.model.annotate_input_spmd_types",
            side_effect=lambda _context, inputs, _shardings: inputs,
        ):
            inputs, labels, model_kwargs, loss_kwargs = model.preprocess_inputs(
                input_dict,
                parallelism_context=SimpleNamespace(cp_enabled=False),
                parallelism=ParallelismConfig(),
            )

        torch.testing.assert_close(inputs, torch.arange(8))
        torch.testing.assert_close(labels, torch.arange(8))
        self.assertNotIn("token_loss_weight", model_kwargs)
        torch.testing.assert_close(loss_kwargs["token_loss_weight"], token_loss_weight)

    def test_mtp_flops_include_auxiliary_dsa_indexer(self) -> None:
        config = build_model_config(
            "glm5_2_debugmodel_mtp",
            parallelism=ParallelismConfig(),
            seq_len=8,
        )
        with torch.device("meta"):
            model = config.build()

        total, active = config.get_parameter_counts(model)
        self.assertGreaterEqual(total, active)
        estimator = config.build_flops_estimator(model, seq_len=8)
        one = estimator({"input": torch.ones(8, dtype=torch.int64)})
        two = estimator({"input": torch.ones(16, dtype=torch.int64)})
        self.assertGreater(one, 0)
        self.assertEqual(two, 2 * one)

    def test_cp_selects_official_kv_all_gather_attention(self) -> None:
        eager_config = build_model_config(
            "debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        cp_config = build_model_config(
            "debugmodel",
            parallelism=ParallelismConfig(context_parallel_degree=2),
            seq_len=128,
        )
        self.assertIsInstance(
            eager_config.layers[0].attention.inner_attention,
            MLAAttention.Config,
        )
        self.assertIsInstance(
            cp_config.layers[0].attention.inner_attention,
            Glm5CPMLAAttention.Config,
        )
        sharding_config = cp_config.layers[0].attention.inner_attention.sharding_config
        self.assertEqual(
            set(sharding_config.in_dst_shardings or {}),
            {
                "q_THK",
                "kv_c_normed_TL",
                "k_shared_TR",
                "topk_indices_QS",
            },
        )

    def test_cp_transform_covers_mtp_and_preserves_checkpoint_layout(self) -> None:
        import spmd_types as spmd

        from torchtitan.models.common.decoder_sharding import (
            mla_kv_projection_sharding_config,
        )

        eager = build_model_config(
            "glm5_2_debugmodel_mtp", parallelism=ParallelismConfig(), seq_len=128
        )
        config = build_model_config(
            "glm5_2_debugmodel_mtp",
            parallelism=ParallelismConfig(context_parallel_degree=2),
            seq_len=128,
        )
        for layer in config.layers + list(config.mtp_layers or ()):
            inner = layer.attention.inner_attention
            assert isinstance(inner, Glm5CPMLAAttention.Config)
            self.assertIsInstance(inner, AllGatherKVMLAFlexAttention.Config)
            self.assertEqual(
                inner.wkv_b.sharding_config,
                mla_kv_projection_sharding_config(cp=spmd.R),
            )
            with self.assertRaisesRegex(ValueError, "Glm5SparseAttention"):
                replace(inner, inner_attention=inner)
        with torch.device("meta"):
            expected = eager.build().state_dict()
            actual = config.build().state_dict()
        self.assertEqual(
            {name: tuple(value.shape) for name, value in actual.items()},
            {name: tuple(value.shape) for name, value in expected.items()},
        )

    def test_shared_indexer_layers_do_not_own_redundant_parameters(self) -> None:
        config = build_model_config(
            "glm5_2_debugmodel",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        for layer_idx, source_idx in enumerate(config.index_sources):
            indexer = config.layers[layer_idx].attention.indexer
            self.assertEqual(indexer is not None, layer_idx == source_idx)

        mtp_config = build_model_config(
            "glm5_2_debugmodel_mtp",
            parallelism=ParallelismConfig(),
            seq_len=128,
        )
        self.assertIsNotNone(mtp_config.mtp_layers[0].attention.indexer)


if __name__ == "__main__":
    unittest.main()
