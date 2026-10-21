# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.models.common.flops import get_nparams_and_active_nparams
from torchtitan.models.common.moe import MoE
from torchtitan.models.glm5 import build_model_config, MODEL_FLAVORS
from torchtitan.models.glm5.dsa import Glm5AllGatherKVFlexAttention, Glm5FlexAttention
from torchtitan.models.glm5.model import Glm5Attention
from torchtitan.models.glm5.mtp import Glm5MTPTransformerBlock


class TestGlm5Config(unittest.TestCase):
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
        from torchtitan.models.deepseek_v3.mtp import MTPLoss
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

        nparams, active_nparams = get_nparams_and_active_nparams(model)
        indexer_nparams = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if ".attention.indexer." in name
        )
        attention = config.layers[0].attention
        expected_flops = 6 * active_nparams - 4 * indexer_nparams
        expected_flops += (
            2
            * (len(config.layers) + len(config.mtp_layers))
            * attention.indexer.n_heads
            * (attention.indexer.head_dim + 1)
            * 8
        )

        self.assertEqual(
            config.get_nparams_and_flops(model, seq_len=8),
            (nparams, expected_flops),
        )

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
            Glm5FlexAttention.Config,
        )
        self.assertIsInstance(
            cp_config.layers[0].attention.inner_attention,
            Glm5AllGatherKVFlexAttention.Config,
        )
        sharding_config = cp_config.layers[0].attention.inner_attention.sharding_config
        self.assertEqual(
            set(sharding_config.in_dst_shardings or {}),
            {
                "q_THK",
                "k_THK",
                "v_THV",
                "attention_metadata",
                "topk_indices_QS",
            },
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
