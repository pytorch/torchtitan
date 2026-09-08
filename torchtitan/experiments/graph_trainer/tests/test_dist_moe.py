# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration and CUDA-graph fixtures for GraphTrainer DistMoE."""

import unittest
from collections.abc import Callable
from unittest.mock import patch

import torch
from dist_moe import BlockScaledFormat

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.quantization import MXFP8LinearConverter
from torchtitan.experiments.graph_trainer.deepseek_v3.config_registry import (
    graph_trainer_deepseek_v3_16b_dist_moe_bf16,
    graph_trainer_deepseek_v3_16b_dist_moe_mxfp8,
    graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf,
    graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_16gpu,
    graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_64gpu,
    graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_64gpu_coda,
    graph_trainer_deepseek_v3_671b_dist_moe_bf16,
    graph_trainer_deepseek_v3_671b_dist_moe_mxfp8,
    graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf,
    graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf_256gpu,
    graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf_64gpu,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.attention import VarlenAttention
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.router_gate import RouterGateLinear
from torchtitan.models.deepseek_v3.config_registry import enable_mlperf_packing


def graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_cudagraph_test() -> (
    GraphTrainer.Config
):
    """Build the DSV3 16B MXFP8 CUDA-graph test config.

    Returns:
        MLPerf-packed GraphTrainer configuration with forced routing and
        required capture.
    """
    config = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8()
    config.training.num_tokens_per_microbatch_per_dp_rank = 4096
    config.training.num_tokens_per_train_step = -1
    enable_mlperf_packing(config)
    config.debug.moe_force_load_balance = True
    # The 128-token test row needs four chunks so each MXFP8 lm_head GEMM has
    # the 32 rows required by its column quantizer. Production keeps eight.
    config.loss.num_chunks = 4
    return config


class DistMoeGraphTrainerConfigTest(unittest.TestCase):
    """Validate the production DistMoE GraphTrainer configurations."""

    def _assert_policy(
        self,
        factory: Callable[[], GraphTrainer.Config],
        *,
        num_moe_layers: int,
        blockscaled_format: BlockScaledFormat | None,
        q_projection: str,
        main_mxfp8_policy: bool,
    ) -> None:
        """Validate one backend config and every MoE layer.

        Args:
            factory: GraphTrainer configuration factory.
            num_moe_layers: Expected number of routed-expert layers.
            blockscaled_format: Expected format, or ``None`` for BF16.
            q_projection: Query projection mutated by fused Q RoPE.
            main_mxfp8_policy: Whether to expect the Main MXFP8 policies.
        """
        config = factory()
        self.assertEqual("float32", config.training.dtype)
        self.assertEqual("bfloat16", config.training.mixed_precision_param)
        self.assertEqual("bfloat16", config.training.mixed_precision_reduce)
        self.assertEqual("fused_opt_states_bf16", config.optimizer.implementation)
        self.assertEqual("deepseek_v3", config.model_spec.name)
        self.assertIsNotNone(config.model_spec.post_parallelize_fn)
        self.assertIsNotNone(config.model_spec.cleanup_fn)
        self.assertEqual(
            ["layers.*.moe.routed_experts"],
            config.compile.fsdp_contiguous_module_fqns,
        )
        self.assertEqual(config.compile.components, [])
        self.assertEqual(config.compile.inductor_compilation, "none")
        self.assertEqual(config.compile.memory_policy, "full")
        self.assertEqual(
            config.compile.force_recompute_mm_shapes_by_fqns,
            [q_projection],
        )
        self.assertTrue(config.compile.require_cudagraph)
        self.assertNotIn("cudagraph_pass", config.compile.disable_passes)
        self.assertIn(
            "torchtitan.overrides.fused_mla.fused_mla",
            config.override.imports,
        )
        dense_swiglu = "torchtitan.overrides.fused_swiglu.fused_swiglu"
        grouped_swiglu = "torchtitan.overrides.fused_swiglu.fused_grouped_experts"
        if main_mxfp8_policy:
            self.assertEqual(1, config.override.imports.count(dense_swiglu))
            self.assertNotIn(grouped_swiglu, config.override.imports)
        else:
            self.assertNotIn(dense_swiglu, config.override.imports)
        for layer in config.model_spec.model.layers:
            attention = layer.attention.inner_attention
            self.assertIsInstance(attention, VarlenAttention.Config)
            self.assertEqual(attention.max_num_documents, 512)
            self.assertFalse(attention.single_document_rows)
            if layer.moe is not None:
                gate = layer.moe.router.gate
                self.assertIsInstance(gate, RouterGateLinear.Config)
                self.assertEqual(gate.forward.input_dtype, torch.bfloat16)
                self.assertEqual(gate.forward.compute_mode, "bf16")
                self.assertEqual(gate.forward.output_dtype, torch.float32)
                self.assertEqual(gate.backward.input_dtype, torch.float32)
                self.assertEqual(gate.backward.compute_mode, "tf32")
                self.assertEqual(gate.backward.output_dtype, torch.float32)
        experts = [
            expert_config
            for _fqn, expert_config, _parent, _attr in config.traverse(
                DistMoeRoutedExperts.Config
            )
        ]
        self.assertEqual(num_moe_layers, len(experts))
        for expert_config in experts:
            if blockscaled_format is None:
                self.assertIsNone(expert_config.backend.blockscaled)
            else:
                self.assertIsNotNone(expert_config.backend.blockscaled)
                self.assertEqual(
                    blockscaled_format,
                    expert_config.backend.blockscaled.format,
                )
                self.assertTrue(expert_config.backend.blockscaled.fast_math)
                self.assertEqual(expert_config.backend.blockscaled.pipeline, "staged")
            if main_mxfp8_policy:
                self.assertEqual(
                    "maximum_useful",
                    expert_config.backend.device_memory_budget_bytes,
                )
                self.assertIsNone(
                    expert_config.backend.vmm_host_scratch_imbalance_factor
                )
            else:
                self.assertIsNone(expert_config.backend.device_memory_budget_bytes)
                self.assertEqual(
                    "auto",
                    expert_config.backend.vmm_host_scratch_imbalance_factor,
                )

    def test_bf16_configs_replace_all_16b_and_671b_experts(self) -> None:
        """BF16 policy replaces both stock and converted expert configs."""
        self._assert_policy(
            graph_trainer_deepseek_v3_16b_dist_moe_bf16,
            num_moe_layers=26,
            blockscaled_format=None,
            q_projection="attention.wq",
            main_mxfp8_policy=False,
        )
        self._assert_policy(
            graph_trainer_deepseek_v3_671b_dist_moe_bf16,
            num_moe_layers=58,
            blockscaled_format=None,
            q_projection="attention.wq_b",
            main_mxfp8_policy=False,
        )

    def test_mxfp8_configs_replace_all_16b_and_671b_experts(self) -> None:
        """MXFP8 policy replaces both stock and converted expert configs."""
        with (
            patch.object(MXFP8LinearConverter, "__init__", return_value=None),
            patch.object(
                MXFP8LinearConverter,
                "convert",
                side_effect=lambda config: config,
            ),
        ):
            self._assert_policy(
                graph_trainer_deepseek_v3_16b_dist_moe_mxfp8,
                num_moe_layers=26,
                blockscaled_format=BlockScaledFormat.MXFP8_E4M3,
                q_projection="attention.wq",
                main_mxfp8_policy=True,
            )
            self._assert_policy(
                graph_trainer_deepseek_v3_671b_dist_moe_mxfp8,
                num_moe_layers=58,
                blockscaled_format=BlockScaledFormat.MXFP8_E4M3,
                q_projection="attention.wq_b",
                main_mxfp8_policy=True,
            )

    def test_mxfp8_mlperf_configs_enable_dense_attention(self) -> None:
        """MLPerf GraphTrainer configs select dense attention safely."""
        with (
            patch.object(MXFP8LinearConverter, "__init__", return_value=None),
            patch.object(
                MXFP8LinearConverter,
                "convert",
                side_effect=lambda config: config,
            ),
        ):
            configs = [
                graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf(),
                graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf(),
            ]
        for config in configs:
            self.assertFalse(config.dataloader.dataset.mask_document_boundaries)
            self.assertTrue(config.debug.moe_force_load_balance)
            attentions = [
                attention
                for _, attention, _, _ in config.traverse(VarlenAttention.Config)
            ]
            self.assertGreater(len(attentions), 0)
            self.assertTrue(
                all(attention.single_document_rows for attention in attentions)
            )
            self.assertTrue(
                all(
                    attention.max_num_documents
                    == config.training.num_tokens_per_microbatch_per_dp_rank
                    // config.training.max_context_length
                    for attention in attentions
                )
            )

    def test_671b_mxfp8_mlperf_64gpu_config(self) -> None:
        """The 64-GPU recipe fixes topology before configuring packing."""
        with (
            patch.object(MXFP8LinearConverter, "__init__", return_value=None),
            patch.object(
                MXFP8LinearConverter,
                "convert",
                side_effect=lambda config: config,
            ),
        ):
            config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf_64gpu()

        self.assertEqual(
            4096,
            config.training.num_tokens_per_microbatch_per_dp_rank,
        )
        self.assertEqual(4096 * 4096, config.training.num_tokens_per_train_step)
        self.assertTrue(config.compile.numerics_changing_optim)
        self.assertIsInstance(config.loss, CrossEntropyLoss.Config)
        self.assertEqual(1, config.parallelism.data_parallel_replicate_degree)
        self.assertEqual(64, config.parallelism.data_parallel_shard_degree)
        self.assertEqual(1, config.parallelism.tensor_parallel_degree)
        self.assertEqual(1, config.parallelism.context_parallel_degree)
        self.assertEqual(1, config.parallelism.pipeline_parallel_degree)
        self.assertEqual(64, config.parallelism.expert_parallel_degree)
        self.assertIs(config.dataloader.dataset.dataset, DATASETS["c4_test"])
        self.assertFalse(config.dataloader.shuffle)
        self.assertTrue(config.dataloader.repeat)
        attentions = [
            attention for _, attention, _, _ in config.traverse(VarlenAttention.Config)
        ]
        self.assertGreater(len(attentions), 0)
        self.assertTrue(
            all(
                attention.single_document_rows and attention.max_num_documents == 1
                for attention in attentions
            )
        )

    def test_16b_mxfp8_mlperf_16gpu_config(self) -> None:
        """The 16-GPU proxy matches the target accumulation and eFSDP degree."""
        with (
            patch.object(MXFP8LinearConverter, "__init__", return_value=None),
            patch.object(
                MXFP8LinearConverter,
                "convert",
                side_effect=lambda config: config,
            ),
        ):
            config = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_16gpu()

        self.assertEqual(
            4096,
            config.training.num_tokens_per_microbatch_per_dp_rank,
        )
        self.assertEqual(256 * 4096, config.training.num_tokens_per_train_step)
        self.assertEqual(4096, config.training.max_context_length)
        self.assertIsNone(config.activation_checkpoint)
        self.assertTrue(config.compile.numerics_changing_optim)
        self.assertIsInstance(config.loss, CrossEntropyLoss.Config)
        self.assertEqual(1, config.parallelism.data_parallel_replicate_degree)
        self.assertEqual(16, config.parallelism.data_parallel_shard_degree)
        self.assertEqual(1, config.parallelism.tensor_parallel_degree)
        self.assertEqual(1, config.parallelism.context_parallel_degree)
        self.assertEqual(1, config.parallelism.pipeline_parallel_degree)
        self.assertEqual(4, config.parallelism.expert_parallel_degree)
        self.assertEqual("never", config.parallelism.fsdp_reshard_after_forward)
        self.assertIs(config.dataloader.dataset.dataset, DATASETS["c4_test"])
        self.assertFalse(config.dataloader.shuffle)
        self.assertTrue(config.dataloader.repeat)
        self.assertEqual(
            16,
            config.training.num_tokens_per_train_step
            // config.training.num_tokens_per_microbatch_per_dp_rank
            // config.parallelism.data_parallel_shard_degree,
        )
        self.assertEqual(
            4,
            config.parallelism.data_parallel_shard_degree
            // config.parallelism.expert_parallel_degree,
        )
        self.assertTrue(config.debug.moe_force_load_balance)
        attentions = [
            attention for _, attention, _, _ in config.traverse(VarlenAttention.Config)
        ]
        self.assertGreater(len(attentions), 0)
        self.assertTrue(
            all(
                attention.single_document_rows and attention.max_num_documents == 1
                for attention in attentions
            )
        )

    def test_16b_mxfp8_mlperf_64gpu_coda_pair(self) -> None:
        """The 64-GPU baseline and CODA arm differ only in CODA settings."""
        with (
            patch.object(MXFP8LinearConverter, "__init__", return_value=None),
            patch.object(
                MXFP8LinearConverter,
                "convert",
                side_effect=lambda config: config,
            ),
        ):
            baseline = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_64gpu()
            coda = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_mlperf_64gpu_coda()

        self.assertEqual(64, baseline.parallelism.data_parallel_shard_degree)
        self.assertEqual(64, baseline.parallelism.expert_parallel_degree)
        self.assertEqual(64 * 16 * 4096, baseline.training.num_tokens_per_train_step)
        self.assertEqual("regional", baseline.compile.inductor_compilation)
        self.assertFalse(baseline.compile.enable_coda)
        self.assertTrue(coda.compile.enable_coda)
        self.assertEqual(
            [
                "F_swiglu",
                "B_swiglu_backward_activation",
                "B_parallel_mm_dx_merge",
                "B_mm_dx_residual_add",
                "B_linear_dw_bf16_to_fp32",
            ],
            coda.compile.coda_patterns,
        )

    def test_671b_mxfp8_mlperf_256gpu_config(self) -> None:
        """The 256-GPU recipe configures LBS2 before attention packing."""
        with (
            patch.object(MXFP8LinearConverter, "__init__", return_value=None),
            patch.object(
                MXFP8LinearConverter,
                "convert",
                side_effect=lambda config: config,
            ),
        ):
            config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_mlperf_256gpu()

        self.assertEqual(
            2 * 4096,
            config.training.num_tokens_per_microbatch_per_dp_rank,
        )
        self.assertEqual(4096 * 4096, config.training.num_tokens_per_train_step)
        self.assertTrue(config.compile.numerics_changing_optim)
        self.assertIsInstance(config.loss, CrossEntropyLoss.Config)
        self.assertEqual(1, config.parallelism.data_parallel_replicate_degree)
        self.assertEqual(256, config.parallelism.data_parallel_shard_degree)
        self.assertEqual(1, config.parallelism.tensor_parallel_degree)
        self.assertEqual(1, config.parallelism.context_parallel_degree)
        self.assertEqual(1, config.parallelism.pipeline_parallel_degree)
        self.assertEqual(64, config.parallelism.expert_parallel_degree)
        self.assertIs(config.dataloader.dataset.dataset, DATASETS["c4_test"])
        self.assertFalse(config.dataloader.shuffle)
        self.assertTrue(config.dataloader.repeat)
        data_parallel_degree = (
            config.parallelism.data_parallel_replicate_degree
            * config.parallelism.data_parallel_shard_degree
        )
        self.assertEqual(
            8,
            config.training.num_tokens_per_train_step
            // config.training.num_tokens_per_microbatch_per_dp_rank
            // data_parallel_degree,
        )
        self.assertEqual(
            4,
            config.parallelism.data_parallel_shard_degree
            // config.parallelism.expert_parallel_degree,
        )
        attentions = [
            attention for _, attention, _, _ in config.traverse(VarlenAttention.Config)
        ]
        self.assertGreater(len(attentions), 0)
        self.assertTrue(
            all(
                attention.single_document_rows and attention.max_num_documents == 2
                for attention in attentions
            )
        )

    def test_mxfp8_cudagraph_fixture_uses_context_planning(self) -> None:
        """The capture fixture delegates capacity and recompute to the context."""
        with (
            patch.object(MXFP8LinearConverter, "__init__", return_value=None),
            patch.object(
                MXFP8LinearConverter,
                "convert",
                side_effect=lambda config: config,
            ),
        ):
            config = graph_trainer_deepseek_v3_16b_dist_moe_mxfp8_cudagraph_test()
        self.assertTrue(config.compile.require_cudagraph)
        self.assertTrue(config.debug.moe_force_load_balance)
        backends = [
            expert_config.backend
            for _fqn, expert_config, _parent, _attr in config.traverse(
                DistMoeRoutedExperts.Config
            )
        ]
        self.assertEqual(26, len(backends))
        self.assertTrue(
            all(
                backend.device_memory_budget_bytes == "maximum_useful"
                and backend.vmm_host_scratch_imbalance_factor is None
                for backend in backends
            )
        )
        attentions = [
            attention for _, attention, _, _ in config.traverse(VarlenAttention.Config)
        ]
        self.assertGreater(len(attentions), 0)
        self.assertTrue(all(attention.single_document_rows for attention in attentions))
