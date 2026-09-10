# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.checkpointer.torch_checkpointing import (
    TorchCheckpointingManager,
)
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.config.manager import ConfigManager
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.trainer import Trainer

from torchtitan_recipes.dcpv2_512.config_registry import (
    dsv3_671b_save_dcp,
    dsv3_671b_save_dcpv2,
    dsv3_debug_save_dcp,
    dsv3_debug_save_dcpv2,
)


class TestDCPv2CheckpointProducerConfigs(unittest.TestCase):
    def test_producer_pair_is_matched(self):
        cases = [
            (dsv3_671b_save_dcp, CheckpointManager.Config),
            (dsv3_671b_save_dcpv2, TorchCheckpointingManager.Config),
        ]

        for factory, checkpoint_config_type in cases:
            with self.subTest(factory=factory.__name__):
                config = factory()
                parallel_dims = ParallelDims.from_config(
                    config.parallelism, world_size=256
                )
                self.assertEqual(
                    (
                        parallel_dims.pp,
                        parallel_dims.dp_replicate,
                        parallel_dims.dp_shard,
                        parallel_dims.cp,
                        parallel_dims.tp,
                        parallel_dims.ep,
                    ),
                    (1, 1, 256, 1, 1, 64),
                )
                self.assertEqual(
                    parallel_dims.dp_shard
                    * parallel_dims.cp
                    * parallel_dims.tp
                    // parallel_dims.ep,
                    4,
                )

                self.assertEqual(config.training.dtype, "bfloat16")
                self.assertEqual(
                    config.training.num_tokens_per_microbatch_per_dp_rank,
                    4 * 4096,
                )
                self.assertEqual(config.training.max_context_length, 4096)
                self.assertEqual(config.training.steps, 151)
                self.assertTrue(config.training.disable_cuda_graphs)
                self.assertEqual(config.lr_scheduler.total_steps, 10000)
                self.assertEqual(config.debug.seed, 42)
                self.assertTrue(config.debug.deterministic)
                self.assertFalse(config.debug.deterministic_warn_only)
                self.assertTrue(config.debug.moe_force_load_balance)
                self.assertIsInstance(config.activation_checkpoint, FullAC.Config)
                assert isinstance(config.dataloader, GrainDataLoader.Config)
                dataset = config.dataloader.dataset
                assert isinstance(dataset, ConcatThenSplitPackingConfig)
                source_dataset = dataset.dataset
                assert isinstance(source_dataset, SingleDatasetConfig)
                source = source_dataset.source
                assert isinstance(source, HuggingFaceStreamingSource.Config)
                self.assertEqual(source.path, "/mnt/mffuse/c4")

                self.assertIsInstance(config.checkpoint, checkpoint_config_type)
                self.assertTrue(config.checkpoint.enable)
                self.assertEqual(config.checkpoint.interval, 50)
                self.assertEqual(config.checkpoint.keep_latest_k, 2)
                self.assertTrue(config.checkpoint.last_save_model_only)
                self.assertTrue(config.checkpoint.last_save_in_hf)
                self.assertIsNone(config.checkpoint.initial_load_path)
                self.assertFalse(config.checkpoint.initial_load_model_only)
                self.assertFalse(config.checkpoint.load_only)
                self.assertEqual(config.checkpoint.export_dtype, "bfloat16")

    def test_dcp_producer_uses_pinned_memory_async_saves(self):
        config = dsv3_671b_save_dcp()

        assert isinstance(config.checkpoint, CheckpointManager.Config)
        self.assertEqual(config.checkpoint.async_mode, "async_with_pinned_mem")

    def test_smoke_pair_exercises_periodic_and_final_saves(self):
        cases = [
            (dsv3_debug_save_dcp, CheckpointManager.Config),
            (dsv3_debug_save_dcpv2, TorchCheckpointingManager.Config),
        ]

        for factory, checkpoint_config_type in cases:
            with self.subTest(factory=factory.__name__):
                config = factory()
                parallel_dims = ParallelDims.from_config(
                    config.parallelism, world_size=8
                )
                self.assertEqual(
                    (
                        parallel_dims.dp_replicate,
                        parallel_dims.dp_shard,
                        parallel_dims.tp,
                        parallel_dims.ep,
                    ),
                    (1, 8, 1, 2),
                )
                self.assertEqual(
                    parallel_dims.dp_shard
                    * parallel_dims.cp
                    * parallel_dims.tp
                    // parallel_dims.ep,
                    4,
                )
                self.assertEqual(config.training.dtype, "bfloat16")
                self.assertEqual(config.training.steps, 3)
                self.assertTrue(config.training.disable_cuda_graphs)
                self.assertEqual(config.debug.seed, 42)
                self.assertTrue(config.debug.deterministic)
                self.assertTrue(config.debug.moe_force_load_balance)
                self.assertIsInstance(config.activation_checkpoint, FullAC.Config)
                self.assertIsInstance(config.checkpoint, checkpoint_config_type)
                self.assertTrue(config.checkpoint.enable)
                self.assertEqual(config.checkpoint.interval, 2)
                self.assertFalse(config.checkpoint.initial_load_model_only)
                self.assertTrue(config.checkpoint.last_save_model_only)
                self.assertTrue(config.checkpoint.last_save_in_hf)
                self.assertEqual(config.checkpoint.export_dtype, "bfloat16")

    def test_all_recipes_pass_config_manager_validation(self):
        config_names = (
            "dsv3_671b_save_dcp",
            "dsv3_671b_save_dcpv2",
            "dsv3_debug_save_dcp",
            "dsv3_debug_save_dcpv2",
        )

        for config_name in config_names:
            with self.subTest(config_name=config_name):
                config = ConfigManager().parse_args(
                    [
                        "--module",
                        "torchtitan_recipes.dcpv2_512",
                        "--config",
                        config_name,
                    ]
                )
                assert isinstance(config, Trainer.Config)
                self.assertIsNotNone(config.model_spec)
