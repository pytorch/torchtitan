# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import io
import sys
import typing
import unittest
from unittest import mock

import pytest
import tyro
from torchtitan.config import ConfigManager, ParallelismConfig, TrainingConfig
from torchtitan.models.deepseek_v3.config_registry import (
    deepseek_v3_debugmodel_hybridep,
    deepseek_v3_debugmodel_minimal_async_ep,
)
from torchtitan.models.llama3.config_registry import llama3_debugmodel_dist_gemm
from torchtitan.models.qwen3.config_registry import qwen3_moe_deepep
from torchtitan.observability.sdc_replayer import SDCReplayer
from torchtitan.trainer import Trainer


class TestConfigManager(unittest.TestCase):
    def test_model_config_args(self):
        """--module and --config together load the correct config."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"
        assert config.training.steps == 10

    def test_model_config_args_equals_form(self):
        """--module=X --config=Y form works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module=llama3", "--config=llama3_debugmodel"]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_parse_args_uses_current_sys_argv(self):
        """parse_args() without args reads sys.argv at call time."""
        config_manager = ConfigManager()
        argv = ["train.py", "--module", "nonexistent", "--config", "foo"]
        with mock.patch.object(sys, "argv", argv):
            with pytest.raises(ImportError, match="Cannot import module 'nonexistent'"):
                config_manager.parse_args()

    def test_model_without_config_errors(self):
        """--module alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--config is required"):
            config_manager.parse_args(["--module", "llama3"])

    def test_config_without_model_errors(self):
        """--config alone raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args(["--config", "llama3_debugmodel"])

    def test_missing_both_errors(self):
        """No --module or --config raises ValueError."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="--module is required"):
            config_manager.parse_args([])

    def test_torchtitan_recipes_package_resolves(self):
        """torchtitan_recipes is importable and its configs load."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan_recipes.tests.features",
                "--config",
                "llama3_debugmodel_fsdp2_cp2",
            ]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"
        assert config.parallelism.context_parallel_degree == 2

    def test_invalid_model_errors(self):
        """--module with unknown module name raises ImportError."""
        config_manager = ConfigManager()
        with pytest.raises(ImportError, match="Cannot import module"):
            config_manager.parse_args(["--module", "nonexistent", "--config", "foo"])

    def test_invalid_config_errors(self):
        """--config with unknown function name lists available functions."""
        config_manager = ConfigManager()
        with pytest.raises(ValueError, match="Available config functions"):
            config_manager.parse_args(["--module", "llama3", "--config", "nonexistent"])

    def test_rl_examples_registered_as_shorthands(self):
        """RL examples are valid --module shorthands (resolved under rl/examples).

        End-to-end resolution + build is covered by the RL integration tests
        (they run ``--module alphabet_sort``); kept out of here since importing an
        example's config_registry pulls in vLLM, which isn't available on CPU.
        """
        from torchtitan.experiments import _supported_experiments

        assert "alphabet_sort" in _supported_experiments
        assert "search_r1" in _supported_experiments

    def test_cli_overrides(self):
        """CLI args override config defaults."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.steps",
                "5",
                "--training.num_tokens_per_microbatch_per_dp_rank",
                "4096",
                "--training.num_tokens_per_train_step",
                "8192",
                "--training.max_context_length",
                "1024",
            ]
        )
        assert config.training.steps == 5
        assert config.training.num_tokens_per_microbatch_per_dp_rank == 4096
        assert config.training.num_tokens_per_train_step == 8192
        assert config.training.max_context_length == 1024

    def test_num_tokens_per_microbatch_must_be_positive(self):
        config_manager = ConfigManager()
        with pytest.raises(SystemExit):
            config_manager.parse_args(
                [
                    "--module",
                    "llama3",
                    "--config",
                    "llama3_debugmodel",
                    "--training.num_tokens_per_microbatch_per_dp_rank",
                    "0",
                ]
            )

    def test_num_tokens_per_train_step_must_be_positive_or_unset(self):
        with pytest.raises(ValueError, match="must be -1 or greater than 0"):
            TrainingConfig(num_tokens_per_train_step=0)

    def test_max_context_length_must_be_positive(self):
        for max_context_length in (0, -1):
            with pytest.raises(ValueError, match="must be greater than 0"):
                TrainingConfig(max_context_length=max_context_length)

    def test_num_pp_microbatches_does_not_constrain_non_pp_training(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--parallelism.num_pp_microbatches",
                "3",
            ]
        )
        assert config.parallelism.pipeline_parallel_degree == 1
        assert config.parallelism.num_pp_microbatches == 3

    def test_cuda_graphs_reject_pipeline_parallelism(self):
        config_manager = ConfigManager()
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with pytest.raises((ValueError, SystemExit)) as exc_info:
                config_manager.parse_args(
                    [
                        "--module",
                        "llama3",
                        "--config",
                        "llama3_debugmodel",
                        "--parallelism.pipeline_parallel_degree",
                        "2",
                    ]
                )

        if isinstance(exc_info.value, SystemExit):
            assert exc_info.value.code == 2
            error = stderr.getvalue()
        else:
            error = str(exc_info.value)
        assert "do not support pipeline parallelism" in error

    def test_cuda_graphs_enabled_by_default(self):
        config = ConfigManager().parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert not config.training.disable_cuda_graphs

    def test_cuda_graphs_reject_unsupported_expert_parallelism(self):
        config_manager = ConfigManager()
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            with pytest.raises((ValueError, SystemExit)) as exc_info:
                config_manager.parse_args(
                    [
                        "--module",
                        "deepseek_v3",
                        "--config",
                        "deepseek_v3_debugmodel",
                        "--parallelism.expert_parallel_degree",
                        "2",
                    ]
                )

        if isinstance(exc_info.value, SystemExit):
            assert exc_info.value.code == 2
            error = stderr.getvalue()
        else:
            error = str(exc_info.value)
        assert "without CPU synchronization" in error

    def test_cuda_graphs_allow_non_blocking_hybridep(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "deepseek_v3",
                "--config",
                "deepseek_v3_debugmodel_hybridep",
                "--parallelism.expert_parallel_degree",
                "2",
            ]
        )
        assert not config.training.disable_cuda_graphs

    def test_disable_cuda_graphs_allows_pipeline_parallelism(self):
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.disable_cuda_graphs",
                "--parallelism.pipeline_parallel_degree",
                "2",
            ]
        )
        assert config.training.disable_cuda_graphs

    def test_sdc_replay_requires_determinism(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--training.disable_cuda_graphs",
            ]
        )
        config.sdc_replayer = SDCReplayer.Config()

        with pytest.raises(ValueError, match="debug.deterministic=True"):
            config._validate_sdc_replay()

        config.debug.deterministic = True
        config.debug.deterministic_warn_only = True
        with pytest.raises(ValueError, match="deterministic_warn_only=False"):
            config._validate_sdc_replay()

    def test_sdc_replay_is_off_the_cli(self):
        hints = typing.get_type_hints(Trainer.Config, include_extras=True)
        assert tyro.conf.Suppress in hints["sdc_replayer"].__metadata__

        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--debug.deterministic",
                "--training.disable_cuda_graphs",
            ]
        )
        config.sdc_replayer = SDCReplayer.Config(num_steps=3, num_replays=2)
        config._validate_sdc_replay()

    def test_sdc_replay_rejects_multiple_replays_with_cuda_graphs(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--debug.deterministic",
            ]
        )
        config.sdc_replayer = SDCReplayer.Config(num_replays=2)

        with pytest.raises(ValueError, match="at most one replay"):
            config._validate_sdc_replay()

    def test_sdc_replay_allows_multiple_replays_without_cuda_graphs(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--debug.deterministic",
                "--training.disable_cuda_graphs",
            ]
        )
        config.sdc_replayer = SDCReplayer.Config(num_replays=2)

        config._validate_sdc_replay()

    def test_sdc_replay_accepts_execution_modes(self):
        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--debug.deterministic",
            ]
        )
        config.sdc_replayer = SDCReplayer.Config()
        config.parallelism.enable_fsdp_symm_mem = True
        config.compile.enable_async_tensor_parallel = True
        configs = {
            "symm_mem_async_tp": config,
            "distributed_gemm": llama3_debugmodel_dist_gemm(seq_len=2048),
            "hybrid_ep": deepseek_v3_debugmodel_hybridep(seq_len=2048),
            "minimal_async_ep": deepseek_v3_debugmodel_minimal_async_ep(seq_len=2048),
            "deep_ep": qwen3_moe_deepep(seq_len=512),
        }

        for name, config in configs.items():
            with self.subTest(config=name):
                config.debug.deterministic = True
                config.sdc_replayer = SDCReplayer.Config()
                config._validate_sdc_replay()

    def test_cuda_graphs_reject_blocking_hybridep(self):
        from torchtitan.models.common.token_dispatcher import HybridEPTokenDispatcher
        from torchtitan.models.deepseek_v3.config_registry import (
            deepseek_v3_debugmodel_hybridep,
        )

        config = deepseek_v3_debugmodel_hybridep(seq_len=2048)
        dispatcher_configs = list(
            config.model_spec.model.traverse(HybridEPTokenDispatcher.Config)
        )
        assert dispatcher_configs
        for _, dispatcher_config, _, _ in dispatcher_configs:
            dispatcher_config.non_blocking_capacity_factor = None
        config.parallelism.expert_parallel_degree = 2

        with pytest.raises(ValueError, match="non_blocking_capacity_factor"):
            dataclasses.replace(config)

    def test_cli_override_dump_folder(self):
        """CLI args override config defaults for nested fields."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--dump_folder",
                "/tmp/test_tt/",
            ]
        )
        assert config.dump_folder == "/tmp/test_tt/"

    def test_parse_module_fqns_per_model_part(self):
        """module_fqns_per_model_part defaults to None."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.parallelism.module_fqns_per_model_part is None

    def test_parse_exclude_from_loading(self):
        """exclude_from_loading defaults to [] and can be overridden."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert config.checkpoint.exclude_from_loading == []

        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--checkpoint.exclude_from_loading",
                "optimizer,lr_scheduler",
            ]
        )
        assert config.checkpoint.exclude_from_loading == [
            "optimizer",
            "lr_scheduler",
        ]

    def test_concrete_checkpoint_fields_remain_overridable(self):
        from torchtitan.components.checkpointer import CheckpointManager

        config = ConfigManager().parse_args(
            [
                "--module",
                "llama3",
                "--config",
                "llama3_debugmodel",
                "--checkpoint.async_mode",
                "async",
            ]
        )

        assert isinstance(config.checkpoint, CheckpointManager.Config)
        assert config.checkpoint.async_mode == "async"

    def test_trainer_config_quantization_default(self):
        from torchtitan.components.quantization.utils import has_quantization

        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "llama3", "--config", "llama3_debugmodel"]
        )
        assert not has_quantization(config.model_spec.model)

    # TODO: remove this test when we remove the merge functionality
    def test_extend_trainer_config_directly(self):
        """Test that _merge_configs works to extend config types."""
        from dataclasses import dataclass

        from torchtitan.trainer import Trainer

        @dataclass
        class CustomCheckpoint:
            convert_path: str = "/custom/path"
            fake_model: bool = True

        @dataclass
        class CustomTrainerConfig:
            checkpoint: CustomCheckpoint

        MergedTrainerConfig = ConfigManager._merge_configs(
            Trainer.Config, CustomTrainerConfig
        )

        # Verify the merged type has both base and custom fields
        merged = MergedTrainerConfig()
        assert hasattr(merged, "checkpoint")
        assert hasattr(merged.checkpoint, "convert_path")
        assert merged.checkpoint.convert_path == "/custom/path"
        assert merged.checkpoint.fake_model is True
        assert hasattr(merged, "model_spec")

    def test_flux_config_via_cli(self):
        """Test that --module flux --config flux_debugmodel works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "flux", "--config", "flux_debugmodel"]
        )
        assert config.model_spec.name == "flux"
        assert hasattr(config, "encoder")
        assert config.parallelism.context_parallel_load_balancer == "headtail"

    def test_default_context_parallel_load_balancer(self):
        assert ParallelismConfig().context_parallel_load_balancer == "headtail"

    def test_deepseek_config(self):
        """Test that --module deepseek_v3 --config deepseek_v3_debugmodel works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            ["--module", "deepseek_v3", "--config", "deepseek_v3_debugmodel"]
        )
        assert config.model_spec.name == "deepseek_v3"
        assert config.model_spec.flavor == "debugmodel"

    def test_fqn_module_with_config_registry(self):
        """--module torchtitan.models.llama3.config_registry works."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan.models.llama3.config_registry",
                "--config",
                "llama3_debugmodel",
            ]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_fqn_module_without_config_registry(self):
        """--module torchtitan.models.llama3 (auto-appends .config_registry)."""
        config_manager = ConfigManager()
        config = config_manager.parse_args(
            [
                "--module",
                "torchtitan.models.llama3",
                "--config",
                "llama3_debugmodel",
            ]
        )
        assert config.model_spec.name == "llama3"
        assert config.model_spec.flavor == "debugmodel"

    def test_fqn_module_invalid_errors(self):
        """--module with invalid FQN raises ImportError."""
        config_manager = ConfigManager()
        with pytest.raises(ImportError, match="Cannot import module"):
            config_manager.parse_args(
                [
                    "--module",
                    "torchtitan.models.nonexistent",
                    "--config",
                    "foo",
                ]
            )


if __name__ == "__main__":
    unittest.main()
